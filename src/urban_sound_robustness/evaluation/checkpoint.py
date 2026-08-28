"""Checkpoint validation and deterministic noisy-condition inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset

from urban_sound_robustness.audio import AudioPreprocessor, load_audio
from urban_sound_robustness.evaluation.corruption import (
    DeterministicNoiseCorruptor,
    RobustnessCondition,
    discover_noise_files,
)
from urban_sound_robustness.utils.paths import resolve_project_path

if TYPE_CHECKING:
    from urban_sound_robustness.datasets.records import AudioSampleRecord


RESEARCH_CHECKPOINT_VERSION = 2
SMOKE_LIMIT_KEYS = (
    "max_train_samples",
    "max_validation_samples",
    "max_train_batches",
    "max_validation_batches",
)


class CheckpointEvaluationError(ValueError):
    """Raised when a checkpoint cannot support a research evaluation."""


@dataclass(frozen=True)
class ResearchCheckpoint:
    """Validated model state and provenance needed for test-time inference."""

    path: Path
    experiment_id: str
    epoch: int
    model_class: str
    model_state_dict: Mapping[str, Tensor]
    configuration: Mapping[str, Any]
    best_metric: float | None


@dataclass(frozen=True)
class ConditionPredictionCollection:
    """Predictions and per-sample corruption provenance for one condition."""

    targets: np.ndarray
    predictions: np.ndarray
    sample_ids: tuple[str, ...]
    metadata: pd.DataFrame


def load_research_checkpoint(
    checkpoint_path: str | Path,
    current_configuration: Mapping[str, Any],
    *,
    map_location: str | torch.device = "cpu",
) -> ResearchCheckpoint:
    """Load a full-run best checkpoint and validate its experiment manifest.

    The current evaluation section is intentionally not compared with the saved
    section. This permits a corrected, held-out noise path to be applied to
    checkpoints trained before the external-noise collection was available.
    Dataset, audio, augmentation, model, and training definitions must still
    match exactly.
    """
    resolved = Path(checkpoint_path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {resolved}")
    if resolved.name != "best.pt":
        raise CheckpointEvaluationError(
            "Research evaluation requires the selected run's best.pt checkpoint."
        )

    checkpoint = torch.load(
        resolved,
        map_location=map_location,
        weights_only=False,
    )
    if not isinstance(checkpoint, dict):
        raise CheckpointEvaluationError("Checkpoint root must be a mapping.")
    if int(checkpoint.get("checkpoint_version", 0)) != RESEARCH_CHECKPOINT_VERSION:
        raise CheckpointEvaluationError(
            "Checkpoint does not use the complete version-2 research schema."
        )

    saved_configuration = checkpoint.get("configuration")
    if not isinstance(saved_configuration, dict):
        raise CheckpointEvaluationError(
            "Checkpoint is missing its composed training configuration."
        )
    for section_name in ("dataset", "audio", "augmentation", "model", "training"):
        if saved_configuration.get(section_name) != current_configuration.get(
            section_name
        ):
            raise CheckpointEvaluationError(
                f"Checkpoint and manifest differ in '{section_name}'."
            )

    runtime = saved_configuration.get("runtime_overrides", {})
    if not isinstance(runtime, dict):
        raise CheckpointEvaluationError(
            "Checkpoint runtime_overrides must be a mapping."
        )
    active_limits = {
        key: runtime.get(key)
        for key in SMOKE_LIMIT_KEYS
        if runtime.get(key) is not None
    }
    if active_limits:
        raise CheckpointEvaluationError(
            "Smoke or bounded checkpoints cannot produce research results; "
            f"active limits: {active_limits}."
        )

    model_class = checkpoint.get("model_class")
    model_state_dict = checkpoint.get("model_state_dict")
    if not isinstance(model_class, str) or not model_class:
        raise CheckpointEvaluationError("Checkpoint has no valid model_class.")
    if not isinstance(model_state_dict, Mapping):
        raise CheckpointEvaluationError("Checkpoint has no model_state_dict.")

    saved_best_metric = checkpoint.get("best_metric")
    return ResearchCheckpoint(
        path=resolved,
        experiment_id=resolved.parent.name,
        epoch=int(checkpoint["epoch"]),
        model_class=model_class,
        model_state_dict=model_state_dict,
        configuration=saved_configuration,
        best_metric=(
            None if saved_best_metric is None else float(saved_best_metric)
        ),
    )


def validate_noise_isolation(
    checkpoint_configuration: Mapping[str, Any],
    evaluation_settings: Mapping[str, Any],
    project_root: str | Path,
) -> tuple[Path, tuple[Path, ...]]:
    """Resolve evaluation noise and reject overlap with training noise files."""
    noise_directory = resolve_project_path(
        str(evaluation_settings["noise_directory"]),
        project_root,
    )
    evaluation_noise = tuple(discover_noise_files(noise_directory))

    augmentation = checkpoint_configuration.get("augmentation", {})
    if not isinstance(augmentation, Mapping):
        raise CheckpointEvaluationError(
            "Checkpoint augmentation configuration must be a mapping."
        )
    waveform = augmentation.get("waveform", {})
    background_noise = (
        waveform.get("background_noise", {})
        if isinstance(waveform, Mapping)
        else {}
    )
    if isinstance(background_noise, Mapping) and background_noise.get(
        "enabled", False
    ):
        training_directory = resolve_project_path(
            str(background_noise["noise_directory"]),
            project_root,
        )
        training_noise = set(discover_noise_files(training_directory))
        overlap = training_noise.intersection(evaluation_noise)
        if overlap:
            example = sorted(overlap)[0]
            raise CheckpointEvaluationError(
                "Training and evaluation noise collections overlap. "
                f"Example shared file: {example}"
            )
    return noise_directory, evaluation_noise


class RobustnessEvaluationDataset(Dataset[dict[str, object]]):
    """Create model features after deterministic clean/noisy corruption."""

    def __init__(
        self,
        records: Sequence[AudioSampleRecord],
        preprocessor: AudioPreprocessor,
        corruptor: DeterministicNoiseCorruptor,
        condition: RobustnessCondition,
    ) -> None:
        if not records:
            raise ValueError("Robustness evaluation requires at least one record.")
        self.records = tuple(records)
        self.preprocessor = preprocessor.eval()
        self.corruptor = corruptor
        self.condition = condition

    def __len__(self) -> int:
        """Return the number of clean source records."""
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, object]:
        """Return one corrupted feature tensor and complete provenance."""
        record = self.records[index]
        loaded = load_audio(record.audio_path)
        clean = self.preprocessor.prepare_waveform(
            loaded.waveform,
            loaded.sample_rate,
            training=False,
        )
        corruption = self.corruptor.corrupt(
            clean,
            record.sample_id,
            self.condition,
        )
        features = self.preprocessor.extract_features(corruption.waveform)
        mixture = corruption.mix_result
        return {
            "features": features,
            "label": torch.tensor(record.label, dtype=torch.long),
            "sample_id": record.sample_id,
            "class_name": record.class_name,
            "fold": -1 if record.fold is None else record.fold,
            "condition": self.condition.name,
            "target_snr_db": (
                float("nan")
                if self.condition.snr_db is None
                else float(self.condition.snr_db)
            ),
            "achieved_snr_db": (
                float("nan")
                if mixture is None or mixture.achieved_snr_db is None
                else float(mixture.achieved_snr_db)
            ),
            "noise_path": (
                "" if corruption.noise_path is None else str(corruption.noise_path)
            ),
            "noise_selection_seed": (
                -1
                if corruption.selection_seed is None
                else int(corruption.selection_seed)
            ),
            "noise_applied": False if mixture is None else bool(mixture.applied),
        }


def _batch_values(value: object, expected_length: int, name: str) -> list[object]:
    """Convert one default-collated field to a length-checked Python list."""
    if isinstance(value, Tensor):
        values = value.detach().cpu().tolist()
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        values = [value]
    if len(values) != expected_length:
        raise TypeError(
            f"Batch field '{name}' has {len(values)} values; "
            f"expected {expected_length}."
        )
    return values


def collect_condition_predictions(
    model: nn.Module,
    data_loader: Iterable[Mapping[str, object]],
    device: torch.device,
) -> ConditionPredictionCollection:
    """Run inference once while retaining sample-level corruption provenance."""
    model.eval()
    targets: list[int] = []
    predictions: list[int] = []
    sample_ids: list[str] = []
    metadata_rows: list[dict[str, object]] = []
    provenance_fields = (
        "class_name",
        "fold",
        "condition",
        "target_snr_db",
        "achieved_snr_db",
        "noise_path",
        "noise_selection_seed",
        "noise_applied",
    )

    with torch.inference_mode():
        for batch in data_loader:
            features = batch.get("features")
            labels = batch.get("label")
            if not isinstance(features, Tensor) or not isinstance(labels, Tensor):
                raise TypeError("DataLoader batches require tensor features and label.")
            logits = model(features.to(device))
            batch_predictions = logits.argmax(dim=1).cpu()
            batch_size = int(labels.shape[0])
            batch_ids = _batch_values(batch.get("sample_id"), batch_size, "sample_id")
            field_values = {
                name: _batch_values(batch.get(name), batch_size, name)
                for name in provenance_fields
            }

            targets.extend(int(value) for value in labels.cpu().tolist())
            predictions.extend(int(value) for value in batch_predictions.tolist())
            sample_ids.extend(str(value) for value in batch_ids)
            for index in range(batch_size):
                metadata_rows.append(
                    {
                        name: field_values[name][index]
                        for name in provenance_fields
                    }
                )

    if not targets:
        raise ValueError("Evaluation DataLoader produced no samples.")
    return ConditionPredictionCollection(
        targets=np.asarray(targets, dtype=np.int64),
        predictions=np.asarray(predictions, dtype=np.int64),
        sample_ids=tuple(sample_ids),
        metadata=pd.DataFrame(metadata_rows),
    )
