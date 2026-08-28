"""PyTorch dataset integration for validated audio records."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence, TypedDict

import torch
from torch import Tensor
from torch.utils.data import Dataset

from urban_sound_robustness.audio import AudioPreprocessor, load_audio
from urban_sound_robustness.augmentation import (
    AugmentedAudioPreprocessor,
    create_configured_preprocessor,
)
from urban_sound_robustness.datasets.base import AudioDatasetAdapter
from urban_sound_robustness.datasets.records import AudioSampleRecord


class AudioDatasetItem(TypedDict):
    """One model-ready item plus lightweight provenance fields."""

    features: Tensor
    label: Tensor
    sample_id: str
    class_name: str
    fold: int
    source_path: str
    source_sample_rate: int
    source_num_channels: int
    source_num_frames: int


class PreprocessedAudioDataset(Dataset[AudioDatasetItem]):
    """Load and preprocess audio lazily from common dataset records."""

    def __init__(
        self,
        records: Sequence[AudioSampleRecord],
        preprocessor: AudioPreprocessor | AugmentedAudioPreprocessor,
        *,
        training: bool = False,
    ) -> None:
        self.records = tuple(records)
        self.preprocessor = preprocessor
        self.training = training

    def __len__(self) -> int:
        """Return the number of metadata records in this dataset view."""
        return len(self.records)

    def __getitem__(self, index: int) -> AudioDatasetItem:
        """Load one record and return fixed-size features plus provenance."""
        record = self.records[index]
        loaded = load_audio(record.audio_path)
        processed = self.preprocessor(
            loaded.waveform,
            loaded.sample_rate,
            training=self.training,
        )
        return {
            "features": processed.features,
            "label": torch.tensor(record.label, dtype=torch.long),
            "sample_id": record.sample_id,
            "class_name": record.class_name,
            "fold": -1 if record.fold is None else record.fold,
            "source_path": str(record.audio_path),
            "source_sample_rate": processed.source_sample_rate,
            "source_num_channels": processed.source_num_channels,
            "source_num_frames": processed.source_num_frames,
        }


def create_preprocessed_dataset(
    adapter: AudioDatasetAdapter,
    audio_settings: Mapping[str, Any],
    split_name: str,
    *,
    training: bool | None = None,
    augmentation_settings: Mapping[str, Any] | None = None,
    project_root: str | Path | None = None,
    noise_paths: Sequence[str | Path] | None = None,
) -> PreprocessedAudioDataset:
    """Build a configured lazy PyTorch dataset for one official split or all data."""
    if split_name == "all":
        records = adapter.load_records()
    else:
        records = adapter.records_for_split(split_name)

    is_training = split_name == "train" if training is None else training
    if augmentation_settings and augmentation_settings.get("enabled", False):
        if project_root is None:
            raise ValueError(
                "project_root is required when augmentation is enabled."
            )
        preprocessor = create_configured_preprocessor(
            audio_settings,
            augmentation_settings,
            project_root=project_root,
            noise_paths=noise_paths,
        )
    else:
        preprocessor = AudioPreprocessor(audio_settings)
    return PreprocessedAudioDataset(
        records,
        preprocessor,
        training=is_training,
    )
