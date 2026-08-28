"""Tests for safe checkpoint-driven robustness evaluation."""

from copy import deepcopy
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf
import torch
from torch import nn
from torch.utils.data import DataLoader

from urban_sound_robustness.audio import AudioPreprocessor
from urban_sound_robustness.datasets.records import AudioSampleRecord
from urban_sound_robustness.evaluation import (
    CheckpointEvaluationError,
    DeterministicNoiseCorruptor,
    RobustnessCondition,
    RobustnessEvaluationDataset,
    collect_condition_predictions,
    load_research_checkpoint,
    validate_noise_isolation,
)


def _checkpoint_configuration() -> dict:
    """Return the minimum sections required by checkpoint validation."""
    return {
        "dataset": {"name": "fixture"},
        "audio": {"sample_rate": 8_000},
        "augmentation": {
            "name": "baseline",
            "enabled": False,
            "waveform": {"background_noise": {"enabled": False}},
        },
        "model": {"name": "fixture"},
        "training": {"epochs": 2},
        "evaluation": {"noise_directory": "old/path"},
    }


def _write_checkpoint(
    tmp_path: Path,
    configuration: dict,
    *,
    filename: str = "best.pt",
) -> Path:
    """Write a structurally complete lightweight version-2 checkpoint."""
    path = tmp_path / "experiment" / filename
    path.parent.mkdir(parents=True)
    torch.save(
        {
            "checkpoint_version": 2,
            "epoch": 2,
            "model_class": "FixtureModel",
            "model_state_dict": {"weight": torch.ones(1)},
            "best_metric": 0.75,
            "configuration": configuration,
        },
        path,
    )
    return path


def _audio_settings() -> dict:
    """Return a small but complete log-Mel configuration."""
    return {
        "sample_rate": 8_000,
        "clip_duration_seconds": 0.1,
        "mono": True,
        "length_normalization": {
            "padding_mode": "zero",
            "training_crop": "random",
            "evaluation_crop": "center",
        },
        "representation": "log_mel",
        "log_mel": {
            "n_fft": 64,
            "win_length": 64,
            "hop_length": 32,
            "n_mels": 8,
            "f_min": 0.0,
            "f_max": 4_000.0,
            "power": 2.0,
            "top_db": 80.0,
            "center": True,
            "pad_mode": "reflect",
            "mel_scale": "htk",
        },
        "normalization": {
            "method": "per_example_standardization",
            "epsilon": 1.0e-6,
        },
    }


def test_evaluation_package_imports_in_fresh_interpreter() -> None:
    """Public evaluator imports must not depend on package import order."""
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from urban_sound_robustness.evaluation import "
                "load_research_checkpoint, RobustnessEvaluationDataset"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_checkpoint_accepts_corrected_live_evaluation_section(tmp_path: Path) -> None:
    """Evaluation settings may change while all training sections remain fixed."""
    saved = _checkpoint_configuration()
    saved["runtime_overrides"] = {
        "max_train_samples": None,
        "max_validation_samples": None,
        "max_train_batches": None,
        "max_validation_batches": None,
    }
    checkpoint_path = _write_checkpoint(tmp_path, saved)
    current = deepcopy(saved)
    current["evaluation"] = {"noise_directory": "held-out/noise_test"}

    loaded = load_research_checkpoint(checkpoint_path, current)

    assert loaded.experiment_id == "experiment"
    assert loaded.epoch == 2
    assert loaded.best_metric == pytest.approx(0.75)


def test_checkpoint_rejects_smoke_limits(tmp_path: Path) -> None:
    """A bounded training checkpoint must never be reported as research."""
    saved = _checkpoint_configuration()
    saved["runtime_overrides"] = {"max_train_batches": 20}
    checkpoint_path = _write_checkpoint(tmp_path, saved)

    with pytest.raises(CheckpointEvaluationError, match="Smoke or bounded"):
        load_research_checkpoint(checkpoint_path, saved)


def test_checkpoint_requires_best_file(tmp_path: Path) -> None:
    """The evaluator should not silently benchmark last.pt."""
    configuration = _checkpoint_configuration()
    checkpoint_path = _write_checkpoint(
        tmp_path,
        configuration,
        filename="last.pt",
    )

    with pytest.raises(CheckpointEvaluationError, match="best.pt"):
        load_research_checkpoint(checkpoint_path, configuration)


def test_noise_isolation_rejects_shared_training_files(tmp_path: Path) -> None:
    """Evaluation discovery must not include augmentation-training recordings."""
    training_directory = tmp_path / "noise_train"
    test_directory = tmp_path / "noise_test"
    training_directory.mkdir()
    test_directory.mkdir()
    (training_directory / "train.wav").write_bytes(b"fixture")
    (test_directory / "test.wav").write_bytes(b"fixture")
    configuration = {
        "augmentation": {
            "waveform": {
                "background_noise": {
                    "enabled": True,
                    "noise_directory": str(training_directory),
                }
            }
        }
    }

    resolved, files = validate_noise_isolation(
        configuration,
        {"noise_directory": str(test_directory)},
        tmp_path,
    )
    assert resolved == test_directory.resolve()
    assert files == ((test_directory / "test.wav").resolve(),)

    with pytest.raises(CheckpointEvaluationError, match="overlap"):
        validate_noise_isolation(
            configuration,
            {"noise_directory": str(tmp_path)},
            tmp_path,
        )


class _AlwaysZeroModel(nn.Module):
    """Return deterministic two-class logits for collector testing."""

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch_size = features.shape[0]
        logits = torch.zeros((batch_size, 2), device=features.device)
        logits[:, 0] = 1.0
        return logits


def test_condition_dataset_and_collector_preserve_noise_provenance(
    tmp_path: Path,
) -> None:
    """Real mixing metadata should survive the DataLoader inference pass."""
    clean_path = tmp_path / "clean.wav"
    noise_directory = tmp_path / "noise_test"
    noise_directory.mkdir()
    noise_path = noise_directory / "noise.wav"
    time_values = np.linspace(0.0, 0.1, 800, endpoint=False)
    sf.write(clean_path, np.sin(2 * np.pi * 440 * time_values), 8_000)
    sf.write(noise_path, np.cos(2 * np.pi * 170 * time_values), 8_000)
    record = AudioSampleRecord(
        sample_id="clean.wav",
        audio_path=clean_path,
        label=0,
        class_name="zero",
        fold=10,
        dataset_name="fixture",
        metadata={},
    )
    preprocessor = AudioPreprocessor(_audio_settings())
    corruptor = DeterministicNoiseCorruptor(
        noise_directory,
        target_sample_rate=8_000,
        corruption_seed=2025,
    )
    dataset = RobustnessEvaluationDataset(
        [record],
        preprocessor,
        corruptor,
        RobustnessCondition("snr_10db", 10.0),
    )

    item = dataset[0]
    assert item["achieved_snr_db"] == pytest.approx(10.0, abs=1.0e-5)
    assert item["noise_path"] == str(noise_path.resolve())

    collection = collect_condition_predictions(
        _AlwaysZeroModel(),
        DataLoader(dataset, batch_size=1),
        torch.device("cpu"),
    )
    assert collection.targets.tolist() == [0]
    assert collection.predictions.tolist() == [0]
    assert collection.sample_ids == ("clean.wav",)
    assert collection.metadata.loc[0, "condition"] == "snr_10db"
    assert collection.metadata.loc[0, "achieved_snr_db"] == pytest.approx(
        10.0,
        abs=1.0e-5,
    )
