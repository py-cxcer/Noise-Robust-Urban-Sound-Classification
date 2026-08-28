"""Tests for device, logging, seeding, and experiment-management utilities."""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import random

import numpy as np
import pytest
import torch
import yaml

from urban_sound_robustness.utils.device import select_device
from urban_sound_robustness.utils.experiment import (
    build_experiment_id,
    create_experiment_layout,
    load_experiment_layout,
)
from urban_sound_robustness.utils.logging_utils import configure_logging
from urban_sound_robustness.utils.reproducibility import (
    create_data_loader_generator,
    seed_everything,
)


def test_seed_everything_repeats_random_sequences() -> None:
    """Python, NumPy, and PyTorch should repeat after reseeding."""
    seed_everything(123, deterministic=True)
    first_values = (
        random.random(),
        np.random.random(),
        torch.rand(3),
    )

    seed_everything(123, deterministic=True)
    second_values = (
        random.random(),
        np.random.random(),
        torch.rand(3),
    )

    assert first_values[0] == second_values[0]
    assert first_values[1] == second_values[1]
    assert torch.equal(first_values[2], second_values[2])


def test_data_loader_generators_repeat() -> None:
    """Equal generator seeds should produce equal sampling sequences."""
    first_generator = create_data_loader_generator(42)
    second_generator = create_data_loader_generator(42)

    assert torch.equal(
        torch.randperm(20, generator=first_generator),
        torch.randperm(20, generator=second_generator),
    )


def test_auto_device_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Automatic selection must remain usable on CPU-only machines."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert select_device("auto") == torch.device("cpu")


def test_explicit_cuda_fails_clearly_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An explicit CUDA request should not silently change the methodology."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA was requested"):
        select_device("cuda")


def test_logging_reconfiguration_does_not_duplicate_handlers(tmp_path: Path) -> None:
    """Repeated setup should still write each message once to the active file."""
    first_log = tmp_path / "first.log"
    second_log = tmp_path / "second.log"
    configure_logging(first_log)
    logger = configure_logging(second_log)
    logger.info("one test message")

    for handler in logger.handlers:
        handler.flush()

    assert not first_log.read_text(encoding="utf-8")
    assert second_log.read_text(encoding="utf-8").count("one test message") == 1

    # Close the temporary file handler before pytest removes its directory.
    configure_logging(level="WARNING")


def test_build_experiment_id_is_readable_and_stable() -> None:
    """Explicit timestamps make identifier formatting easy to verify."""
    timestamp = datetime(2026, 8, 20, 8, 30, tzinfo=timezone.utc)

    experiment_id = build_experiment_id(
        "ResNet18",
        "Augmented",
        run_label="Smoke Run",
        timestamp=timestamp,
    )

    assert experiment_id == "resnet18_augmented_smoke_run_20260820t083000z"


def test_experiment_layout_saves_snapshots_without_overwriting(tmp_path: Path) -> None:
    """A run should receive isolated outputs and reject a reused identifier."""
    path_settings = {
        "experiments": "experiments",
        "checkpoints": "checkpoints",
        "logs": "logs",
        "results": "results",
    }
    configuration = {"experiment": {"id": "cnn_baseline_test"}, "seed": 42}

    paths = create_experiment_layout(
        "cnn_baseline_test",
        path_settings,
        tmp_path,
        configuration,
    )

    saved_configuration = yaml.safe_load(
        paths.config_snapshot.read_text(encoding="utf-8")
    )
    saved_environment = json.loads(
        paths.environment_snapshot.read_text(encoding="utf-8")
    )

    assert saved_configuration == configuration
    assert saved_environment["python_executable"] == str(Path(sys_executable()).resolve())
    assert paths.metrics_directory.is_dir()
    assert paths.predictions_directory.is_dir()

    with pytest.raises(FileExistsError, match="would overwrite"):
        create_experiment_layout(
            "cnn_baseline_test",
            path_settings,
            tmp_path,
            configuration,
        )

    reopened = load_experiment_layout(
        "cnn_baseline_test",
        path_settings,
        tmp_path,
    )
    assert reopened == paths


def sys_executable() -> str:
    """Return the current executable while keeping the assertion readable."""
    import sys

    return sys.executable
