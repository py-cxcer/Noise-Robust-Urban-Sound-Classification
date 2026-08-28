"""Create non-overwriting experiment directories and reproducibility snapshots."""

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import platform
from pathlib import Path
import re
import sys
from typing import Any, Mapping

import torch
import yaml

from urban_sound_robustness.utils.paths import resolve_path_settings


@dataclass(frozen=True)
class ExperimentPaths:
    """Filesystem locations allocated to one experiment run."""

    experiment_directory: Path
    checkpoint_directory: Path
    log_directory: Path
    metrics_directory: Path
    figures_directory: Path
    confusion_matrices_directory: Path
    predictions_directory: Path
    config_snapshot: Path
    environment_snapshot: Path


def build_experiment_id(
    architecture: str,
    augmentation_condition: str,
    run_label: str | None = None,
    timestamp: datetime | None = None,
) -> str:
    """
    Construct a readable experiment identifier with a UTC timestamp.

    Parameters
    ----------
    architecture : str
        Model family, for example ``cnn`` or ``resnet18``.
    augmentation_condition : str
        Training condition, normally ``baseline`` or ``augmented``.
    run_label : str or None
        Optional short label such as ``smoke`` or ``fold10``.
    timestamp : datetime or None
        Explicit timestamp for testing. Current UTC time is used when omitted.

    Returns
    -------
    str
        Filesystem-safe experiment identifier.
    """
    parts = [architecture, augmentation_condition]

    if run_label:
        parts.append(run_label)

    safe_parts = [_to_safe_identifier(part) for part in parts]
    current_time = datetime.now(timezone.utc) if timestamp is None else timestamp
    timestamp_text = current_time.astimezone(timezone.utc).strftime("%Y%m%dt%H%M%Sz")
    safe_parts.append(timestamp_text)
    return "_".join(safe_parts)


def create_experiment_layout(
    experiment_id: str,
    path_settings: Mapping[str, str | Path],
    project_root: str | Path,
    resolved_configuration: Mapping[str, Any],
) -> ExperimentPaths:
    """
    Create isolated output directories and record the resolved environment.

    Parameters
    ----------
    experiment_id : str
        Unique filesystem-safe run identifier.
    path_settings : Mapping[str, str or Path]
        Configured experiments, checkpoints, logs, and results roots.
    project_root : str or Path
        Repository root used to resolve relative paths.
    resolved_configuration : Mapping[str, Any]
        Complete configuration to save with the run.

    Returns
    -------
    ExperimentPaths
        All created directories and snapshot file paths.

    Raises
    ------
    ValueError
        If the ID is unsafe or a required root is missing.
    FileExistsError
        If any run directory exists, preventing accidental overwrite.
    """
    if not experiment_id or experiment_id != _to_safe_identifier(experiment_id):
        raise ValueError(
            "Experiment IDs may contain lowercase letters, numbers, underscores, "
            "and hyphens only."
        )

    resolved_roots = resolve_path_settings(path_settings, project_root)
    required_roots = {"experiments", "checkpoints", "logs", "results"}
    missing_roots = required_roots - set(resolved_roots)

    if missing_roots:
        missing_names = ", ".join(sorted(missing_roots))
        raise ValueError(f"Path settings are missing required roots: {missing_names}")

    experiment_directory = resolved_roots["experiments"] / experiment_id
    checkpoint_directory = resolved_roots["checkpoints"] / experiment_id
    log_directory = resolved_roots["logs"] / experiment_id
    results_root = resolved_roots["results"]
    metrics_directory = results_root / "metrics" / experiment_id
    figures_directory = results_root / "figures" / experiment_id
    confusion_matrices_directory = (
        results_root / "confusion_matrices" / experiment_id
    )
    predictions_directory = results_root / "predictions" / experiment_id

    directories = (
        experiment_directory,
        checkpoint_directory,
        log_directory,
        metrics_directory,
        figures_directory,
        confusion_matrices_directory,
        predictions_directory,
    )
    existing_directories = [directory for directory in directories if directory.exists()]

    if existing_directories:
        formatted_paths = ", ".join(str(path) for path in existing_directories)
        raise FileExistsError(
            f"Experiment '{experiment_id}' would overwrite existing paths: "
            f"{formatted_paths}"
        )

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=False)

    config_snapshot = experiment_directory / "config.yaml"
    environment_snapshot = experiment_directory / "environment.json"

    with config_snapshot.open("w", encoding="utf-8") as file_handle:
        yaml.safe_dump(
            dict(resolved_configuration),
            file_handle,
            sort_keys=False,
            allow_unicode=True,
        )

    with environment_snapshot.open("w", encoding="utf-8") as file_handle:
        json.dump(
            collect_environment_information(),
            file_handle,
            indent=2,
            sort_keys=True,
        )
        file_handle.write("\n")

    return ExperimentPaths(
        experiment_directory=experiment_directory,
        checkpoint_directory=checkpoint_directory,
        log_directory=log_directory,
        metrics_directory=metrics_directory,
        figures_directory=figures_directory,
        confusion_matrices_directory=confusion_matrices_directory,
        predictions_directory=predictions_directory,
        config_snapshot=config_snapshot,
        environment_snapshot=environment_snapshot,
    )


def load_experiment_layout(
    experiment_id: str,
    path_settings: Mapping[str, str | Path],
    project_root: str | Path,
) -> ExperimentPaths:
    """Open an existing run layout without creating or replacing any paths."""
    if not experiment_id or experiment_id != _to_safe_identifier(experiment_id):
        raise ValueError(
            "Experiment IDs may contain lowercase letters, numbers, underscores, "
            "and hyphens only."
        )
    resolved_roots = resolve_path_settings(path_settings, project_root)
    required_roots = {"experiments", "checkpoints", "logs", "results"}
    missing_roots = required_roots - set(resolved_roots)
    if missing_roots:
        missing_names = ", ".join(sorted(missing_roots))
        raise ValueError(f"Path settings are missing required roots: {missing_names}")

    experiment_directory = resolved_roots["experiments"] / experiment_id
    checkpoint_directory = resolved_roots["checkpoints"] / experiment_id
    log_directory = resolved_roots["logs"] / experiment_id
    results_root = resolved_roots["results"]
    paths = ExperimentPaths(
        experiment_directory=experiment_directory,
        checkpoint_directory=checkpoint_directory,
        log_directory=log_directory,
        metrics_directory=results_root / "metrics" / experiment_id,
        figures_directory=results_root / "figures" / experiment_id,
        confusion_matrices_directory=(
            results_root / "confusion_matrices" / experiment_id
        ),
        predictions_directory=results_root / "predictions" / experiment_id,
        config_snapshot=experiment_directory / "config.yaml",
        environment_snapshot=experiment_directory / "environment.json",
    )
    required_directories = (
        paths.experiment_directory,
        paths.checkpoint_directory,
        paths.log_directory,
        paths.metrics_directory,
        paths.figures_directory,
        paths.confusion_matrices_directory,
        paths.predictions_directory,
    )
    missing_directories = [
        directory for directory in required_directories if not directory.is_dir()
    ]
    missing_files = [
        path
        for path in (paths.config_snapshot, paths.environment_snapshot)
        if not path.is_file()
    ]
    if missing_directories or missing_files:
        missing = missing_directories + missing_files
        formatted = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"Experiment '{experiment_id}' is incomplete; missing: {formatted}"
        )
    return paths


def collect_environment_information() -> dict[str, Any]:
    """
    Collect serializable runtime details needed to reproduce an experiment.

    Returns
    -------
    dict[str, Any]
        Python, operating-system, and PyTorch/CUDA information.
    """
    information: dict[str, Any] = {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "pytorch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cudnn_version": torch.backends.cudnn.version(),
    }

    if torch.cuda.is_available():
        information["cuda_device_count"] = torch.cuda.device_count()
        information["cuda_devices"] = [
            torch.cuda.get_device_name(index)
            for index in range(torch.cuda.device_count())
        ]

    return information


def _to_safe_identifier(value: str) -> str:
    normalized_value = value.strip().lower().replace(" ", "_")
    normalized_value = re.sub(r"[^a-z0-9_-]+", "", normalized_value)
    normalized_value = re.sub(r"_+", "_", normalized_value)
    return normalized_value.strip("_-")
