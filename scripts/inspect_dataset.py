"""Validate a configured dataset and save numerical inspection outputs."""

import argparse
import logging
from pathlib import Path

from urban_sound_robustness.datasets import (
    DatasetNotFoundError,
    DatasetValidationError,
    create_dataset_adapter,
    inspect_dataset,
    save_inspection_result,
)
from urban_sound_robustness.utils.config import load_experiment_config
from urban_sound_robustness.utils.logging_utils import configure_logging
from urban_sound_robustness.utils.paths import resolve_path_settings


def parse_arguments() -> argparse.Namespace:
    """Parse dataset inspection command-line settings."""
    parser = argparse.ArgumentParser(
        description="Validate dataset metadata and save inspection summaries."
    )
    parser.add_argument(
        "config",
        nargs="?",
        type=Path,
        default=Path("configs/experiment/development.yaml"),
        help="Composed experiment configuration.",
    )
    parser.add_argument(
        "--split",
        choices=("all", "train", "validation", "test"),
        default="all",
        help="Inspect all official folds or one configured split.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=None,
        help="Override the configured dataset-inspection output directory.",
    )
    parser.add_argument(
        "--skip-audio-scan",
        action="store_true",
        help="Use metadata durations without reading audio file headers.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the audio-header progress bar.",
    )
    return parser.parse_args()


def main() -> int:
    """Run configured dataset validation and structured inspection."""
    arguments = parse_arguments()
    project_config = load_experiment_config(arguments.config)
    resolved_paths = resolve_path_settings(
        project_config.section("paths"), project_config.project_root
    )
    logger = configure_logging(
        resolved_paths["logs"] / "dataset_inspection.log", level="INFO"
    )

    try:
        adapter = create_dataset_adapter(
            project_config.section("dataset"), project_config.project_root
        )
        logger.info("Loading and validating dataset: %s", adapter.dataset_name)
        records = (
            adapter.load_records()
            if arguments.split == "all"
            else adapter.records_for_split(arguments.split)
        )
        inspection_settings = project_config.section("dataset").get(
            "inspection", {}
        )
        inspect_headers = inspection_settings.get("inspect_audio_headers", True)

        if arguments.skip_audio_scan:
            inspect_headers = False

        result = inspect_dataset(
            records=records,
            class_names=adapter.class_names,
            inspect_audio_headers=inspect_headers,
            imbalance_warning_ratio=float(
                inspection_settings.get("imbalance_warning_ratio", 1.5)
            ),
            show_progress=not arguments.no_progress,
        )
        output_directory = arguments.output_directory

        if output_directory is None:
            output_directory = (
                resolved_paths["results"] / "metrics" / "dataset_inspection"
            )

        output_paths = save_inspection_result(result, output_directory)
    except (DatasetNotFoundError, DatasetValidationError, ValueError) as error:
        logger.error("Dataset inspection failed: %s", error)
        return 1

    logger.info("Dataset samples: %d", result.summary["total_samples"])
    logger.info("Missing audio files: %d", result.summary["missing_file_count"])
    logger.info("Unreadable audio files: %d", result.summary["unreadable_file_count"])

    for output_name, output_path in output_paths.items():
        logger.info("Saved %s: %s", output_name, output_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

