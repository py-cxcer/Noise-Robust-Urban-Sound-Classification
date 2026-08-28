"""Validate and summarize a composed experiment configuration."""

import argparse
from pathlib import Path

from urban_sound_robustness.utils.config import load_experiment_config
from urban_sound_robustness.utils.device import describe_device, select_device
from urban_sound_robustness.utils.paths import resolve_path_settings


def parse_arguments() -> argparse.Namespace:
    """
    Parse the optional experiment configuration path.

    Returns
    -------
    argparse.Namespace
        Parsed command-line values.
    """
    parser = argparse.ArgumentParser(
        description="Validate a composed experiment configuration."
    )
    parser.add_argument(
        "config",
        nargs="?",
        type=Path,
        default=Path("configs/experiment/development.yaml"),
        help="Experiment YAML file to validate.",
    )
    return parser.parse_args()


def main() -> None:
    """Load a configuration and print its important resolved settings."""
    arguments = parse_arguments()
    project_config = load_experiment_config(arguments.config)
    selected_device = select_device(project_config.section("training")["device"])
    resolved_paths = resolve_path_settings(
        project_config.section("paths"), project_config.project_root
    )

    experiment = project_config.section("experiment")
    dataset = project_config.section("dataset")
    audio = project_config.section("audio")
    model = project_config.section("model")
    training = project_config.section("training")

    print(f"Configuration valid: {project_config.source_path}")
    print(f"Experiment ID: {experiment['id']}")
    print(f"Dataset: {dataset['name']}")
    print(f"Model: {model['name']}")
    print(f"Sample rate: {audio['sample_rate']} Hz")
    print(f"Clip duration: {audio['clip_duration_seconds']} seconds")
    print(f"Training folds: {dataset['folds']['train']}")
    print(f"Validation folds: {dataset['folds']['validation']}")
    print(f"Test folds: {dataset['folds']['test']}")
    print(f"Batch size: {training['batch_size']}")
    print(f"Device: {describe_device(selected_device)}")
    print("Resolved paths:")

    for path_name, path_value in resolved_paths.items():
        print(f"  {path_name}: {path_value}")


if __name__ == "__main__":
    main()

