"""Tests for configuration loading, composition, and validation."""

from copy import deepcopy
from pathlib import Path

import pytest

from urban_sound_robustness.utils.config import (
    ConfigurationError,
    deep_merge,
    load_experiment_config,
    load_yaml_file,
    validate_project_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT_CONFIG = PROJECT_ROOT / "configs" / "experiment" / "development.yaml"


@pytest.mark.parametrize(
    ("file_name", "model_name", "augmentation_enabled"),
    [
        ("cnn_baseline.yaml", "cnn", False),
        ("cnn_augmented.yaml", "cnn", True),
        ("crnn_baseline.yaml", "crnn", False),
        ("crnn_augmented.yaml", "crnn", True),
        ("resnet18_baseline.yaml", "resnet18", False),
        ("resnet18_augmented.yaml", "resnet18", True),
    ],
)
def test_methodology_experiment_manifests_compose(
    file_name: str,
    model_name: str,
    augmentation_enabled: bool,
) -> None:
    """Every required baseline/augmented model condition must be runnable."""
    manifest = PROJECT_ROOT / "configs" / "experiment" / file_name

    configuration = load_experiment_config(manifest)

    assert configuration.section("model")["name"] == model_name
    assert configuration.section("augmentation")["enabled"] is augmentation_enabled
    assert configuration.section("dataset")["num_classes"] == 10


def test_development_configuration_composes_component_files() -> None:
    """The manifest should merge smoke overrides without losing base settings."""
    configuration = load_experiment_config(DEVELOPMENT_CONFIG)

    assert configuration.section("dataset")["name"] == "urbansound8k"
    assert configuration.section("training")["epochs"] == 2
    assert configuration.section("training")["batch_size"] == 4
    assert configuration.section("training")["optimizer"]["name"] == "adamw"
    assert configuration.section("training")["logging"]["tensorboard"] is False
    assert configuration.project_root == PROJECT_ROOT


def test_deep_merge_is_recursive_and_does_not_mutate_inputs() -> None:
    """Nested mappings merge, while lists are intentionally replaced."""
    base_values = {"training": {"epochs": 50, "tags": ["base"]}}
    override_values = {"training": {"epochs": 2, "tags": ["smoke"]}}

    merged_values = deep_merge(base_values, override_values)

    assert merged_values == {"training": {"epochs": 2, "tags": ["smoke"]}}
    assert base_values["training"]["epochs"] == 50
    assert override_values["training"]["tags"] == ["smoke"]


def test_overlapping_dataset_folds_are_rejected() -> None:
    """A fold must not appear in more than one data split."""
    configuration = load_experiment_config(DEVELOPMENT_CONFIG)
    invalid_values = deepcopy(configuration.data)
    invalid_values["dataset"]["folds"]["validation"] = [8]

    with pytest.raises(ConfigurationError, match="folds overlap"):
        validate_project_config(invalid_values)


def test_invalid_augmentation_probability_is_rejected() -> None:
    """Configuration errors should appear before an augmentation pipeline runs."""
    configuration = load_experiment_config(DEVELOPMENT_CONFIG)
    invalid_values = deepcopy(configuration.data)
    invalid_values["augmentation"]["waveform"]["time_shift"] = {
        "enabled": True,
        "probability": 1.5,
    }

    with pytest.raises(ConfigurationError, match="between zero and one"):
        validate_project_config(invalid_values)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("length_normalization", "padding_mode"), "reflect", "padding_mode"),
        (("representation",), "mfcc", "representation"),
        (("log_mel", "n_mels"), 63.5, "positive integer"),
        (("log_mel", "top_db"), -1.0, "top_db"),
        (("log_mel", "center"), "yes", "center"),
        (("normalization", "method"), "none", "normalization.method"),
    ],
)
def test_invalid_audio_preprocessing_configuration_is_rejected(
    path: tuple[str, ...],
    value,
    message: str,
) -> None:
    """Unsupported preprocessing choices should fail during config composition."""
    configuration = load_experiment_config(DEVELOPMENT_CONFIG)
    invalid_values = deepcopy(configuration.data)
    destination = invalid_values["audio"]
    for key in path[:-1]:
        destination = destination[key]
    destination[path[-1]] = value

    with pytest.raises(ConfigurationError, match=message):
        validate_project_config(invalid_values)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("noise_directory",), "", "noise_directory"),
        (("mixing", "power_epsilon"), 0.0, "power_epsilon"),
    ],
)
def test_invalid_snr_configuration_is_rejected(
    path: tuple[str, ...],
    value,
    message: str,
) -> None:
    """Noise locations and numerical thresholds must be valid before evaluation."""
    configuration = load_experiment_config(DEVELOPMENT_CONFIG)
    invalid_values = deepcopy(configuration.data)
    destination = invalid_values["evaluation"]
    for key in path[:-1]:
        destination = destination[key]
    destination[path[-1]] = value

    with pytest.raises(ConfigurationError, match=message):
        validate_project_config(invalid_values)


def test_yaml_root_must_be_a_mapping(tmp_path: Path) -> None:
    """A YAML list cannot serve as a named configuration component."""
    invalid_yaml = tmp_path / "invalid.yaml"
    invalid_yaml.write_text("- first\n- second\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="mapping at its root"):
        load_yaml_file(invalid_yaml)
