"""Load, compose, and validate YAML experiment configurations."""

from copy import deepcopy
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping

import yaml

from urban_sound_robustness.utils.paths import find_project_root


COMPONENT_SECTIONS = (
    "paths",
    "dataset",
    "audio",
    "augmentation",
    "model",
    "training",
    "evaluation",
)


class ConfigurationError(ValueError):
    """Raised when a configuration file is missing, malformed, or inconsistent."""


@dataclass(frozen=True)
class ProjectConfig:
    """Resolved experiment configuration together with its source locations."""

    data: dict[str, Any]
    source_path: Path
    project_root: Path

    def section(self, name: str) -> dict[str, Any]:
        """
        Return one named configuration section.

        Parameters
        ----------
        name : str
            Top-level section name such as ``audio`` or ``training``.

        Returns
        -------
        dict[str, Any]
            Requested section.

        Raises
        ------
        ConfigurationError
            If the section does not exist or is not a mapping.
        """
        value = self.data.get(name)

        if not isinstance(value, dict):
            raise ConfigurationError(
                f"Configuration section '{name}' is missing or is not a mapping."
            )

        return value


def load_yaml_file(file_path: str | Path) -> dict[str, Any]:
    """
    Load one YAML file and require a mapping at its root.

    Parameters
    ----------
    file_path : str or Path
        YAML file to read.

    Returns
    -------
    dict[str, Any]
        Parsed configuration mapping.

    Raises
    ------
    ConfigurationError
        If the file is missing, unreadable, invalid, empty, or not a mapping.
    """
    path = Path(file_path).expanduser().resolve()

    if not path.is_file():
        raise ConfigurationError(f"Configuration file does not exist: {path}")

    try:
        with path.open("r", encoding="utf-8") as file_handle:
            loaded_value = yaml.safe_load(file_handle)
    except OSError as error:
        raise ConfigurationError(
            f"Could not read configuration file '{path}': {error}"
        ) from error
    except yaml.YAMLError as error:
        raise ConfigurationError(
            f"Invalid YAML in configuration file '{path}': {error}"
        ) from error

    if loaded_value is None:
        raise ConfigurationError(f"Configuration file is empty: {path}")

    if not isinstance(loaded_value, dict):
        raise ConfigurationError(
            f"Configuration file must contain a mapping at its root: {path}"
        )

    return loaded_value


def deep_merge(
    base_values: Mapping[str, Any],
    override_values: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Recursively merge configuration overrides without modifying either input.

    Dictionaries are merged recursively. Lists and scalar values are replaced as
    complete values because merging list elements would be ambiguous.

    Parameters
    ----------
    base_values : Mapping[str, Any]
        Original configuration values.
    override_values : Mapping[str, Any]
        Values that should replace or extend the originals.

    Returns
    -------
    dict[str, Any]
        Independent merged configuration.
    """
    merged_values = deepcopy(dict(base_values))

    for key, override_value in override_values.items():
        original_value = merged_values.get(key)

        if isinstance(original_value, dict) and isinstance(override_value, Mapping):
            merged_values[key] = deep_merge(original_value, override_value)
        else:
            merged_values[key] = deepcopy(override_value)

    return merged_values


def load_experiment_config(
    experiment_file: str | Path,
    project_root: str | Path | None = None,
) -> ProjectConfig:
    """
    Compose component YAML files selected by an experiment manifest.

    Parameters
    ----------
    experiment_file : str or Path
        Manifest containing ``experiment``, ``components``, and optional
        ``overrides`` mappings.
    project_root : str or Path or None
        Explicit repository root. It is discovered when omitted.

    Returns
    -------
    ProjectConfig
        Validated resolved configuration and its source information.
    """
    source_path = Path(experiment_file).expanduser().resolve()
    manifest = load_yaml_file(source_path)
    resolved_project_root = (
        find_project_root(source_path)
        if project_root is None
        else Path(project_root).expanduser().resolve()
    )

    experiment_settings = _require_mapping(manifest, "experiment", "manifest")
    component_references = _require_mapping(manifest, "components", "manifest")
    missing_components = set(COMPONENT_SECTIONS) - set(component_references)
    unknown_components = set(component_references) - set(COMPONENT_SECTIONS)

    if missing_components:
        missing_names = ", ".join(sorted(missing_components))
        raise ConfigurationError(
            f"Experiment manifest is missing component references: {missing_names}"
        )

    if unknown_components:
        unknown_names = ", ".join(sorted(unknown_components))
        raise ConfigurationError(
            f"Experiment manifest contains unknown components: {unknown_names}"
        )

    resolved_data: dict[str, Any] = {
        "experiment": deepcopy(dict(experiment_settings))
    }

    for section_name in COMPONENT_SECTIONS:
        component_reference = component_references[section_name]

        if not isinstance(component_reference, str) or not component_reference.strip():
            raise ConfigurationError(
                f"Component reference '{section_name}' must be a non-empty path string."
            )

        component_path = (source_path.parent / component_reference).resolve()
        resolved_data[section_name] = load_yaml_file(component_path)

    overrides = manifest.get("overrides", {})

    if not isinstance(overrides, dict):
        raise ConfigurationError("Experiment manifest 'overrides' must be a mapping.")

    allowed_override_sections = set(COMPONENT_SECTIONS) | {"experiment"}
    unknown_override_sections = set(overrides) - allowed_override_sections

    if unknown_override_sections:
        unknown_names = ", ".join(sorted(unknown_override_sections))
        raise ConfigurationError(f"Overrides contain unknown sections: {unknown_names}")

    for section_name, section_overrides in overrides.items():
        if not isinstance(section_overrides, dict):
            raise ConfigurationError(
                f"Overrides for section '{section_name}' must be a mapping."
            )

        resolved_data[section_name] = deep_merge(
            resolved_data[section_name], section_overrides
        )

    validate_project_config(resolved_data)
    return ProjectConfig(
        data=resolved_data,
        source_path=source_path,
        project_root=resolved_project_root,
    )


def validate_project_config(configuration: Mapping[str, Any]) -> None:
    """
    Validate cross-section assumptions required by the reusable pipeline.

    Parameters
    ----------
    configuration : Mapping[str, Any]
        Fully composed project configuration.

    Returns
    -------
    None

    Raises
    ------
    ConfigurationError
        If a required value is absent, incorrectly typed, or inconsistent.
    """
    for section_name in ("experiment", *COMPONENT_SECTIONS):
        _require_mapping(configuration, section_name, "configuration")

    _validate_experiment(configuration["experiment"])
    _validate_paths(configuration["paths"])
    _validate_dataset(configuration["dataset"])
    _validate_audio(configuration["audio"])
    _validate_augmentation(configuration["augmentation"])
    _validate_model(configuration["model"], configuration["dataset"])
    _validate_training(configuration["training"])
    _validate_evaluation(configuration["evaluation"])


def _require_mapping(
    values: Mapping[str, Any],
    key: str,
    location: str,
) -> Mapping[str, Any]:
    value = values.get(key)

    if not isinstance(value, dict):
        raise ConfigurationError(
            f"'{key}' in {location} must exist and contain a mapping."
        )

    return value


def _require_positive_number(
    values: Mapping[str, Any],
    key: str,
    location: str,
) -> float:
    value = values.get(key)

    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ConfigurationError(f"'{location}.{key}' must be a positive number.")

    return float(value)


def _require_non_empty_string(
    values: Mapping[str, Any],
    key: str,
    location: str,
) -> str:
    value = values.get(key)

    if not isinstance(value, str) or not value.strip():
        raise ConfigurationError(f"'{location}.{key}' must be a non-empty string.")

    return value


def _validate_experiment(settings: Mapping[str, Any]) -> None:
    _require_non_empty_string(settings, "id", "experiment")


def _validate_paths(settings: Mapping[str, Any]) -> None:
    required_names = {"data", "experiments", "checkpoints", "logs", "results"}
    missing_names = required_names - set(settings)

    if missing_names:
        names = ", ".join(sorted(missing_names))
        raise ConfigurationError(f"The paths section is missing: {names}")

    for name, path_value in settings.items():
        if not isinstance(path_value, str) or not path_value.strip():
            raise ConfigurationError(f"'paths.{name}' must be a non-empty path string.")


def _validate_dataset(settings: Mapping[str, Any]) -> None:
    _require_non_empty_string(settings, "name", "dataset")
    _require_non_empty_string(settings, "adapter", "dataset")
    _require_non_empty_string(settings, "dataset_root", "dataset")
    _require_non_empty_string(settings, "metadata_file", "dataset")

    num_classes = settings.get("num_classes")
    class_names = settings.get("class_names")

    if isinstance(num_classes, bool) or not isinstance(num_classes, int) or num_classes <= 1:
        raise ConfigurationError("'dataset.num_classes' must be an integer above one.")

    valid_class_names = isinstance(class_names, list) and all(
        isinstance(name, str) and name.strip() for name in class_names
    )

    if not valid_class_names:
        raise ConfigurationError("'dataset.class_names' must be a list of names.")

    if len(class_names) != num_classes:
        raise ConfigurationError(
            "'dataset.num_classes' must equal the number of dataset class names."
        )

    if len(set(class_names)) != len(class_names):
        raise ConfigurationError("'dataset.class_names' must not contain duplicates.")

    folds = _require_mapping(settings, "folds", "dataset")
    required_splits = {"train", "validation", "test"}

    if set(folds) != required_splits:
        raise ConfigurationError(
            "'dataset.folds' must contain exactly train, validation, and test."
        )

    assigned_folds: dict[str, set[int]] = {}

    for split_name in ("train", "validation", "test"):
        fold_values = folds[split_name]

        if not isinstance(fold_values, list) or not fold_values:
            raise ConfigurationError(
                f"'dataset.folds.{split_name}' must be a non-empty list."
            )

        valid_folds = all(
            isinstance(fold, int) and not isinstance(fold, bool) and fold > 0
            for fold in fold_values
        )

        if not valid_folds:
            raise ConfigurationError(
                f"'dataset.folds.{split_name}' must contain positive integers."
            )

        if len(set(fold_values)) != len(fold_values):
            raise ConfigurationError(
                f"'dataset.folds.{split_name}' contains duplicate fold IDs."
            )

        assigned_folds[split_name] = set(fold_values)

    split_pairs = (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    )

    for first_split, second_split in split_pairs:
        overlap = assigned_folds[first_split] & assigned_folds[second_split]

        if overlap:
            raise ConfigurationError(
                f"Dataset folds overlap between {first_split} and {second_split}: "
                f"{sorted(overlap)}"
            )


def _validate_audio(settings: Mapping[str, Any]) -> None:
    sample_rate = _require_positive_number(settings, "sample_rate", "audio")
    _require_positive_number(settings, "clip_duration_seconds", "audio")

    if not isinstance(settings.get("mono"), bool):
        raise ConfigurationError("'audio.mono' must be true or false.")

    length_normalization = _require_mapping(
        settings, "length_normalization", "audio"
    )
    if length_normalization.get("padding_mode") != "zero":
        raise ConfigurationError(
            "'audio.length_normalization.padding_mode' must currently be zero."
        )
    for crop_name in ("training_crop", "evaluation_crop"):
        crop_mode = length_normalization.get(crop_name)
        if crop_mode not in {"center", "random"}:
            raise ConfigurationError(
                f"'audio.length_normalization.{crop_name}' must be center or random."
            )

    if settings.get("representation") != "log_mel":
        raise ConfigurationError("'audio.representation' must currently be log_mel.")

    log_mel = _require_mapping(settings, "log_mel", "audio")
    n_fft = _require_positive_number(log_mel, "n_fft", "audio.log_mel")
    win_length = _require_positive_number(log_mel, "win_length", "audio.log_mel")
    _require_positive_number(log_mel, "hop_length", "audio.log_mel")
    _require_positive_number(log_mel, "n_mels", "audio.log_mel")

    for integer_name in ("n_fft", "win_length", "hop_length", "n_mels"):
        integer_value = log_mel.get(integer_name)
        if isinstance(integer_value, bool) or not isinstance(integer_value, int):
            raise ConfigurationError(
                f"'audio.log_mel.{integer_name}' must be a positive integer."
            )

    if win_length > n_fft:
        raise ConfigurationError("'audio.log_mel.win_length' cannot exceed n_fft.")

    f_min = log_mel.get("f_min")

    if isinstance(f_min, bool) or not isinstance(f_min, (int, float)) or f_min < 0:
        raise ConfigurationError("'audio.log_mel.f_min' must be zero or greater.")

    f_max = log_mel.get("f_max")

    if f_max is not None:
        if isinstance(f_max, bool) or not isinstance(f_max, (int, float)):
            raise ConfigurationError("'audio.log_mel.f_max' must be a number or null.")

        if f_max <= f_min or f_max > sample_rate / 2:
            raise ConfigurationError(
                "'audio.log_mel.f_max' must be above f_min and no higher than Nyquist."
            )

    _require_positive_number(log_mel, "power", "audio.log_mel")
    top_db = log_mel.get("top_db")
    if top_db is not None:
        if (
            isinstance(top_db, bool)
            or not isinstance(top_db, (int, float))
            or top_db <= 0
        ):
            raise ConfigurationError(
                "'audio.log_mel.top_db' must be a positive number or null."
            )

    if not isinstance(log_mel.get("center", True), bool):
        raise ConfigurationError("'audio.log_mel.center' must be true or false.")
    if log_mel.get("pad_mode", "reflect") not in {"reflect", "constant"}:
        raise ConfigurationError(
            "'audio.log_mel.pad_mode' must be reflect or constant."
        )
    if log_mel.get("mel_scale", "htk") not in {"htk", "slaney"}:
        raise ConfigurationError("'audio.log_mel.mel_scale' must be htk or slaney.")

    mfcc = _require_mapping(settings, "mfcc", "audio")
    n_mfcc = mfcc.get("n_mfcc")
    if (
        isinstance(n_mfcc, bool)
        or not isinstance(n_mfcc, int)
        or n_mfcc <= 0
    ):
        raise ConfigurationError("'audio.mfcc.n_mfcc' must be a positive integer.")
    if n_mfcc > log_mel["n_mels"]:
        raise ConfigurationError("'audio.mfcc.n_mfcc' cannot exceed n_mels.")

    normalization = _require_mapping(settings, "normalization", "audio")
    if normalization.get("method") != "per_example_standardization":
        raise ConfigurationError(
            "'audio.normalization.method' must currently be "
            "per_example_standardization."
        )
    _require_positive_number(normalization, "epsilon", "audio.normalization")


def _validate_augmentation(settings: Mapping[str, Any]) -> None:
    _require_non_empty_string(settings, "name", "augmentation")
    if not isinstance(settings.get("enabled"), bool):
        raise ConfigurationError("'augmentation.enabled' must be true or false.")

    _validate_probability_values(settings, "augmentation")
    waveform = _require_mapping(settings, "waveform", "augmentation")
    spectrogram = _require_mapping(settings, "spectrogram", "augmentation")
    components = {
        "time_shift": _require_mapping(waveform, "time_shift", "augmentation.waveform"),
        "random_gain": _require_mapping(waveform, "random_gain", "augmentation.waveform"),
        "background_noise": _require_mapping(
            waveform, "background_noise", "augmentation.waveform"
        ),
        "pitch_shift": _require_mapping(waveform, "pitch_shift", "augmentation.waveform"),
        "time_stretch": _require_mapping(waveform, "time_stretch", "augmentation.waveform"),
        "frequency_mask": _require_mapping(
            spectrogram, "frequency_mask", "augmentation.spectrogram"
        ),
        "time_mask": _require_mapping(
            spectrogram, "time_mask", "augmentation.spectrogram"
        ),
    }
    for name, component in components.items():
        if not isinstance(component.get("enabled"), bool):
            raise ConfigurationError(
                f"'augmentation.{name}.enabled' must be true or false."
            )

    for unsupported_name in ("pitch_shift", "time_stretch"):
        if components[unsupported_name]["enabled"]:
            raise ConfigurationError(
                f"'augmentation.waveform.{unsupported_name}' is not currently supported."
            )

    if components["time_shift"]["enabled"]:
        maximum_shift = components["time_shift"].get("max_shift_fraction")
        if (
            isinstance(maximum_shift, bool)
            or not isinstance(maximum_shift, (int, float))
            or not 0 <= maximum_shift <= 1
        ):
            raise ConfigurationError("'max_shift_fraction' must be between zero and one.")
    if components["random_gain"]["enabled"]:
        minimum_gain = components["random_gain"].get("min_gain_db")
        maximum_gain = components["random_gain"].get("max_gain_db")
        if not all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in (minimum_gain, maximum_gain)
        ) or minimum_gain > maximum_gain:
            raise ConfigurationError("Random-gain dB bounds must be ordered numbers.")
    if components["background_noise"]["enabled"]:
        _require_non_empty_string(
            components["background_noise"],
            "noise_directory",
            "augmentation.waveform.background_noise",
        )
        minimum_snr = components["background_noise"].get("min_snr_db")
        maximum_snr = components["background_noise"].get("max_snr_db")
        if not all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in (minimum_snr, maximum_snr)
        ) or minimum_snr > maximum_snr:
            raise ConfigurationError("Background-noise SNR bounds must be ordered numbers.")
    for mask_name, width_name in (
        ("frequency_mask", "max_mask_bins"),
        ("time_mask", "max_mask_frames"),
    ):
        if components[mask_name]["enabled"]:
            width = components[mask_name].get(width_name)
            if isinstance(width, bool) or not isinstance(width, int) or width < 1:
                raise ConfigurationError(
                    f"'augmentation.{mask_name}.{width_name}' must be a positive integer."
                )


def _validate_probability_values(values: Mapping[str, Any], location: str) -> None:
    for key, value in values.items():
        current_location = f"{location}.{key}"

        if key == "probability":
            is_number = isinstance(value, (int, float)) and not isinstance(value, bool)

            if not is_number or not 0 <= value <= 1:
                raise ConfigurationError(
                    f"'{current_location}' must be a number between zero and one."
                )

        if isinstance(value, dict):
            _validate_probability_values(value, current_location)


def _validate_model(
    model_settings: Mapping[str, Any],
    dataset_settings: Mapping[str, Any],
) -> None:
    model_name = _require_non_empty_string(model_settings, "name", "model").lower()
    if model_name not in {"cnn", "crnn", "resnet18"}:
        raise ConfigurationError("'model.name' must be cnn, crnn, or resnet18.")

    if model_settings.get("num_classes") != dataset_settings.get("num_classes"):
        raise ConfigurationError(
            "'model.num_classes' must match 'dataset.num_classes'."
        )
    input_channels = model_settings.get("input_channels")
    if isinstance(input_channels, bool) or not isinstance(input_channels, int) or input_channels < 1:
        raise ConfigurationError("'model.input_channels' must be a positive integer.")
    dropout = model_settings.get("dropout")
    if (
        isinstance(dropout, bool)
        or not isinstance(dropout, (int, float))
        or not 0 <= dropout < 1
    ):
        raise ConfigurationError("'model.dropout' must be between zero and one.")

    channel_key = "channels" if model_name == "cnn" else "cnn_channels"
    if model_name in {"cnn", "crnn"}:
        channels = model_settings.get(channel_key)
        if not isinstance(channels, list) or not channels or not all(
            isinstance(value, int) and not isinstance(value, bool) and value > 0
            for value in channels
        ):
            raise ConfigurationError(
                f"'model.{channel_key}' must be a non-empty list of positive integers."
            )
    if model_name == "crnn":
        if model_settings.get("recurrent_type") != "gru":
            raise ConfigurationError("'model.recurrent_type' must currently be gru.")
        for key in ("recurrent_hidden_size", "recurrent_layers"):
            value = model_settings.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ConfigurationError(f"'model.{key}' must be a positive integer.")
        if not isinstance(model_settings.get("bidirectional"), bool):
            raise ConfigurationError("'model.bidirectional' must be true or false.")
    if model_name == "resnet18" and not isinstance(
        model_settings.get("pretrained"), bool
    ):
        raise ConfigurationError("'model.pretrained' must be true or false.")


def _validate_training(settings: Mapping[str, Any]) -> None:
    seed = settings.get("seed")

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ConfigurationError("'training.seed' must be a non-negative integer.")

    device = _require_non_empty_string(settings, "device", "training").lower()

    if device not in {"auto", "cpu", "cuda"} and not device.startswith("cuda:"):
        raise ConfigurationError(
            "'training.device' must be auto, cpu, cuda, or a device such as cuda:0."
        )

    for key in ("epochs", "batch_size", "gradient_accumulation_steps"):
        _require_positive_number(settings, key, "training")
        value = settings.get(key)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigurationError(f"'training.{key}' must be a positive integer.")

    num_workers = settings.get("num_workers")

    if isinstance(num_workers, bool) or not isinstance(num_workers, int) or num_workers < 0:
        raise ConfigurationError("'training.num_workers' must be a non-negative integer.")

    if "deterministic" in settings and not isinstance(settings["deterministic"], bool):
        raise ConfigurationError("'training.deterministic' must be true or false.")
    for boolean_name in ("pin_memory", "mixed_precision"):
        if not isinstance(settings.get(boolean_name), bool):
            raise ConfigurationError(f"'training.{boolean_name}' must be true or false.")

    loss = _require_mapping(settings, "loss", "training")
    if loss.get("name") != "cross_entropy":
        raise ConfigurationError("'training.loss.name' must be cross_entropy.")
    optimizer = _require_mapping(settings, "optimizer", "training")
    if optimizer.get("name") not in {"adam", "adamw"}:
        raise ConfigurationError("'training.optimizer.name' must be adam or adamw.")
    _require_positive_number(optimizer, "learning_rate", "training.optimizer")
    weight_decay = optimizer.get("weight_decay", 0.0)
    if (
        isinstance(weight_decay, bool)
        or not isinstance(weight_decay, (int, float))
        or weight_decay < 0
    ):
        raise ConfigurationError("'training.optimizer.weight_decay' cannot be negative.")

    scheduler = _require_mapping(settings, "scheduler", "training")
    if scheduler.get("name") not in {"none", "reduce_on_plateau", "cosine"}:
        raise ConfigurationError("Unsupported training scheduler.")
    early_stopping = _require_mapping(settings, "early_stopping", "training")
    checkpointing = _require_mapping(settings, "checkpointing", "training")
    logging_settings = _require_mapping(settings, "logging", "training")
    for location, values, key in (
        ("early_stopping", early_stopping, "enabled"),
        ("checkpointing", checkpointing, "save_best"),
        ("checkpointing", checkpointing, "save_last"),
        ("logging", logging_settings, "tensorboard"),
    ):
        if not isinstance(values.get(key), bool):
            raise ConfigurationError(f"'training.{location}.{key}' must be true or false.")


def _validate_evaluation(settings: Mapping[str, Any]) -> None:
    corruption_seed = settings.get("corruption_seed")

    if (
        isinstance(corruption_seed, bool)
        or not isinstance(corruption_seed, int)
        or corruption_seed < 0
    ):
        raise ConfigurationError(
            "'evaluation.corruption_seed' must be a non-negative integer."
        )

    conditions = settings.get("conditions")

    if not isinstance(conditions, list) or not conditions:
        raise ConfigurationError("'evaluation.conditions' must be a non-empty list.")

    condition_names: list[str] = []
    clean_condition_count = 0

    for index, condition in enumerate(conditions):
        if not isinstance(condition, dict):
            raise ConfigurationError(
                f"'evaluation.conditions[{index}]' must be a mapping."
            )

        name = _require_non_empty_string(
            condition, "name", f"evaluation.conditions[{index}]"
        )
        condition_names.append(name)
        snr_db = condition.get("snr_db")

        if snr_db is None:
            clean_condition_count += 1
            continue

        is_finite_number = (
            isinstance(snr_db, (int, float))
            and not isinstance(snr_db, bool)
            and math.isfinite(snr_db)
        )

        if not is_finite_number:
            raise ConfigurationError(
                f"SNR for evaluation condition '{name}' must be finite or null."
            )

    if len(set(condition_names)) != len(condition_names):
        raise ConfigurationError("Evaluation condition names must be unique.")

    if clean_condition_count != 1:
        raise ConfigurationError(
            "Evaluation conditions must contain exactly one clean condition with null SNR."
        )

    _require_non_empty_string(settings, "noise_directory", "evaluation")
    mixing = _require_mapping(settings, "mixing", "evaluation")
    _require_positive_number(mixing, "power_epsilon", "evaluation.mixing")
