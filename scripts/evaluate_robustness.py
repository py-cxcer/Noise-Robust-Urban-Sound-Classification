"""Evaluate one trained best checkpoint across configured SNR conditions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import pandas as pd
import torch
from torch.utils.data import DataLoader

from urban_sound_robustness.audio import AudioPreprocessor
from urban_sound_robustness.datasets import create_dataset_adapter
from urban_sound_robustness.evaluation import (
    DeterministicNoiseCorruptor,
    RobustnessEvaluationDataset,
    calculate_classification_metrics,
    calculate_robustness_metrics,
    collect_condition_predictions,
    load_research_checkpoint,
    parse_robustness_conditions,
    save_classification_result,
    save_robustness_analysis,
    validate_noise_isolation,
)
from urban_sound_robustness.models import count_trainable_parameters, create_model
from urban_sound_robustness.utils import (
    configure_logging,
    create_data_loader_generator,
    describe_device,
    load_experiment_config,
    resolve_path_settings,
    seed_data_loader_worker,
    seed_everything,
    select_device,
)


def parse_arguments() -> argparse.Namespace:
    """Parse a manifest/checkpoint pair and bounded verification options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        type=Path,
        help="Current experiment manifest; its evaluation section is authoritative.",
    )
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="Full training run's checkpoints/<experiment-id>/best.pt.",
    )
    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default="test",
        help="Use validation for a smoke check; test is the research default.",
    )
    parser.add_argument("--device", default=None, help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Evaluate a deterministic prefix only; marks output as a smoke run.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=None,
        help="Override results/robustness/<experiment-id>/<split>.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing evaluation directory after inspection.",
    )
    return parser.parse_args()


def _bounded_records(records, maximum_samples: int | None):
    """Return all records or a non-empty deterministic prefix."""
    if maximum_samples is not None and maximum_samples < 1:
        raise ValueError("--max-samples must be at least one.")
    selected = records if maximum_samples is None else records[:maximum_samples]
    if not selected:
        raise ValueError("The selected evaluation split contains no samples.")
    return selected


def _safe_condition_directory(name: str) -> str:
    """Reject condition names that could escape their output directory."""
    if name in {"", ".", ".."} or Path(name).name != name:
        raise ValueError(f"Unsafe robustness condition name: {name!r}")
    if "/" in name or "\\" in name:
        raise ValueError(f"Unsafe robustness condition name: {name!r}")
    return name


def _resolve_output_directory(arguments, project_config, experiment_id: str) -> Path:
    """Resolve an isolated output path and reject accidental replacement."""
    if arguments.output_directory is not None:
        output = arguments.output_directory.expanduser().resolve()
    else:
        paths = resolve_path_settings(
            project_config.section("paths"),
            project_config.project_root,
        )
        suffix = arguments.split
        if arguments.max_samples is not None:
            suffix = f"{suffix}_first{arguments.max_samples}"
        output = paths["results"] / "robustness" / experiment_id / suffix
    if output.is_dir() and any(output.iterdir()) and not arguments.overwrite:
        raise FileExistsError(
            f"Evaluation output already exists and is not empty: {output}. "
            "Inspect it, then pass --overwrite only when replacement is intended."
        )
    output.mkdir(parents=True, exist_ok=True)
    return output


def main() -> int:
    """Run deterministic checkpoint inference and save robustness artifacts."""
    arguments = parse_arguments()
    started_at = time.perf_counter()
    project_config = load_experiment_config(arguments.config)
    training_settings = project_config.section("training")
    seed = int(training_settings["seed"])
    seed_everything(
        seed,
        deterministic=bool(training_settings["deterministic"]),
    )

    research_checkpoint = load_research_checkpoint(
        arguments.checkpoint,
        project_config.data,
        map_location="cpu",
    )
    output_directory = _resolve_output_directory(
        arguments,
        project_config,
        research_checkpoint.experiment_id,
    )
    logger = configure_logging(output_directory / "evaluation.log")

    device_preference = arguments.device or str(training_settings["device"])
    device = select_device(device_preference)
    model = create_model(project_config.section("model"))
    if model.__class__.__name__ != research_checkpoint.model_class:
        raise ValueError(
            "Checkpoint model class does not match the configured model: "
            f"{research_checkpoint.model_class} != {model.__class__.__name__}."
        )
    model.load_state_dict(research_checkpoint.model_state_dict, strict=True)
    model.to(device).eval()

    evaluation_settings = project_config.section("evaluation")
    noise_directory, noise_paths = validate_noise_isolation(
        research_checkpoint.configuration,
        evaluation_settings,
        project_config.project_root,
    )
    conditions = parse_robustness_conditions(evaluation_settings)
    preprocessor = AudioPreprocessor(project_config.section("audio")).eval()
    corruptor = DeterministicNoiseCorruptor(
        noise_directory,
        target_sample_rate=preprocessor.sample_rate,
        corruption_seed=int(evaluation_settings["corruption_seed"]),
        power_epsilon=float(evaluation_settings["mixing"]["power_epsilon"]),
    )

    adapter = create_dataset_adapter(
        project_config.section("dataset"),
        project_config.project_root,
    )
    records = _bounded_records(
        adapter.records_for_split(arguments.split),
        arguments.max_samples,
    )
    num_workers = (
        int(training_settings["num_workers"])
        if arguments.num_workers is None
        else arguments.num_workers
    )
    batch_size = (
        int(training_settings["batch_size"])
        if arguments.batch_size is None
        else arguments.batch_size
    )
    if num_workers < 0:
        raise ValueError("--num-workers cannot be negative.")
    if batch_size < 1:
        raise ValueError("--batch-size must be at least one.")

    dataset_settings = project_config.section("dataset")
    class_names = dataset_settings["class_names"]
    architecture = str(project_config.section("model")["name"])
    training_condition = str(project_config.section("augmentation")["name"])
    logger.info("Experiment: %s", research_checkpoint.experiment_id)
    logger.info("Checkpoint: %s", research_checkpoint.path)
    logger.info("Device: %s", describe_device(device))
    logger.info(
        "Data: %d %s samples; %d held-out noise files",
        len(records),
        arguments.split,
        len(noise_paths),
    )
    logger.info(
        "Model: %s (%d trainable parameters)",
        model.__class__.__name__,
        count_trainable_parameters(model),
    )

    condition_rows: list[dict[str, object]] = []
    condition_outputs: dict[str, dict[str, object]] = {}
    for condition_index, condition in enumerate(conditions):
        condition_name = _safe_condition_directory(condition.name)
        logger.info(
            "Evaluating condition %s (%s dB)",
            condition.name,
            "clean" if condition.snr_db is None else condition.snr_db,
        )
        condition_dataset = RobustnessEvaluationDataset(
            records,
            preprocessor,
            corruptor,
            condition,
        )
        data_loader = DataLoader(
            condition_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=bool(training_settings["pin_memory"])
            and device.type == "cuda",
            worker_init_fn=seed_data_loader_worker,
            generator=create_data_loader_generator(seed + condition_index),
        )
        collection = collect_condition_predictions(model, data_loader, device)
        classification = calculate_classification_metrics(
            collection.targets,
            collection.predictions,
            class_names,
        )
        condition_directory = output_directory / "conditions" / condition_name
        artifact_paths = save_classification_result(
            classification,
            condition_directory,
            class_names=class_names,
            sample_ids=collection.sample_ids,
            prediction_metadata={
                column: collection.metadata[column].tolist()
                for column in collection.metadata.columns
            },
        )
        condition_rows.append(
            {
                "experiment_id": research_checkpoint.experiment_id,
                "architecture": architecture,
                "training_condition": training_condition,
                "condition": condition.name,
                "snr_db": condition.snr_db,
                **classification.summary,
            }
        )
        condition_outputs[condition.name] = {
            "snr_db": condition.snr_db,
            "metrics": classification.summary,
            "files": {
                name: str(path) for name, path in artifact_paths.items()
            },
        }
        logger.info(
            "Condition %s complete: accuracy=%.4f macro_f1=%.4f",
            condition.name,
            classification.summary["accuracy"],
            classification.summary["macro_f1"],
        )

    condition_table = pd.DataFrame(condition_rows)
    robustness = calculate_robustness_metrics(
        condition_table,
        group_columns=(
            "experiment_id",
            "architecture",
            "training_condition",
        ),
    )
    robustness_paths = save_robustness_analysis(
        robustness,
        output_directory,
    )
    smoke_run = arguments.max_samples is not None or arguments.split != "test"
    protocol = {
        "source_manifest": str(project_config.source_path),
        "checkpoint": str(research_checkpoint.path),
        "checkpoint_epoch": research_checkpoint.epoch,
        "checkpoint_best_validation_metric": research_checkpoint.best_metric,
        "checkpoint_saved_evaluation_ignored": True,
        "split": arguments.split,
        "folds": dataset_settings["folds"][arguments.split],
        "evaluation": evaluation_settings,
        "resolved_noise_directory": str(noise_directory),
        "noise_file_count": len(noise_paths),
        "sample_limit": arguments.max_samples,
        "corruption_is_deterministic_per_sample": True,
    }
    protocol_path = output_directory / "evaluation_protocol.json"
    protocol_path.write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    summary = {
        "experiment_id": research_checkpoint.experiment_id,
        "architecture": architecture,
        "training_condition": training_condition,
        "checkpoint": str(research_checkpoint.path),
        "checkpoint_epoch": research_checkpoint.epoch,
        "split": arguments.split,
        "smoke_run": smoke_run,
        "num_samples": len(records),
        "device": describe_device(device),
        "elapsed_seconds": time.perf_counter() - started_at,
        "noise_directory": str(noise_directory),
        "noise_file_count": len(noise_paths),
        "conditions": condition_outputs,
        "robustness_metrics": robustness.summary.iloc[0].to_dict(),
        "robustness_files": {
            name: str(path) for name, path in robustness_paths.items()
        },
        "evaluation_protocol": str(protocol_path),
    }
    summary_path = output_directory / "evaluation_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger.info("Robustness evaluation complete: %s", summary_path)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if smoke_run:
        print(
            "Smoke/validation evaluation only verifies execution; "
            "it is not a final test result."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
