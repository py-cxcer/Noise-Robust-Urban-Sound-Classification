"""Train any configured audio classifier and preserve reproducible run artifacts."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader, Subset

from urban_sound_robustness.datasets import (
    create_dataset_adapter,
    create_preprocessed_dataset,
)
from urban_sound_robustness.evaluation import (
    calculate_classification_metrics,
    collect_model_predictions,
    save_classification_result,
)
from urban_sound_robustness.models import count_trainable_parameters, create_model
from urban_sound_robustness.training import Trainer, load_checkpoint
from urban_sound_robustness.utils import (
    build_experiment_id,
    configure_logging,
    create_data_loader_generator,
    create_experiment_layout,
    describe_device,
    load_experiment_config,
    load_experiment_layout,
    seed_data_loader_worker,
    seed_everything,
    select_device,
)


def parse_arguments() -> argparse.Namespace:
    """Parse the experiment manifest and optional smoke-run limits."""
    parser = argparse.ArgumentParser(
        description="Train one baseline or augmented audio-classification model."
    )
    parser.add_argument(
        "config",
        nargs="?",
        type=Path,
        default=Path("configs/experiment/development.yaml"),
        help="Composed experiment YAML manifest.",
    )
    parser.add_argument(
        "--run-label",
        default=None,
        help="Optional label such as smoke, run01, or fold10.",
    )
    parser.add_argument(
        "--experiment-id",
        default=None,
        help="Exact output ID. Existing experiment paths are never overwritten.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Continue the same run from its atomic last.pt checkpoint.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Total target epoch count, including already completed epochs.",
    )
    parser.add_argument("--device", default=None, help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-validation-samples", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-validation-batches", type=int, default=None)
    return parser.parse_args()


def _bounded_subset(dataset, maximum_samples: int | None) -> Subset:
    """Return a deterministic prefix subset for smoke testing or all samples."""
    if maximum_samples is not None and maximum_samples < 1:
        raise ValueError("Sample limits must be at least one when provided.")
    subset_size = len(dataset)
    if maximum_samples is not None:
        subset_size = min(subset_size, maximum_samples)
    if subset_size == 0:
        raise ValueError("The selected dataset split contains no samples.")
    return Subset(dataset, range(subset_size))


def _create_data_loaders(
    project_config,
    *,
    device: torch.device,
    num_workers_override: int | None,
    max_train_samples: int | None,
    max_validation_samples: int | None,
) -> tuple[DataLoader, DataLoader, list[str]]:
    """Create train/validation loaders using only configured fold assignments."""
    dataset_settings = project_config.section("dataset")
    training_settings = project_config.section("training")
    adapter = create_dataset_adapter(dataset_settings, project_config.project_root)

    train_dataset = create_preprocessed_dataset(
        adapter,
        project_config.section("audio"),
        "train",
        augmentation_settings=project_config.section("augmentation"),
        project_root=project_config.project_root,
    )
    validation_dataset = create_preprocessed_dataset(
        adapter,
        project_config.section("audio"),
        "validation",
        training=False,
    )
    train_subset = _bounded_subset(train_dataset, max_train_samples)
    validation_subset = _bounded_subset(
        validation_dataset,
        max_validation_samples,
    )

    num_workers = int(training_settings["num_workers"])
    if num_workers_override is not None:
        if num_workers_override < 0:
            raise ValueError("--num-workers cannot be negative.")
        num_workers = num_workers_override
    seed = int(training_settings["seed"])
    common_loader_settings = {
        "batch_size": int(training_settings["batch_size"]),
        "num_workers": num_workers,
        "pin_memory": bool(training_settings["pin_memory"]) and device.type == "cuda",
        "worker_init_fn": seed_data_loader_worker,
    }
    train_loader = DataLoader(
        train_subset,
        shuffle=True,
        generator=create_data_loader_generator(seed),
        **common_loader_settings,
    )
    validation_loader = DataLoader(
        validation_subset,
        shuffle=False,
        generator=create_data_loader_generator(seed + 1),
        **common_loader_settings,
    )
    validation_ids = [
        validation_dataset.records[index].sample_id
        for index in validation_subset.indices
    ]
    return train_loader, validation_loader, validation_ids


def _prepare_resume(arguments: argparse.Namespace, project_config):
    """Validate a resume request and restore its original runtime limits."""
    if arguments.resume is None:
        return None, None
    if arguments.experiment_id is not None or arguments.run_label is not None:
        raise ValueError(
            "--resume cannot be combined with --experiment-id or --run-label; "
            "the original experiment identity is preserved."
        )
    checkpoint_path = arguments.resume.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if int(checkpoint.get("checkpoint_version", 0)) != 2:
        raise ValueError(
            "This checkpoint predates complete resume support. Start a new run."
        )
    saved_configuration = checkpoint.get("configuration")
    if not isinstance(saved_configuration, dict):
        raise ValueError("Resume checkpoint has no saved experiment configuration.")
    for section_name in ("dataset", "audio", "augmentation", "model", "training"):
        if saved_configuration.get(section_name) != project_config.data.get(
            section_name
        ):
            raise ValueError(
                f"The supplied manifest differs from the checkpoint in "
                f"'{section_name}'."
            )

    saved_runtime = dict(saved_configuration.get("runtime_overrides", {}))
    saved_options = {
        "num_workers": saved_runtime.get(
            "effective_num_workers",
            saved_configuration["training"]["num_workers"],
        ),
        "max_train_samples": saved_runtime.get("max_train_samples"),
        "max_validation_samples": saved_runtime.get("max_validation_samples"),
        "max_train_batches": saved_runtime.get("max_train_batches"),
        "max_validation_batches": saved_runtime.get("max_validation_batches"),
    }
    for option_name, saved_value in saved_options.items():
        requested_value = getattr(arguments, option_name)
        if requested_value is None:
            setattr(arguments, option_name, saved_value)
        elif requested_value != saved_value:
            raise ValueError(
                f"--{option_name.replace('_', '-')} cannot change while resuming "
                f"(checkpoint={saved_value}, requested={requested_value})."
            )

    saved_epochs = saved_runtime.get("effective_epochs")
    if saved_epochs is None:
        saved_epochs = saved_runtime.get("epochs")
    if saved_epochs is None:
        saved_epochs = saved_configuration["training"]["epochs"]
    if arguments.epochs is None:
        arguments.epochs = int(saved_epochs)
    if arguments.epochs < int(checkpoint["epoch"]):
        raise ValueError(
            f"Requested total epochs {arguments.epochs} is below checkpoint "
            f"epoch {checkpoint['epoch']}."
        )
    return checkpoint_path, saved_configuration


def main() -> None:
    """Run training, restore the best checkpoint, and save validation outputs."""
    arguments = parse_arguments()
    started_at = time.perf_counter()
    project_config = load_experiment_config(arguments.config)
    resume_checkpoint, saved_configuration = _prepare_resume(
        arguments,
        project_config,
    )
    training_settings = project_config.section("training")
    seed_everything(
        int(training_settings["seed"]),
        deterministic=bool(training_settings["deterministic"]),
    )
    device_preference = arguments.device or str(training_settings["device"])
    device = select_device(device_preference)

    train_loader, validation_loader, validation_ids = _create_data_loaders(
        project_config,
        device=device,
        num_workers_override=arguments.num_workers,
        max_train_samples=arguments.max_train_samples,
        max_validation_samples=arguments.max_validation_samples,
    )
    model_settings = project_config.section("model")
    augmentation_settings = project_config.section("augmentation")
    effective_epochs = (
        int(training_settings["epochs"])
        if arguments.epochs is None
        else arguments.epochs
    )
    effective_num_workers = (
        int(training_settings["num_workers"])
        if arguments.num_workers is None
        else arguments.num_workers
    )
    runtime_overrides = {
        "source_manifest": str(project_config.source_path),
        "epochs": arguments.epochs,
        "effective_epochs": effective_epochs,
        "device": arguments.device,
        "num_workers": arguments.num_workers,
        "effective_num_workers": effective_num_workers,
        "max_train_samples": arguments.max_train_samples,
        "max_validation_samples": arguments.max_validation_samples,
        "max_train_batches": arguments.max_train_batches,
        "max_validation_batches": arguments.max_validation_batches,
        "run_label": arguments.run_label,
        "resume_from": (
            None if resume_checkpoint is None else str(resume_checkpoint)
        ),
    }
    if resume_checkpoint is None:
        run_configuration = deepcopy(project_config.data)
        run_configuration["runtime_overrides"] = runtime_overrides
        experiment_id = arguments.experiment_id or build_experiment_id(
            str(model_settings["name"]),
            str(augmentation_settings["name"]),
            arguments.run_label,
        )
        paths = create_experiment_layout(
            experiment_id,
            project_config.section("paths"),
            project_config.project_root,
            run_configuration,
        )
    else:
        run_configuration = deepcopy(saved_configuration)
        experiment_id = resume_checkpoint.parent.name
        paths = load_experiment_layout(
            experiment_id,
            project_config.section("paths"),
            project_config.project_root,
        )
        expected_checkpoint = paths.checkpoint_directory / "last.pt"
        if resume_checkpoint != expected_checkpoint:
            raise ValueError(
                "Resume must use the original run's checkpoints/<experiment-id>/"
                "last.pt file."
            )
    logger = configure_logging(paths.log_directory / "training.log")
    model = create_model(model_settings)
    parameter_count = count_trainable_parameters(model)
    logger.info("Experiment: %s", experiment_id)
    if resume_checkpoint is not None:
        logger.info("Resume checkpoint: %s", resume_checkpoint)
    logger.info("Device: %s", describe_device(device))
    logger.info(
        "Data: %d training samples, %d validation samples",
        len(train_loader.dataset),
        len(validation_loader.dataset),
    )
    logger.info("Model: %s (%d trainable parameters)", model.__class__.__name__, parameter_count)

    trainer = Trainer(
        model,
        project_config.section("dataset")["class_names"],
        training_settings,
        device=device,
        checkpoint_directory=paths.checkpoint_directory,
        history_path=paths.metrics_directory / "training_history.csv",
        tensorboard_directory=paths.log_directory / "tensorboard",
        configuration=run_configuration,
    )
    outcome = trainer.fit(
        train_loader,
        validation_loader,
        epochs=effective_epochs,
        max_train_batches=arguments.max_train_batches,
        max_validation_batches=arguments.max_validation_batches,
        resume_from=resume_checkpoint,
    )
    if outcome.best_checkpoint is not None:
        load_checkpoint(outcome.best_checkpoint, model, map_location=device)

    targets, predictions = collect_model_predictions(model, validation_loader, device)
    classification = calculate_classification_metrics(
        targets,
        predictions,
        project_config.section("dataset")["class_names"],
    )
    evaluation_paths = save_classification_result(
        classification,
        paths.metrics_directory / "validation",
        class_names=project_config.section("dataset")["class_names"],
        sample_ids=validation_ids,
    )
    summary = {
        "experiment_id": experiment_id,
        "runtime_overrides": runtime_overrides,
        "smoke_run": any(
            value is not None
            for value in (
                arguments.max_train_samples,
                arguments.max_validation_samples,
                arguments.max_train_batches,
                arguments.max_validation_batches,
            )
        ),
        "device": describe_device(device),
        "train_samples": len(train_loader.dataset),
        "validation_samples": len(validation_loader.dataset),
        "trainable_parameters": parameter_count,
        "epochs_completed": outcome.epochs_completed,
        "stopped_early": outcome.stopped_early,
        "best_validation_metric": outcome.best_metric,
        "validation_metrics": classification.summary,
        "elapsed_seconds": time.perf_counter() - started_at,
        "history": str(outcome.history_path),
        "best_checkpoint": (
            None if outcome.best_checkpoint is None else str(outcome.best_checkpoint)
        ),
        "evaluation_files": {
            name: str(path) for name, path in evaluation_paths.items()
        },
    }
    summary_path = paths.experiment_directory / "run_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger.info("Experiment complete. Summary: %s", summary_path)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["smoke_run"]:
        print("Smoke-run metrics only verify execution; they are not research results.")


if __name__ == "__main__":
    main()
