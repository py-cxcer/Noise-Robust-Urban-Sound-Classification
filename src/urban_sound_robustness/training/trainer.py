"""Training orchestration, checkpoints, history, and early stopping."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from urban_sound_robustness.training.engine import EpochResult, run_epoch
from urban_sound_robustness.training.factory import (
    create_loss_function,
    create_optimizer,
    create_scheduler,
)


LOGGER = logging.getLogger("urban_sound_robustness.training")
CHECKPOINT_VERSION = 2


class EarlyStopping:
    """Track a monitored value and report when patience is exhausted."""

    def __init__(self, *, mode: str, patience: int) -> None:
        if mode not in {"min", "max"}:
            raise ValueError("Early-stopping mode must be min or max.")
        if patience < 1:
            raise ValueError("Early-stopping patience must be at least one.")
        self.mode = mode
        self.patience = patience
        self.best_value: float | None = None
        self.bad_epochs = 0

    def update(self, value: float) -> bool:
        """Return true when the current value exhausts configured patience."""
        improved = (
            self.best_value is None
            or self.mode == "max"
            and value > self.best_value
            or self.mode == "min"
            and value < self.best_value
        )
        if improved:
            self.best_value = value
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience

    def state_dict(self) -> dict[str, Any]:
        """Return the state required to continue patience accounting."""
        return {
            "mode": self.mode,
            "patience": self.patience,
            "best_value": self.best_value,
            "bad_epochs": self.bad_epochs,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore state while rejecting a changed stopping configuration."""
        if state.get("mode") != self.mode or int(state["patience"]) != self.patience:
            raise ValueError(
                "The resume checkpoint uses different early-stopping settings."
            )
        best_value = state.get("best_value")
        self.best_value = None if best_value is None else float(best_value)
        self.bad_epochs = int(state["bad_epochs"])


@dataclass(frozen=True)
class TrainingOutcome:
    """Paths and state produced by a completed training call."""

    history: pd.DataFrame
    history_path: Path
    best_checkpoint: Path | None
    last_checkpoint: Path | None
    best_metric: float | None
    epochs_completed: int
    stopped_early: bool


def save_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Mapping[str, float],
    configuration: Mapping[str, Any] | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler
    | torch.optim.lr_scheduler.ReduceLROnPlateau
    | None = None,
    gradient_scaler: torch.amp.GradScaler | None = None,
    early_stopping: EarlyStopping | None = None,
    best_metric: float | None = None,
    history: Sequence[Mapping[str, float | int]] | None = None,
    train_loader=None,
    validation_loader=None,
) -> Path:
    """Atomically save all state required to restart at an epoch boundary."""
    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = resolved.with_name(f".{resolved.name}.tmp")
    checkpoint = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "epoch": epoch,
        "model_class": model.__class__.__name__,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": (
            None if scheduler is None else scheduler.state_dict()
        ),
        "gradient_scaler_state_dict": (
            None if gradient_scaler is None else gradient_scaler.state_dict()
        ),
        "early_stopping_state_dict": (
            None if early_stopping is None else early_stopping.state_dict()
        ),
        "best_metric": best_metric,
        "metrics": dict(metrics),
        "history": [] if history is None else [dict(row) for row in history],
        "configuration": None if configuration is None else dict(configuration),
        "random_states": _capture_random_states(),
        "data_loader_generator_states": {
            "train": _capture_loader_generator_state(train_loader),
            "validation": _capture_loader_generator_state(validation_loader),
        },
    }
    try:
        torch.save(checkpoint, temporary_path)
        os.replace(temporary_path, resolved)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return resolved


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler
    | torch.optim.lr_scheduler.ReduceLROnPlateau
    | None = None,
    gradient_scaler: torch.amp.GradScaler | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Restore model state and optional training components from a checkpoint."""
    checkpoint = torch.load(
        Path(path).expanduser().resolve(),
        map_location=map_location,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler_state = checkpoint.get("scheduler_state_dict")
    if scheduler is not None and scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)
    scaler_state = checkpoint.get("gradient_scaler_state_dict")
    if gradient_scaler is not None and scaler_state is not None:
        gradient_scaler.load_state_dict(scaler_state)
    return checkpoint


def _capture_random_states() -> dict[str, Any]:
    """Capture every process-level random generator used by the project."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_random_states(states: Mapping[str, Any]) -> None:
    """Restore process-level random generators saved at an epoch boundary."""
    random.setstate(states["python"])
    np.random.set_state(states["numpy"])
    torch.set_rng_state(states["torch"].cpu())
    cuda_states = states.get("cuda")
    if cuda_states is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([state.cpu() for state in cuda_states])


def _capture_loader_generator_state(data_loader) -> torch.Tensor | None:
    """Capture a DataLoader generator without depending on dataset type."""
    if data_loader is None:
        return None
    generator = getattr(data_loader, "generator", None)
    if not isinstance(generator, torch.Generator):
        return None
    return generator.get_state()


def _restore_loader_generator_state(data_loader, state: torch.Tensor | None) -> None:
    """Restore deterministic shuffling and worker-seed progression."""
    if state is None:
        return
    generator = getattr(data_loader, "generator", None)
    if not isinstance(generator, torch.Generator):
        raise ValueError(
            "The checkpoint contains a DataLoader generator state, but the "
            "current DataLoader has no generator."
        )
    generator.set_state(state.cpu())


class Trainer:
    """Coordinate reusable epoch loops and experiment-state persistence."""

    def __init__(
        self,
        model: nn.Module,
        class_names: Sequence[str],
        training_settings: Mapping[str, Any],
        *,
        device: torch.device,
        checkpoint_directory: str | Path,
        history_path: str | Path,
        tensorboard_directory: str | Path | None = None,
        configuration: Mapping[str, Any] | None = None,
    ) -> None:
        self.model = model.to(device)
        self.class_names = tuple(class_names)
        self.settings = dict(training_settings)
        self.device = device
        self.checkpoint_directory = (
            Path(checkpoint_directory).expanduser().resolve()
        )
        self.history_path = Path(history_path).expanduser().resolve()
        self.tensorboard_directory = (
            None
            if tensorboard_directory is None
            else Path(tensorboard_directory).expanduser().resolve()
        )
        self.configuration = configuration
        self.loss_function = create_loss_function(self.settings["loss"])
        self.optimizer = create_optimizer(
            self.model.parameters(), self.settings["optimizer"]
        )
        self.scheduler = create_scheduler(
            self.optimizer, self.settings.get("scheduler")
        )
        amp_enabled = bool(self.settings.get("mixed_precision", False)) and (
            device.type == "cuda"
        )
        self.gradient_scaler = torch.amp.GradScaler(
            "cuda",
            enabled=amp_enabled,
        )

    def fit(
        self,
        train_loader,
        validation_loader,
        *,
        epochs: int | None = None,
        max_train_batches: int | None = None,
        max_validation_batches: int | None = None,
        resume_from: str | Path | None = None,
    ) -> TrainingOutcome:
        """Train or resume at the next epoch with complete state restoration."""
        total_epochs = int(self.settings["epochs"]) if epochs is None else epochs
        if total_epochs < 1:
            raise ValueError("epochs must be at least one.")
        checkpoint_settings = dict(self.settings.get("checkpointing", {}))
        monitor = str(checkpoint_settings.get("monitor", "macro_f1"))
        mode = str(checkpoint_settings.get("mode", "max"))
        save_best_enabled = bool(checkpoint_settings.get("save_best", True))
        save_last_enabled = bool(checkpoint_settings.get("save_last", True))
        early_settings = dict(self.settings.get("early_stopping", {}))
        early_stopping = None
        if early_settings.get("enabled", False):
            early_stopping = EarlyStopping(
                mode=str(early_settings.get("mode", mode)),
                patience=int(early_settings["patience"]),
            )

        history_rows: list[dict[str, float | int]] = []
        best_metric: float | None = None
        best_checkpoint: Path | None = None
        last_checkpoint: Path | None = None
        stopped_early = False
        first_epoch = 1

        if resume_from is not None:
            checkpoint = load_checkpoint(
                resume_from,
                self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                gradient_scaler=self.gradient_scaler,
                map_location="cpu",
            )
            if int(checkpoint.get("checkpoint_version", 0)) != CHECKPOINT_VERSION:
                raise ValueError(
                    "This checkpoint predates complete resume support. Start a "
                    "new run so future last.pt files contain full training state."
                )
            if checkpoint.get("model_class") != self.model.__class__.__name__:
                raise ValueError("The checkpoint model class does not match the model.")
            checkpoint_configuration = checkpoint.get("configuration")
            if checkpoint_configuration is not None and self.configuration is not None:
                for section_name in (
                    "dataset",
                    "audio",
                    "augmentation",
                    "model",
                    "training",
                ):
                    if checkpoint_configuration.get(section_name) != self.configuration.get(
                        section_name
                    ):
                        raise ValueError(
                            "The resume configuration differs in section "
                            f"'{section_name}'."
                        )
            completed_epoch = int(checkpoint["epoch"])
            if completed_epoch > total_epochs:
                raise ValueError(
                    f"Checkpoint epoch {completed_epoch} exceeds requested total "
                    f"epochs {total_epochs}."
                )
            first_epoch = completed_epoch + 1
            history_rows = [dict(row) for row in checkpoint.get("history", [])]
            if len(history_rows) != completed_epoch:
                raise ValueError(
                    "Checkpoint history length does not match its completed epoch."
                )
            saved_best_metric = checkpoint.get("best_metric")
            best_metric = (
                None if saved_best_metric is None else float(saved_best_metric)
            )
            early_state = checkpoint.get("early_stopping_state_dict")
            if early_stopping is not None and early_state is not None:
                early_stopping.load_state_dict(early_state)
            loader_states = checkpoint.get("data_loader_generator_states", {})
            _restore_loader_generator_state(
                train_loader,
                loader_states.get("train"),
            )
            _restore_loader_generator_state(
                validation_loader,
                loader_states.get("validation"),
            )
            random_states = checkpoint.get("random_states")
            if random_states is None:
                raise ValueError("Resume checkpoint is missing random-number state.")
            _restore_random_states(random_states)
            last_checkpoint = Path(resume_from).expanduser().resolve()
            candidate_best = self.checkpoint_directory / "best.pt"
            if candidate_best.is_file():
                best_checkpoint = candidate_best
            if history_rows:
                self.history_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(history_rows).to_csv(self.history_path, index=False)
            LOGGER.info(
                "Resumed checkpoint at epoch %d; continuing with epoch %d/%d",
                completed_epoch,
                first_epoch,
                total_epochs,
            )

        writer = None
        logging_settings = dict(self.settings.get("logging", {}))
        if logging_settings.get("tensorboard", False):
            if self.tensorboard_directory is None:
                raise ValueError(
                    "tensorboard_directory is required when TensorBoard is enabled."
                )
            writer_options = {}
            if resume_from is not None:
                writer_options["purge_step"] = first_epoch
            writer = SummaryWriter(
                self.tensorboard_directory,
                **writer_options,
            )

        def persist_checkpoint(
            path: Path,
            *,
            epoch: int,
            metrics: Mapping[str, float],
        ) -> Path:
            return save_checkpoint(
                path,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                gradient_scaler=self.gradient_scaler,
                early_stopping=early_stopping,
                epoch=epoch,
                metrics=metrics,
                best_metric=best_metric,
                history=history_rows,
                train_loader=train_loader,
                validation_loader=validation_loader,
                configuration=self.configuration,
            )

        if resume_from is None and save_last_enabled:
            last_checkpoint = persist_checkpoint(
                self.checkpoint_directory / "last.pt",
                epoch=0,
                metrics={},
            )
            LOGGER.info("Saved initial recovery checkpoint: %s", last_checkpoint)

        try:
            for epoch in range(first_epoch, total_epochs + 1):
                LOGGER.info("Starting epoch %d/%d", epoch, total_epochs)
                train_result = run_epoch(
                    self.model,
                    train_loader,
                    self.loss_function,
                    self.device,
                    self.class_names,
                    optimizer=self.optimizer,
                    gradient_scaler=self.gradient_scaler,
                    mixed_precision=bool(
                        self.settings.get("mixed_precision", False)
                    ),
                    gradient_accumulation_steps=int(
                        self.settings.get("gradient_accumulation_steps", 1)
                    ),
                    max_batches=max_train_batches,
                )
                validation_result = run_epoch(
                    self.model,
                    validation_loader,
                    self.loss_function,
                    self.device,
                    self.class_names,
                    mixed_precision=bool(
                        self.settings.get("mixed_precision", False)
                    ),
                    max_batches=max_validation_batches,
                )
                learning_rate = float(self.optimizer.param_groups[0]["lr"])
                row = self._history_row(
                    epoch,
                    train_result,
                    validation_result,
                    learning_rate,
                )
                history_rows.append(row)
                monitored_value = float(validation_result.metrics[monitor])
                LOGGER.info(
                    "Epoch %d complete: train_loss=%.4f validation_loss=%.4f "
                    "validation_%s=%.4f",
                    epoch,
                    train_result.loss,
                    validation_result.loss,
                    monitor,
                    monitored_value,
                )
                improved = (
                    best_metric is None
                    or mode == "max"
                    and monitored_value > best_metric
                    or mode == "min"
                    and monitored_value < best_metric
                )
                if improved:
                    best_metric = monitored_value
                if isinstance(
                    self.scheduler,
                    torch.optim.lr_scheduler.ReduceLROnPlateau,
                ):
                    self.scheduler.step(monitored_value)
                elif self.scheduler is not None:
                    self.scheduler.step()
                should_stop = (
                    early_stopping is not None
                    and early_stopping.update(monitored_value)
                )

                history = pd.DataFrame(history_rows)
                self.history_path.parent.mkdir(parents=True, exist_ok=True)
                history.to_csv(self.history_path, index=False)
                if writer is not None:
                    self._write_tensorboard(
                        writer, epoch, train_result, validation_result, learning_rate
                    )
                if improved and save_best_enabled:
                    best_checkpoint = persist_checkpoint(
                        self.checkpoint_directory / "best.pt",
                        epoch=epoch,
                        metrics=validation_result.metrics,
                    )
                    LOGGER.info("Saved best checkpoint: %s", best_checkpoint)
                if save_last_enabled:
                    last_checkpoint = persist_checkpoint(
                        self.checkpoint_directory / "last.pt",
                        epoch=epoch,
                        metrics=validation_result.metrics,
                    )

                if should_stop:
                    stopped_early = True
                    LOGGER.info("Early stopping triggered after epoch %d", epoch)
                    break
        finally:
            if writer is not None:
                writer.close()

        final_history = pd.DataFrame(history_rows)
        return TrainingOutcome(
            history=final_history,
            history_path=self.history_path,
            best_checkpoint=best_checkpoint,
            last_checkpoint=last_checkpoint,
            best_metric=best_metric,
            epochs_completed=len(final_history),
            stopped_early=stopped_early,
        )

    @staticmethod
    def _history_row(
        epoch: int,
        train: EpochResult,
        validation: EpochResult,
        learning_rate: float,
    ) -> dict[str, float | int]:
        row: dict[str, float | int] = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train_loss": train.loss,
            "validation_loss": validation.loss,
        }
        for name, value in train.metrics.items():
            row[f"train_{name}"] = value
        for name, value in validation.metrics.items():
            row[f"validation_{name}"] = value
        return row

    @staticmethod
    def _write_tensorboard(
        writer: SummaryWriter,
        epoch: int,
        train: EpochResult,
        validation: EpochResult,
        learning_rate: float,
    ) -> None:
        writer.add_scalar("loss/train", train.loss, epoch)
        writer.add_scalar("loss/validation", validation.loss, epoch)
        writer.add_scalar("learning_rate", learning_rate, epoch)
        for name, value in train.metrics.items():
            writer.add_scalar(f"metrics/train_{name}", value, epoch)
        for name, value in validation.metrics.items():
            writer.add_scalar(f"metrics/validation_{name}", value, epoch)
