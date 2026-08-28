"""One-epoch training and validation loops for structured audio batches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import torch
from torch import Tensor, nn

from urban_sound_robustness.evaluation import calculate_classification_metrics


@dataclass(frozen=True)
class EpochResult:
    """Loss and classification metrics collected over one data-loader pass."""

    loss: float
    metrics: dict[str, float]
    targets: tuple[int, ...]
    predictions: tuple[int, ...]
    num_batches: int


def run_epoch(
    model: nn.Module,
    data_loader: Iterable[Mapping[str, object]],
    loss_function: nn.Module,
    device: torch.device,
    class_names: Sequence[str],
    *,
    optimizer: torch.optim.Optimizer | None = None,
    gradient_scaler: torch.amp.GradScaler | None = None,
    mixed_precision: bool = False,
    gradient_accumulation_steps: int = 1,
    max_batches: int | None = None,
) -> EpochResult:
    """Run one training or evaluation epoch using the common batch structure."""
    training = optimizer is not None
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be at least one.")
    if max_batches is not None and max_batches < 1:
        raise ValueError("max_batches must be at least one when provided.")

    model.train(training)
    amp_enabled = mixed_precision and device.type == "cuda"
    if training:
        optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    total_samples = 0
    targets: list[int] = []
    predictions: list[int] = []
    processed_batches = 0
    for batch_index, batch in enumerate(data_loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        features = batch.get("features")
        labels = batch.get("label")
        if not isinstance(features, Tensor) or not isinstance(labels, Tensor):
            raise TypeError("Batches require tensor 'features' and 'label' values.")
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(features)
                loss = loss_function(logits, labels)
                backward_loss = loss / gradient_accumulation_steps
            if training:
                if gradient_scaler is not None and amp_enabled:
                    gradient_scaler.scale(backward_loss).backward()
                else:
                    backward_loss.backward()

        batch_size = int(labels.shape[0])
        total_loss += float(loss.detach().item()) * batch_size
        total_samples += batch_size
        targets.extend(labels.detach().cpu().tolist())
        predictions.extend(logits.detach().argmax(dim=1).cpu().tolist())
        processed_batches += 1

        should_step = processed_batches % gradient_accumulation_steps == 0
        if training and should_step:
            if gradient_scaler is not None and amp_enabled:
                gradient_scaler.step(optimizer)
                gradient_scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    if processed_batches == 0:
        raise ValueError("The data loader produced no batches.")
    if training and processed_batches % gradient_accumulation_steps != 0:
        if gradient_scaler is not None and amp_enabled:
            gradient_scaler.step(optimizer)
            gradient_scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    classification = calculate_classification_metrics(
        targets,
        predictions,
        class_names,
    )
    return EpochResult(
        loss=total_loss / total_samples,
        metrics=classification.summary,
        targets=tuple(targets),
        predictions=tuple(predictions),
        num_batches=processed_batches,
    )
