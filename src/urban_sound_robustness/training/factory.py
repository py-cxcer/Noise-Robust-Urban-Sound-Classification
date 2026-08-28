"""Loss, optimizer, and scheduler construction from training configuration."""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn


def create_loss_function(settings: Mapping[str, Any]) -> nn.Module:
    """Create the configured single-label classification loss."""
    name = str(settings.get("name", "")).lower()
    if name == "cross_entropy":
        return nn.CrossEntropyLoss()
    raise ValueError(f"Unsupported loss function: {name!r}.")


def create_optimizer(
    parameters,
    settings: Mapping[str, Any],
) -> torch.optim.Optimizer:
    """Create Adam or AdamW from readable configuration values."""
    name = str(settings.get("name", "")).lower()
    common = {
        "lr": float(settings["learning_rate"]),
        "weight_decay": float(settings.get("weight_decay", 0.0)),
    }
    if name == "adam":
        return torch.optim.Adam(parameters, **common)
    if name == "adamw":
        return torch.optim.AdamW(parameters, **common)
    raise ValueError(f"Unsupported optimizer: {name!r}.")


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    settings: Mapping[str, Any] | None,
):
    """Create an optional plateau or cosine learning-rate scheduler."""
    if not settings:
        return None
    name = str(settings.get("name", "none")).lower()
    if name in {"none", "disabled"}:
        return None
    if name == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=str(settings.get("mode", "max")),
            factor=float(settings.get("factor", 0.5)),
            patience=int(settings.get("patience", 3)),
        )
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(settings["t_max"]),
            eta_min=float(settings.get("minimum_learning_rate", 0.0)),
        )
    raise ValueError(f"Unsupported scheduler: {name!r}.")
