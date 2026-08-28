"""Construct audio classifiers from the common model configuration."""

from __future__ import annotations

from typing import Any, Mapping

from torch import nn

from urban_sound_robustness.models.cnn import AudioCNN
from urban_sound_robustness.models.crnn import AudioCRNN
from urban_sound_robustness.models.resnet import AudioResNet18


def create_model(settings: Mapping[str, Any]) -> nn.Module:
    """Create the selected architecture with a shared logits interface."""
    model_name = str(settings.get("name", "")).lower()
    common = {
        "input_channels": int(settings["input_channels"]),
        "num_classes": int(settings["num_classes"]),
        "dropout": float(settings["dropout"]),
    }
    if model_name == "cnn":
        return AudioCNN(
            **common,
            channels=[int(value) for value in settings["channels"]],
        )
    if model_name == "crnn":
        recurrent_type = str(settings.get("recurrent_type", "gru")).lower()
        if recurrent_type != "gru":
            raise ValueError("AudioCRNN currently supports recurrent_type=gru.")
        return AudioCRNN(
            **common,
            cnn_channels=[int(value) for value in settings["cnn_channels"]],
            recurrent_hidden_size=int(settings["recurrent_hidden_size"]),
            recurrent_layers=int(settings["recurrent_layers"]),
            bidirectional=bool(settings["bidirectional"]),
        )
    if model_name == "resnet18":
        return AudioResNet18(
            **common,
            pretrained=bool(settings["pretrained"]),
        )
    raise ValueError(
        f"Unsupported model '{model_name}'. Choose cnn, crnn, or resnet18."
    )


def count_trainable_parameters(model: nn.Module) -> int:
    """Return the number of parameters updated by optimization."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
