"""Readable 2D CNN baseline for Log-Mel spectrogram classification."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn


class ConvBlock(nn.Sequential):
    """Convolution, normalization, activation, and spatial downsampling."""

    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__(
            nn.Conv2d(
                input_channels,
                output_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )


class AudioCNN(nn.Module):
    """Compact CNN baseline accepting [batch, channel, mel, time] tensors."""

    def __init__(
        self,
        *,
        input_channels: int,
        num_classes: int,
        channels: Sequence[int],
        dropout: float,
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        current_channels = input_channels
        for output_channels in channels:
            blocks.append(ConvBlock(current_channels, output_channels))
            current_channels = output_channels
        self.feature_extractor = nn.Sequential(*blocks)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(current_channels, num_classes),
        )

    def forward(self, features: Tensor) -> Tensor:
        """Return unnormalized class logits shaped [batch, num_classes]."""
        if features.ndim != 4:
            raise ValueError(
                "AudioCNN expects [batch, channels, mel, time] input."
            )
        encoded = self.feature_extractor(features)
        pooled = self.global_pool(encoded)
        return self.classifier(pooled)
