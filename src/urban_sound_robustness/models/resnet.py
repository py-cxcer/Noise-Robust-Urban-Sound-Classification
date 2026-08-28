"""ResNet18 adaptation for single-channel audio spectrograms."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torchvision.models import ResNet18_Weights, resnet18


class AudioResNet18(nn.Module):
    """Adapt torchvision ResNet18 from RGB images to spectrogram tensors."""

    def __init__(
        self,
        *,
        input_channels: int,
        num_classes: int,
        pretrained: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        original_convolution = backbone.conv1
        if input_channels != original_convolution.in_channels:
            replacement = nn.Conv2d(
                input_channels,
                original_convolution.out_channels,
                kernel_size=original_convolution.kernel_size,
                stride=original_convolution.stride,
                padding=original_convolution.padding,
                bias=False,
            )
            if pretrained:
                with torch.no_grad():
                    rgb_weights = original_convolution.weight
                    if input_channels == 1:
                        replacement.weight.copy_(rgb_weights.mean(dim=1, keepdim=True))
                    else:
                        repeated = rgb_weights.mean(dim=1, keepdim=True).repeat(
                            1, input_channels, 1, 1
                        )
                        replacement.weight.copy_(repeated)
            backbone.conv1 = replacement
        classifier_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(classifier_features, num_classes),
        )
        self.backbone = backbone
        self.pretrained = pretrained

    def forward(self, features: Tensor) -> Tensor:
        """Return unnormalized logits for a spectrogram batch."""
        if features.ndim != 4:
            raise ValueError(
                "AudioResNet18 expects [batch, channels, mel, time] input."
            )
        return self.backbone(features)
