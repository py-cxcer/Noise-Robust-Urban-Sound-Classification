"""Convolutional recurrent classifier with a bidirectional GRU."""

from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor, nn

from urban_sound_robustness.models.cnn import ConvBlock


class AudioCRNN(nn.Module):
    """CNN frequency encoder followed by temporal BiGRU aggregation."""

    def __init__(
        self,
        *,
        input_channels: int,
        num_classes: int,
        cnn_channels: Sequence[int],
        recurrent_hidden_size: int,
        recurrent_layers: int,
        bidirectional: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        current_channels = input_channels
        for output_channels in cnn_channels:
            blocks.append(ConvBlock(current_channels, output_channels))
            current_channels = output_channels
        self.cnn = nn.Sequential(*blocks)
        self.recurrent = nn.GRU(
            input_size=current_channels,
            hidden_size=recurrent_hidden_size,
            num_layers=recurrent_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if recurrent_layers > 1 else 0.0,
        )
        recurrent_output_size = recurrent_hidden_size * (
            2 if bidirectional else 1
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(recurrent_output_size, num_classes),
        )

    def forward(self, features: Tensor) -> Tensor:
        """Preserve time as a sequence and return class logits.

        Shape flow:
        [B, C, Mel, T] -> CNN [B, C2, Mel2, T2] -> frequency mean
        [B, C2, T2] -> transpose [B, T2, C2] -> BiGRU -> temporal mean
        [B, 2H] -> classifier [B, classes].
        """
        if features.ndim != 4:
            raise ValueError(
                "AudioCRNN expects [batch, channels, mel, time] input."
            )
        encoded = self.cnn(features)
        frequency_reduced = encoded.mean(dim=2)
        sequence = frequency_reduced.transpose(1, 2)
        recurrent_output, _ = self.recurrent(sequence)
        pooled_sequence = recurrent_output.mean(dim=1)
        return self.classifier(pooled_sequence)
