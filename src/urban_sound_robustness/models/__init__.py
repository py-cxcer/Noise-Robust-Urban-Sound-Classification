"""Audio classification model architectures and model construction."""

from urban_sound_robustness.models.cnn import AudioCNN
from urban_sound_robustness.models.crnn import AudioCRNN
from urban_sound_robustness.models.factory import (
    count_trainable_parameters,
    create_model,
)
from urban_sound_robustness.models.resnet import AudioResNet18

__all__ = [
    "AudioCNN",
    "AudioCRNN",
    "AudioResNet18",
    "count_trainable_parameters",
    "create_model",
]
