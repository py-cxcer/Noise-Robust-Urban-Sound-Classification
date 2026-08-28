"""Shape and gradient tests for all common-interface audio classifiers."""

import pytest
import torch

from urban_sound_robustness.models import (
    AudioCNN,
    AudioCRNN,
    AudioResNet18,
    count_trainable_parameters,
    create_model,
)


@pytest.mark.parametrize(
    "settings",
    [
        {
            "name": "cnn",
            "input_channels": 1,
            "num_classes": 10,
            "channels": [8, 16],
            "dropout": 0.1,
        },
        {
            "name": "crnn",
            "input_channels": 1,
            "num_classes": 10,
            "cnn_channels": [8, 16],
            "recurrent_type": "gru",
            "recurrent_hidden_size": 12,
            "recurrent_layers": 1,
            "bidirectional": True,
            "dropout": 0.1,
        },
        {
            "name": "resnet18",
            "input_channels": 1,
            "num_classes": 10,
            "pretrained": False,
            "dropout": 0.1,
        },
    ],
)
def test_all_models_return_common_logits_shape(settings: dict) -> None:
    """Every architecture should accept one-channel Log-Mel batches."""
    model = create_model(settings).eval()
    features = torch.randn(2, 1, 64, 173)

    with torch.inference_mode():
        logits = model(features)

    assert logits.shape == (2, 10)
    assert torch.isfinite(logits).all()
    assert count_trainable_parameters(model) > 0


def test_cnn_supports_backpropagation() -> None:
    """The simple baseline should propagate a classification loss."""
    model = AudioCNN(
        input_channels=1,
        num_classes=10,
        channels=[8, 16],
        dropout=0.1,
    )
    logits = model(torch.randn(2, 1, 32, 64))
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([1, 2]))

    loss.backward()

    assert any(parameter.grad is not None for parameter in model.parameters())


def test_crnn_accepts_variable_time_dimensions() -> None:
    """Temporal pooling should not hardcode the configured 173-frame length."""
    model = AudioCRNN(
        input_channels=1,
        num_classes=10,
        cnn_channels=[8, 16],
        recurrent_hidden_size=12,
        recurrent_layers=1,
        bidirectional=True,
        dropout=0.1,
    ).eval()

    with torch.inference_mode():
        short_logits = model(torch.randn(2, 1, 64, 128))
        long_logits = model(torch.randn(2, 1, 64, 200))

    assert short_logits.shape == long_logits.shape == (2, 10)


def test_resnet_adapts_first_convolution_to_mono() -> None:
    """The torchvision backbone should receive one spectrogram channel."""
    model = AudioResNet18(
        input_channels=1,
        num_classes=10,
        pretrained=False,
        dropout=0.1,
    )

    assert model.backbone.conv1.in_channels == 1
    assert model.backbone.fc[-1].out_features == 10


def test_factory_rejects_unknown_model() -> None:
    """Unsupported architecture names should fail before training."""
    with pytest.raises(ValueError, match="Unsupported model"):
        create_model(
            {
                "name": "unknown",
                "input_channels": 1,
                "num_classes": 10,
                "dropout": 0.1,
            }
        )
