"""Tests for deterministic waveform preprocessing."""

import pytest
import torch

from urban_sound_robustness.audio.preprocessing import (
    AudioPreprocessingError,
    convert_to_mono,
)


def test_convert_to_mono_averages_channels_without_mutation() -> None:
    """Stereo samples should be averaged frame by frame into one channel."""
    waveform = torch.tensor(
        [[1.0, -1.0, 0.5], [-1.0, 0.5, 0.5]],
        dtype=torch.float32,
    )
    original = waveform.clone()

    mono = convert_to_mono(waveform)

    torch.testing.assert_close(mono, torch.tensor([[0.0, -0.25, 0.5]]))
    torch.testing.assert_close(waveform, original)
    assert mono.shape == (1, 3)


def test_convert_to_mono_returns_valid_mono_input_unchanged() -> None:
    """Already-mono audio should not incur an allocation or numerical change."""
    waveform = torch.tensor([[0.1, -0.2, 0.3]], dtype=torch.float32)

    mono = convert_to_mono(waveform)

    assert mono is waveform


def test_convert_to_mono_preserves_dtype_and_gradient_flow() -> None:
    """Channel averaging should remain differentiable and dtype preserving."""
    waveform = torch.arange(12, dtype=torch.float64).reshape(3, 4).requires_grad_()

    mono = convert_to_mono(waveform)
    mono.sum().backward()

    assert mono.dtype == torch.float64
    assert mono.shape == (1, 4)
    torch.testing.assert_close(waveform.grad, torch.full_like(waveform, 1 / 3))


@pytest.mark.parametrize(
    ("waveform", "message"),
    [
        (torch.ones(4), "shape"),
        (torch.empty(0, 4), "one channel"),
        (torch.empty(2, 0), "one frame"),
        (torch.ones(2, 4, dtype=torch.int16), "floating-point"),
        (torch.tensor([[0.0, float("nan")]]), "non-finite"),
    ],
)
def test_convert_to_mono_rejects_invalid_waveforms(
    waveform: torch.Tensor,
    message: str,
) -> None:
    """Invalid waveform contracts must fail before downstream processing."""
    with pytest.raises(AudioPreprocessingError, match=message):
        convert_to_mono(waveform)
