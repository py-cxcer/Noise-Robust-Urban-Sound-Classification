"""Tests for configurable waveform and spectrogram augmentation."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from urban_sound_robustness.audio import AudioPreprocessor, measure_snr_db
from urban_sound_robustness.augmentation import (
    AugmentedAudioPreprocessor,
    SpectrogramAugmenter,
    WaveformAugmenter,
)


def _disabled_waveform_settings() -> dict:
    return {
        "time_shift": {"enabled": False},
        "random_gain": {"enabled": False},
        "background_noise": {"enabled": False},
        "pitch_shift": {"enabled": False},
        "time_stretch": {"enabled": False},
    }


def _audio_settings() -> dict:
    return {
        "sample_rate": 8_000,
        "clip_duration_seconds": 0.25,
        "mono": True,
        "length_normalization": {
            "padding_mode": "zero",
            "training_crop": "random",
            "evaluation_crop": "center",
        },
        "representation": "log_mel",
        "log_mel": {
            "n_fft": 128,
            "win_length": 128,
            "hop_length": 64,
            "n_mels": 16,
            "f_min": 0.0,
            "f_max": None,
            "power": 2.0,
            "top_db": 80.0,
            "center": True,
            "pad_mode": "reflect",
            "mel_scale": "htk",
        },
        "mfcc": {"n_mfcc": 8},
        "normalization": {
            "method": "per_example_standardization",
            "epsilon": 1.0e-6,
        },
    }


def test_disabled_waveform_augmentation_returns_input_unchanged(
    tmp_path: Path,
) -> None:
    """Baseline configuration should perform no waveform transformation."""
    augmenter = WaveformAugmenter(
        _disabled_waveform_settings(),
        sample_rate=8_000,
        project_root=tmp_path,
    )
    waveform = torch.randn(1, 100)

    assert augmenter(waveform) is waveform


def test_time_shift_is_seeded_and_preserves_samples(tmp_path: Path) -> None:
    """Circular shifting should be reproducible and preserve the sample multiset."""
    settings = _disabled_waveform_settings()
    settings["time_shift"] = {
        "enabled": True,
        "probability": 1.0,
        "max_shift_fraction": 0.5,
    }
    augmenter = WaveformAugmenter(
        settings,
        sample_rate=8_000,
        project_root=tmp_path,
    )
    waveform = torch.arange(20, dtype=torch.float32).unsqueeze(0)

    first = augmenter(waveform, generator=torch.Generator().manual_seed(5))
    second = augmenter(waveform, generator=torch.Generator().manual_seed(5))

    torch.testing.assert_close(first, second)
    torch.testing.assert_close(first.sort().values, waveform.sort().values)


def test_random_gain_uses_decibel_amplitude_conversion(tmp_path: Path) -> None:
    """A fixed +6.0206 dB gain should approximately double amplitude."""
    settings = _disabled_waveform_settings()
    settings["random_gain"] = {
        "enabled": True,
        "probability": 1.0,
        "min_gain_db": 6.0206,
        "max_gain_db": 6.0206,
    }
    augmenter = WaveformAugmenter(
        settings,
        sample_rate=8_000,
        project_root=tmp_path,
    )

    augmented = augmenter(torch.ones(1, 100))

    torch.testing.assert_close(
        augmented,
        torch.full((1, 100), 2.0),
        rtol=1.0e-4,
        atol=1.0e-4,
    )


def test_background_noise_uses_configured_snr(
    tmp_path: Path,
) -> None:
    """Training background noise should reuse the tested SNR mixer."""
    noise_path = tmp_path / "noise.wav"
    sf.write(noise_path, np.sin(np.linspace(0, 30, 4_000)), 8_000)
    settings = _disabled_waveform_settings()
    settings["background_noise"] = {
        "enabled": True,
        "probability": 1.0,
        "noise_directory": str(tmp_path),
        "min_snr_db": 10.0,
        "max_snr_db": 10.0,
    }
    augmenter = WaveformAugmenter(
        settings,
        sample_rate=8_000,
        project_root=tmp_path,
        noise_paths=[noise_path],
    )
    clean = torch.linspace(-0.5, 0.5, 2_000).unsqueeze(0)

    augmented = augmenter(
        clean,
        generator=torch.Generator().manual_seed(12),
    )

    assert measure_snr_db(clean, augmented) == pytest.approx(10.0, abs=1.0e-4)


def test_spectrogram_masks_are_reproducible_and_non_mutating() -> None:
    """Frequency/time masks should use the supplied generator and clone input."""
    settings = {
        "frequency_mask": {
            "enabled": True,
            "probability": 1.0,
            "max_mask_bins": 4,
        },
        "time_mask": {
            "enabled": True,
            "probability": 1.0,
            "max_mask_frames": 6,
        },
    }
    augmenter = SpectrogramAugmenter(settings)
    features = torch.ones(1, 8, 20)

    first = augmenter(features, generator=torch.Generator().manual_seed(9))
    second = augmenter(features, generator=torch.Generator().manual_seed(9))

    torch.testing.assert_close(first, second)
    torch.testing.assert_close(features, torch.ones_like(features))
    assert bool((first == 0).any())


def test_augmented_preprocessor_is_deterministic_in_evaluation(
    tmp_path: Path,
) -> None:
    """Validation/test calls must bypass every augmentation."""
    base = AudioPreprocessor(_audio_settings())
    settings = {
        "waveform": _disabled_waveform_settings(),
        "spectrogram": {
            "frequency_mask": {
                "enabled": True,
                "probability": 1.0,
                "max_mask_bins": 8,
            },
            "time_mask": {
                "enabled": True,
                "probability": 1.0,
                "max_mask_frames": 8,
            },
        },
    }
    augmented = AugmentedAudioPreprocessor(
        AudioPreprocessor(_audio_settings()),
        settings,
        project_root=tmp_path,
    )
    waveform = torch.randn(1, 2_000)

    baseline_output = base(waveform, 8_000, training=False)
    augmented_output = augmented(waveform, 8_000, training=False)

    torch.testing.assert_close(augmented_output.waveform, baseline_output.waveform)
    torch.testing.assert_close(augmented_output.features, baseline_output.features)


def test_augmented_preprocessor_repeats_with_equal_seeds(tmp_path: Path) -> None:
    """Training transformations should be exactly repeatable from a seed."""
    settings = {
        "waveform": {
            **_disabled_waveform_settings(),
            "time_shift": {
                "enabled": True,
                "probability": 1.0,
                "max_shift_fraction": 0.25,
            },
        },
        "spectrogram": {
            "frequency_mask": {
                "enabled": True,
                "probability": 1.0,
                "max_mask_bins": 4,
            },
            "time_mask": {"enabled": False},
        },
    }
    preprocessor = AugmentedAudioPreprocessor(
        AudioPreprocessor(_audio_settings()),
        settings,
        project_root=tmp_path,
    )
    waveform = torch.randn(1, 2_000)

    first = preprocessor(
        waveform,
        8_000,
        training=True,
        generator=torch.Generator().manual_seed(22),
    )
    second = preprocessor(
        waveform,
        8_000,
        training=True,
        generator=torch.Generator().manual_seed(22),
    )

    torch.testing.assert_close(first.waveform, second.waveform)
    torch.testing.assert_close(first.features, second.features)
