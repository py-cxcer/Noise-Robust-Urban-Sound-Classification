"""Tests for resampling, duration normalization, features, and dataset integration."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from torch.utils.data import DataLoader

from urban_sound_robustness.audio import (
    AudioPreprocessingError,
    AudioPreprocessor,
    LogMelFeatureExtractor,
    MFCCFeatureExtractor,
    normalize_waveform_duration,
    normalize_waveform_length,
    resample_waveform,
    standardize_features,
)
from urban_sound_robustness.datasets import (
    AudioSampleRecord,
    PreprocessedAudioDataset,
    create_preprocessed_dataset,
)


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


@pytest.mark.parametrize(
    ("source_rate", "target_rate", "source_frames", "expected_frames"),
    [(8_000, 16_000, 800, 1_600), (16_000, 8_000, 800, 400)],
)
def test_resample_waveform_changes_rate_dependent_length(
    source_rate: int,
    target_rate: int,
    source_frames: int,
    expected_frames: int,
) -> None:
    """Upsampling and downsampling should produce the expected frame count."""
    waveform = torch.linspace(-1, 1, source_frames).repeat(2, 1)

    resampled = resample_waveform(waveform, source_rate, target_rate)

    assert resampled.shape == (2, expected_frames)
    assert resampled.dtype == waveform.dtype
    assert torch.isfinite(resampled).all()


def test_resample_waveform_returns_matching_rate_input_unchanged() -> None:
    """No resampling should occur when source and target rates already match."""
    waveform = torch.randn(1, 100)

    assert resample_waveform(waveform, 8_000, 8_000) is waveform


@pytest.mark.parametrize("invalid_rate", [0, -1, 8_000.5, True])
def test_resample_waveform_rejects_invalid_rates(invalid_rate) -> None:
    """Sample rates must be positive integers."""
    with pytest.raises(AudioPreprocessingError, match="positive integer"):
        resample_waveform(torch.ones(1, 10), invalid_rate, 8_000)


def test_normalize_waveform_length_right_pads_with_zeros() -> None:
    """Short clips should keep their samples and receive trailing zero padding."""
    waveform = torch.tensor([[1.0, 2.0, 3.0]])

    normalized = normalize_waveform_length(waveform, 5)

    torch.testing.assert_close(normalized, torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0]]))


def test_normalize_waveform_length_center_crops_long_audio() -> None:
    """Evaluation cropping should deterministically retain the centered frames."""
    waveform = torch.arange(10, dtype=torch.float32).unsqueeze(0)

    normalized = normalize_waveform_length(waveform, 4, crop_mode="center")

    torch.testing.assert_close(normalized, torch.tensor([[3.0, 4.0, 5.0, 6.0]]))


def test_random_crop_is_reproducible_with_seeded_generators() -> None:
    """Identically seeded generators should choose the same training crop."""
    waveform = torch.arange(100, dtype=torch.float32).unsqueeze(0)
    first_generator = torch.Generator().manual_seed(91)
    second_generator = torch.Generator().manual_seed(91)

    first = normalize_waveform_length(
        waveform, 20, crop_mode="random", generator=first_generator
    )
    second = normalize_waveform_length(
        waveform, 20, crop_mode="random", generator=second_generator
    )

    torch.testing.assert_close(first, second)


def test_normalize_duration_produces_exact_frame_count() -> None:
    """Duration normalization should derive an exact target from seconds and rate."""
    waveform = torch.ones(1, 100)

    normalized = normalize_waveform_duration(waveform, 8_000, 0.25)

    assert normalized.shape == (1, 2_000)


def test_standardize_features_handles_variable_and_constant_inputs() -> None:
    """Features should standardize normally while constant input remains stable."""
    features = torch.arange(24, dtype=torch.float32).reshape(1, 4, 6)
    standardized = standardize_features(features)
    silence = standardize_features(torch.full((1, 4, 6), -80.0))

    assert float(standardized.mean()) == pytest.approx(0.0, abs=1.0e-6)
    assert float(standardized.std(unbiased=False)) == pytest.approx(1.0, abs=1.0e-6)
    torch.testing.assert_close(silence, torch.zeros_like(silence))


def test_log_mel_extractor_produces_fixed_finite_features() -> None:
    """The feature transform should produce a stable [channel, mel, time] tensor."""
    sample_rate = 8_000
    time = torch.arange(2_000) / sample_rate
    waveform = torch.sin(2 * torch.pi * 440 * time).unsqueeze(0)
    extractor = LogMelFeatureExtractor(
        sample_rate=sample_rate,
        n_fft=128,
        win_length=128,
        hop_length=64,
        n_mels=16,
        f_min=0.0,
        f_max=None,
        power=2.0,
        top_db=80.0,
    )

    features = extractor(waveform)

    assert features.shape == (1, 16, 32)
    assert torch.isfinite(features).all()
    assert float(features.mean()) == pytest.approx(0.0, abs=1.0e-5)
    assert float(features.std(unbiased=False)) == pytest.approx(1.0, abs=1.0e-5)


def test_mfcc_extractor_produces_fixed_finite_coefficients() -> None:
    """MFCC extraction should share the configured Mel time resolution."""
    sample_rate = 8_000
    time = torch.arange(2_000) / sample_rate
    waveform = torch.sin(2 * torch.pi * 440 * time).unsqueeze(0)
    extractor = MFCCFeatureExtractor(
        sample_rate=sample_rate,
        n_mfcc=8,
        n_fft=128,
        win_length=128,
        hop_length=64,
        n_mels=16,
        f_min=0.0,
        f_max=None,
        power=2.0,
    )

    coefficients = extractor(waveform)

    assert coefficients.shape == (1, 8, 32)
    assert torch.isfinite(coefficients).all()


def test_audio_preprocessor_composes_all_waveform_and_feature_steps() -> None:
    """Stereo high-rate input should become fixed mono model features."""
    settings = _audio_settings()
    source = torch.randn(2, 4_000)
    preprocessor = AudioPreprocessor(settings)

    processed = preprocessor(source, 16_000, training=False)

    assert processed.waveform.shape == (1, 2_000)
    assert processed.features.shape == (1, 16, 32)
    assert processed.source_sample_rate == 16_000
    assert processed.target_sample_rate == 8_000
    assert processed.source_num_channels == 2
    assert processed.source_num_frames == 4_000


def test_audio_preprocessor_rejects_unsupported_representation() -> None:
    """Configuration must not silently select an unimplemented representation."""
    settings = deepcopy(_audio_settings())
    settings["representation"] = "mfcc"

    with pytest.raises(AudioPreprocessingError, match="Unsupported"):
        AudioPreprocessor(settings)


def test_preprocessed_dataset_batches_fixed_features(tmp_path: Path) -> None:
    """Different source rates and lengths should collate into one model-ready batch."""
    first_path = tmp_path / "first.wav"
    second_path = tmp_path / "second.wav"
    sf.write(first_path, np.zeros((1_000, 2), dtype=np.float32), 8_000)
    sf.write(second_path, np.zeros(8_000, dtype=np.float32), 16_000)
    records = [
        AudioSampleRecord("first.wav", first_path, 0, "zero", 1, "fixture", {}),
        AudioSampleRecord("second.wav", second_path, 1, "one", 2, "fixture", {}),
    ]
    dataset = PreprocessedAudioDataset(
        records,
        AudioPreprocessor(_audio_settings()),
        training=False,
    )

    batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))

    assert batch["features"].shape == (2, 1, 16, 32)
    assert batch["label"].tolist() == [0, 1]
    assert batch["sample_id"] == ["first.wav", "second.wav"]
    assert batch["source_sample_rate"].tolist() == [8_000, 16_000]


def test_create_preprocessed_dataset_uses_requested_split() -> None:
    """The factory should use adapter splits and infer training behavior."""
    record = AudioSampleRecord(
        "sample.wav", Path("sample.wav"), 0, "zero", 1, "fixture", {}
    )

    class StubAdapter:
        dataset_name = "fixture"
        class_names = ("zero",)

        def load_records(self):
            return [record, record]

        def records_for_split(self, split_name: str):
            assert split_name == "validation"
            return [record]

    dataset = create_preprocessed_dataset(
        StubAdapter(), _audio_settings(), "validation"
    )

    assert len(dataset) == 1
    assert dataset.training is False
