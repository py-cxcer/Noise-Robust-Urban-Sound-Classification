"""Tests for target-SNR mixing and deterministic external-noise corruption."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from urban_sound_robustness.audio.noise import (
    NoiseMixingError,
    match_noise_length,
    measure_snr_db,
    mix_waveforms_at_snr,
)
from urban_sound_robustness.evaluation.corruption import (
    DeterministicNoiseCorruptor,
    NoiseDatasetError,
    RobustnessCondition,
    discover_noise_files,
    parse_robustness_conditions,
    stable_seed,
)


@pytest.mark.parametrize("target_snr_db", [20.0, 10.0, 0.0, -5.0])
def test_mixing_achieves_requested_snr(target_snr_db: float) -> None:
    """Measured mean-square SNR should match each requested condition."""
    generator = torch.Generator().manual_seed(42)
    clean = torch.randn((1, 16_000), generator=generator) * 0.2
    noise = torch.randn((1, 16_000), generator=generator)

    result = mix_waveforms_at_snr(clean, noise, target_snr_db)

    assert result.applied is True
    assert result.achieved_snr_db == pytest.approx(target_snr_db, abs=1.0e-5)
    assert measure_snr_db(clean, result.waveform) == pytest.approx(
        target_snr_db, abs=1.0e-5
    )


def test_lower_snr_produces_more_noise_power() -> None:
    """A ten-decibel reduction should multiply noise power by ten."""
    clean = torch.linspace(-0.5, 0.5, 2_000).unsqueeze(0)
    noise = torch.ones_like(clean)

    twenty_db = mix_waveforms_at_snr(clean, noise, 20.0)
    ten_db = mix_waveforms_at_snr(clean, noise, 10.0)

    assert ten_db.scaled_noise_power / twenty_db.scaled_noise_power == pytest.approx(
        10.0, rel=1.0e-5
    )


def test_silent_signal_is_preserved_without_false_snr_claim() -> None:
    """Finite SNR is undefined for silence, so no noise should be injected."""
    clean = torch.zeros(1, 1_000)
    noise = torch.randn(1, 1_000)

    result = mix_waveforms_at_snr(clean, noise, 10.0)

    assert result.applied is False
    assert result.reason == "silent_signal"
    assert result.achieved_snr_db is None
    torch.testing.assert_close(result.waveform, clean)


def test_near_silent_noise_is_rejected() -> None:
    """A non-silent signal cannot be mixed using a zero-power noise source."""
    with pytest.raises(NoiseMixingError, match="too close to zero"):
        mix_waveforms_at_snr(torch.ones(1, 100), torch.zeros(1, 100), 10.0)


def test_mixing_requires_matching_shapes() -> None:
    """Length/channel alignment must happen explicitly before SNR scaling."""
    with pytest.raises(NoiseMixingError, match="identical shapes"):
        mix_waveforms_at_snr(torch.ones(1, 100), torch.ones(1, 90), 10.0)


def test_short_noise_is_repeated_to_exact_length() -> None:
    """Short noise should wrap without padding silent gaps."""
    noise = torch.tensor([[1.0, 2.0, 3.0]])
    generator = torch.Generator().manual_seed(1)

    matched = match_noise_length(noise, 8, generator=generator)

    assert matched.shape == (1, 8)
    assert set(matched.flatten().tolist()) == {1.0, 2.0, 3.0}


def test_long_noise_crop_is_reproducible() -> None:
    """Equal generator seeds should select the same long-noise segment."""
    noise = torch.arange(100, dtype=torch.float32).unsqueeze(0)
    first = match_noise_length(
        noise, 20, generator=torch.Generator().manual_seed(88)
    )
    second = match_noise_length(
        noise, 20, generator=torch.Generator().manual_seed(88)
    )

    torch.testing.assert_close(first, second)


def test_stable_seed_depends_on_semantic_identifiers() -> None:
    """Stable seeds should repeat exactly and vary across sample IDs."""
    assert stable_seed(2025, "sample-a") == stable_seed(2025, "sample-a")
    assert stable_seed(2025, "sample-a") != stable_seed(2025, "sample-b")


def test_noise_discovery_is_recursive_and_filters_extensions(
    tmp_path: Path,
) -> None:
    """Only supported audio files should enter the external-noise bank."""
    nested = tmp_path / "nested"
    nested.mkdir()
    sf.write(tmp_path / "first.wav", np.ones(100, dtype=np.float32), 8_000)
    sf.write(nested / "second.flac", np.ones(100, dtype=np.float32), 8_000)
    (tmp_path / "notes.txt").write_text("not audio", encoding="utf-8")

    discovered = discover_noise_files(tmp_path)

    assert [path.name for path in discovered] == ["first.wav", "second.flac"]


def test_noise_discovery_rejects_empty_directory(tmp_path: Path) -> None:
    """Research evaluation must fail clearly when no real noise is available."""
    with pytest.raises(NoiseDatasetError, match="No supported noise"):
        discover_noise_files(tmp_path)


def test_deterministic_corruptor_repeats_selection_and_segment(
    tmp_path: Path,
) -> None:
    """The same sample should receive identical corruption on repeated calls."""
    first_noise = np.linspace(-1, 1, 5_000, dtype=np.float32)
    second_noise = np.sin(np.linspace(0, 20, 5_000)).astype(np.float32)
    sf.write(tmp_path / "first.wav", first_noise, 8_000)
    sf.write(tmp_path / "second.wav", second_noise, 8_000)
    corruptor = DeterministicNoiseCorruptor(
        tmp_path,
        target_sample_rate=8_000,
        corruption_seed=2025,
    )
    clean = torch.linspace(-0.5, 0.5, 2_000).unsqueeze(0)
    condition = RobustnessCondition("snr_10db", 10.0)

    first = corruptor.corrupt(clean, "sample.wav", condition)
    second = corruptor.corrupt(clean, "sample.wav", condition)

    assert first.noise_path == second.noise_path
    assert first.selection_seed == second.selection_seed
    torch.testing.assert_close(first.waveform, second.waveform)


def test_corruptor_uses_same_noise_segment_across_snr_conditions(
    tmp_path: Path,
) -> None:
    """SNR conditions should differ only in scaling for a given clean sample."""
    sf.write(
        tmp_path / "noise.wav",
        np.sin(np.linspace(0, 40, 5_000)).astype(np.float32),
        8_000,
    )
    corruptor = DeterministicNoiseCorruptor(
        tmp_path,
        target_sample_rate=8_000,
        corruption_seed=2025,
    )
    clean = torch.linspace(-0.5, 0.5, 2_000).unsqueeze(0)

    twenty = corruptor.corrupt(
        clean, "sample.wav", RobustnessCondition("snr_20db", 20.0)
    )
    zero = corruptor.corrupt(
        clean, "sample.wav", RobustnessCondition("snr_0db", 0.0)
    )

    assert twenty.mix_result is not None
    assert zero.mix_result is not None
    unscaled_twenty = twenty.mix_result.scaled_noise / twenty.mix_result.scale_factor
    unscaled_zero = zero.mix_result.scaled_noise / zero.mix_result.scale_factor
    torch.testing.assert_close(unscaled_twenty, unscaled_zero)


def test_clean_condition_does_not_require_or_apply_noise(tmp_path: Path) -> None:
    """The clean condition should preserve the waveform exactly."""
    sf.write(tmp_path / "noise.wav", np.ones(100, dtype=np.float32), 8_000)
    corruptor = DeterministicNoiseCorruptor(
        tmp_path,
        target_sample_rate=8_000,
        corruption_seed=2025,
    )
    clean = torch.randn(1, 100)

    result = corruptor.corrupt(
        clean, "sample.wav", RobustnessCondition("clean", None)
    )

    assert result.mix_result is None
    assert result.noise_path is None
    torch.testing.assert_close(result.waveform, clean)


def test_parse_robustness_conditions_preserves_order() -> None:
    """Configuration order should determine reporting order."""
    settings = {
        "conditions": [
            {"name": "clean", "snr_db": None},
            {"name": "snr_10db", "snr_db": 10.0},
        ]
    }

    conditions = parse_robustness_conditions(settings)

    assert conditions == (
        RobustnessCondition("clean", None),
        RobustnessCondition("snr_10db", 10.0),
    )
