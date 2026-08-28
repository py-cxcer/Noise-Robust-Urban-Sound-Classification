"""Tests for validated waveform loading before preprocessing."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from urban_sound_robustness.audio.loading import AudioLoadingError, load_audio


def test_load_audio_preserves_stereo_channels_rate_and_frames(tmp_path: Path) -> None:
    """Loading must preserve source structure and return channels-first float32."""
    sample_rate = 16_000
    frames = 800
    left = np.linspace(-0.5, 0.5, frames, dtype=np.float32)
    right = np.linspace(0.25, -0.25, frames, dtype=np.float32)
    frames_first = np.column_stack((left, right))
    audio_path = tmp_path / "stereo.wav"
    sf.write(audio_path, frames_first, sample_rate, subtype="FLOAT")

    loaded = load_audio(audio_path)

    assert loaded.waveform.shape == (2, frames)
    assert loaded.waveform.dtype == torch.float32
    assert loaded.sample_rate == sample_rate
    assert loaded.num_channels == 2
    assert loaded.num_frames == frames
    assert loaded.duration_seconds == pytest.approx(0.05)
    assert loaded.source_path == audio_path.resolve()
    torch.testing.assert_close(
        loaded.waveform,
        torch.from_numpy(frames_first.T.copy()),
    )


def test_load_audio_keeps_mono_dimension(tmp_path: Path) -> None:
    """A mono file must remain shaped [1, frames], never collapse to one axis."""
    audio_path = tmp_path / "mono.wav"
    sf.write(audio_path, np.zeros(32, dtype=np.float32), 8_000)

    loaded = load_audio(audio_path)

    assert loaded.waveform.shape == (1, 32)
    assert loaded.sample_rate == 8_000


@pytest.mark.parametrize("filename", ["missing.wav", "directory"])
def test_load_audio_rejects_non_file_paths(tmp_path: Path, filename: str) -> None:
    """Missing paths and directories should fail with actionable messages."""
    path = tmp_path / filename
    if filename == "directory":
        path.mkdir()

    with pytest.raises(AudioLoadingError, match="does not exist|not a file"):
        load_audio(path)


def test_load_audio_wraps_decoder_errors(tmp_path: Path) -> None:
    """Corrupt input should expose the source path through a stable exception."""
    audio_path = tmp_path / "corrupt.wav"
    audio_path.write_text("not a waveform", encoding="utf-8")

    with pytest.raises(AudioLoadingError, match=r"Unable to decode.*corrupt\.wav"):
        load_audio(audio_path)


def test_load_audio_rejects_non_finite_samples(tmp_path: Path) -> None:
    """NaN or infinity must be rejected before reaching later transforms."""
    audio_path = tmp_path / "non_finite.wav"
    waveform = np.array([0.0, np.nan, np.inf], dtype=np.float32)
    sf.write(audio_path, waveform, 8_000, subtype="FLOAT")

    with pytest.raises(AudioLoadingError, match="non-finite"):
        load_audio(audio_path)
