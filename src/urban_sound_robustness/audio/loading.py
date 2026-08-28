"""Validated audio loading without preprocessing or feature conversion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import numpy as np
import soundfile as sf
import torch
from torch import Tensor


PathLike: TypeAlias = str | Path


class AudioLoadingError(RuntimeError):
    """Raised when an audio file cannot produce a valid waveform tensor."""


@dataclass(frozen=True)
class LoadedAudio:
    """An unprocessed waveform and the properties reported by its source file.

    The waveform always has shape ``[channels, frames]`` and dtype
    ``torch.float32``. No channel mixing, resampling, padding, or cropping is
    performed while loading.
    """

    waveform: Tensor
    sample_rate: int
    source_path: Path

    @property
    def num_channels(self) -> int:
        """Return the number of source channels."""
        return int(self.waveform.shape[0])

    @property
    def num_frames(self) -> int:
        """Return the number of frames per channel."""
        return int(self.waveform.shape[1])

    @property
    def duration_seconds(self) -> float:
        """Return duration derived from frames and the original sample rate."""
        return self.num_frames / self.sample_rate


def load_audio(path: PathLike) -> LoadedAudio:
    """Load one audio file as a validated channels-first PyTorch tensor.

    No channel mixing, resampling, padding, or cropping is performed.
    """
    source_path = Path(path).expanduser().resolve()
    if not source_path.exists():
        raise AudioLoadingError(f"Audio file does not exist: {source_path}")
    if not source_path.is_file():
        raise AudioLoadingError(f"Audio path is not a file: {source_path}")

    try:
        frames_first, sample_rate = sf.read(
            source_path,
            dtype="float32",
            always_2d=True,
        )
    except (OSError, RuntimeError, sf.LibsndfileError) as error:
        raise AudioLoadingError(
            f"Unable to decode audio file {source_path}: {error}"
        ) from error

    if sample_rate <= 0:
        raise AudioLoadingError(
            f"Audio file reports an invalid sample rate {sample_rate}: {source_path}"
        )
    if frames_first.ndim != 2 or frames_first.shape[1] < 1:
        raise AudioLoadingError(f"Audio file has no channels: {source_path}")
    if frames_first.shape[0] < 1:
        raise AudioLoadingError(f"Audio file contains no frames: {source_path}")
    if not np.isfinite(frames_first).all():
        raise AudioLoadingError(f"Audio file contains non-finite samples: {source_path}")

    waveform = torch.from_numpy(np.ascontiguousarray(frames_first.T))
    return LoadedAudio(
        waveform=waveform,
        sample_rate=int(sample_rate),
        source_path=source_path,
    )
