"""Deterministic waveform preprocessing operations."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as torch_functional
from torchaudio import functional as audio_functional
from torchaudio import transforms as audio_transforms


class AudioPreprocessingError(ValueError):
    """Raised when a waveform is invalid for a preprocessing operation."""


def _validate_waveform(waveform: Tensor) -> None:
    """Validate the shared channels-first waveform contract."""
    if not isinstance(waveform, Tensor):
        raise AudioPreprocessingError("Waveform must be a torch.Tensor.")
    if waveform.ndim != 2:
        raise AudioPreprocessingError(
            "Waveform must have shape [channels, frames]; "
            f"received {tuple(waveform.shape)}."
        )
    if waveform.shape[0] < 1:
        raise AudioPreprocessingError("Waveform must contain at least one channel.")
    if waveform.shape[1] < 1:
        raise AudioPreprocessingError("Waveform must contain at least one frame.")
    if not waveform.is_floating_point():
        raise AudioPreprocessingError(
            f"Waveform must use a floating-point dtype; received {waveform.dtype}."
        )
    if not bool(torch.isfinite(waveform).all()):
        raise AudioPreprocessingError("Waveform contains non-finite samples.")


def convert_to_mono(waveform: Tensor) -> Tensor:
    """Convert a channels-first waveform to one channel by arithmetic mean.

    Mono input is returned unchanged. Multi-channel input is averaged across its
    channel dimension while retaining shape ``[1, frames]``. The operation keeps
    the input dtype and device, does not mutate the source tensor, and remains
    differentiable for future trainable pipelines.
    """
    _validate_waveform(waveform)
    if waveform.shape[0] == 1:
        return waveform
    return waveform.mean(dim=0, keepdim=True)


def _validate_sample_rate(sample_rate: int, name: str) -> None:
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int):
        raise AudioPreprocessingError(f"{name} must be a positive integer.")
    if sample_rate <= 0:
        raise AudioPreprocessingError(f"{name} must be a positive integer.")


def resample_waveform(
    waveform: Tensor,
    source_sample_rate: int,
    target_sample_rate: int,
) -> Tensor:
    """Resample a channels-first waveform while preserving dtype and device."""
    _validate_waveform(waveform)
    _validate_sample_rate(source_sample_rate, "source_sample_rate")
    _validate_sample_rate(target_sample_rate, "target_sample_rate")

    if source_sample_rate == target_sample_rate:
        return waveform

    return audio_functional.resample(
        waveform,
        orig_freq=source_sample_rate,
        new_freq=target_sample_rate,
    )


def normalize_waveform_length(
    waveform: Tensor,
    target_num_frames: int,
    *,
    crop_mode: str = "center",
    padding_mode: str = "zero",
    generator: torch.Generator | None = None,
) -> Tensor:
    """Pad or crop a waveform to an exact number of frames.

    Zero padding is appended to the end of short clips. Long clips are center
    cropped for evaluation or randomly cropped for training. Supplying a seeded
    generator makes random crop selection reproducible.
    """
    _validate_waveform(waveform)
    if (
        isinstance(target_num_frames, bool)
        or not isinstance(target_num_frames, int)
        or target_num_frames <= 0
    ):
        raise AudioPreprocessingError("target_num_frames must be a positive integer.")
    if crop_mode not in {"center", "random"}:
        raise AudioPreprocessingError("crop_mode must be 'center' or 'random'.")
    if padding_mode != "zero":
        raise AudioPreprocessingError("Only zero padding is currently supported.")

    current_num_frames = int(waveform.shape[1])
    if current_num_frames == target_num_frames:
        return waveform
    if current_num_frames < target_num_frames:
        padding = target_num_frames - current_num_frames
        return torch_functional.pad(waveform, (0, padding), mode="constant", value=0.0)

    maximum_offset = current_num_frames - target_num_frames
    if crop_mode == "center":
        start = maximum_offset // 2
    else:
        start = int(
            torch.randint(
                low=0,
                high=maximum_offset + 1,
                size=(),
                generator=generator,
            ).item()
        )
    return waveform[:, start : start + target_num_frames]


def normalize_waveform_duration(
    waveform: Tensor,
    sample_rate: int,
    duration_seconds: float,
    *,
    crop_mode: str = "center",
    padding_mode: str = "zero",
    generator: torch.Generator | None = None,
) -> Tensor:
    """Pad or crop a waveform to an exact duration at its current sample rate."""
    _validate_sample_rate(sample_rate, "sample_rate")
    if (
        isinstance(duration_seconds, bool)
        or not isinstance(duration_seconds, (int, float))
        or not math.isfinite(float(duration_seconds))
        or duration_seconds <= 0
    ):
        raise AudioPreprocessingError("duration_seconds must be a positive number.")

    target_num_frames = round(sample_rate * float(duration_seconds))
    if target_num_frames < 1:
        raise AudioPreprocessingError(
            "duration_seconds and sample_rate produce fewer than one target frame."
        )
    return normalize_waveform_length(
        waveform,
        target_num_frames,
        crop_mode=crop_mode,
        padding_mode=padding_mode,
        generator=generator,
    )


def standardize_features(features: Tensor, epsilon: float = 1.0e-6) -> Tensor:
    """Apply zero-mean, unit-variance standardization to one feature tensor."""
    if not isinstance(features, Tensor) or features.ndim < 2:
        raise AudioPreprocessingError(
            "Features must be a torch.Tensor with at least two dimensions."
        )
    if not features.is_floating_point():
        raise AudioPreprocessingError("Features must use a floating-point dtype.")
    if not bool(torch.isfinite(features).all()):
        raise AudioPreprocessingError("Features contain non-finite values.")
    if (
        isinstance(epsilon, bool)
        or not isinstance(epsilon, (int, float))
        or not math.isfinite(float(epsilon))
        or epsilon <= 0
    ):
        raise AudioPreprocessingError("epsilon must be a positive number.")

    mean = features.mean()
    standard_deviation = features.std(unbiased=False)
    return (features - mean) / standard_deviation.clamp_min(float(epsilon))


class LogMelFeatureExtractor(nn.Module):
    """Convert fixed-rate waveforms into standardized log-Mel spectrograms."""

    def __init__(
        self,
        *,
        sample_rate: int,
        n_fft: int,
        win_length: int,
        hop_length: int,
        n_mels: int,
        f_min: float,
        f_max: float | None,
        power: float,
        top_db: float | None,
        center: bool = True,
        pad_mode: str = "reflect",
        mel_scale: str = "htk",
        standardization_epsilon: float = 1.0e-6,
    ) -> None:
        super().__init__()
        _validate_sample_rate(sample_rate, "sample_rate")
        self.sample_rate = sample_rate
        self.standardization_epsilon = standardization_epsilon
        self.mel_spectrogram = audio_transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            power=power,
            center=center,
            pad_mode=pad_mode,
            normalized=False,
            mel_scale=mel_scale,
        )
        self.to_decibels = audio_transforms.AmplitudeToDB(
            stype="power",
            top_db=top_db,
        )

    def forward(self, waveform: Tensor) -> Tensor:
        """Return finite standardized features shaped [channels, mels, time]."""
        _validate_waveform(waveform)
        mel_spectrogram = self.mel_spectrogram(waveform)
        log_mel = self.to_decibels(mel_spectrogram)
        standardized = standardize_features(
            log_mel,
            epsilon=self.standardization_epsilon,
        )
        if not bool(torch.isfinite(standardized).all()):
            raise AudioPreprocessingError(
                "Log-Mel extraction produced non-finite features."
            )
        return standardized


class MFCCFeatureExtractor(nn.Module):
    """Convert fixed-rate waveforms into MFCCs for EDA and classical models."""

    def __init__(
        self,
        *,
        sample_rate: int,
        n_mfcc: int,
        n_fft: int,
        win_length: int,
        hop_length: int,
        n_mels: int,
        f_min: float,
        f_max: float | None,
        power: float,
        center: bool = True,
        pad_mode: str = "reflect",
        mel_scale: str = "htk",
    ) -> None:
        super().__init__()
        _validate_sample_rate(sample_rate, "sample_rate")
        if isinstance(n_mfcc, bool) or not isinstance(n_mfcc, int) or n_mfcc <= 0:
            raise AudioPreprocessingError("n_mfcc must be a positive integer.")
        if n_mfcc > n_mels:
            raise AudioPreprocessingError("n_mfcc cannot exceed n_mels.")

        # torchaudio performs the Mel projection, logarithm, and discrete cosine
        # transform. Keeping this transform here makes notebooks and future
        # classical baselines share the same signal-processing definition.
        self.mfcc = audio_transforms.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            dct_type=2,
            norm="ortho",
            log_mels=True,
            melkwargs={
                "n_fft": n_fft,
                "win_length": win_length,
                "hop_length": hop_length,
                "f_min": f_min,
                "f_max": f_max,
                "n_mels": n_mels,
                "power": power,
                "center": center,
                "pad_mode": pad_mode,
                "normalized": False,
                "mel_scale": mel_scale,
            },
        )

    def forward(self, waveform: Tensor) -> Tensor:
        """Return finite coefficients shaped [channels, coefficients, time]."""
        _validate_waveform(waveform)
        coefficients = self.mfcc(waveform)
        if not bool(torch.isfinite(coefficients).all()):
            raise AudioPreprocessingError("MFCC extraction produced non-finite values.")
        return coefficients


@dataclass(frozen=True)
class PreprocessedAudio:
    """Waveform and model features produced from one source audio file."""

    waveform: Tensor
    features: Tensor
    source_sample_rate: int
    target_sample_rate: int
    source_num_channels: int
    source_num_frames: int


class AudioPreprocessor(nn.Module):
    """Compose channel, rate, length, and log-Mel preprocessing from config."""

    def __init__(self, settings: Mapping[str, Any]) -> None:
        super().__init__()
        self.settings = dict(settings)
        self.sample_rate = int(self.settings["sample_rate"])
        self.clip_duration_seconds = float(
            self.settings["clip_duration_seconds"]
        )
        self.mono = bool(self.settings["mono"])

        length_settings = dict(self.settings["length_normalization"])
        self.padding_mode = str(length_settings["padding_mode"])
        self.training_crop = str(length_settings["training_crop"])
        self.evaluation_crop = str(length_settings["evaluation_crop"])

        representation = str(self.settings["representation"])
        if representation != "log_mel":
            raise AudioPreprocessingError(
                f"Unsupported audio representation: {representation!r}."
            )

        log_mel = dict(self.settings["log_mel"])
        normalization = dict(self.settings["normalization"])
        if normalization.get("method") != "per_example_standardization":
            raise AudioPreprocessingError(
                "Only per_example_standardization is currently supported."
            )

        self.feature_extractor = LogMelFeatureExtractor(
            sample_rate=self.sample_rate,
            n_fft=int(log_mel["n_fft"]),
            win_length=int(log_mel["win_length"]),
            hop_length=int(log_mel["hop_length"]),
            n_mels=int(log_mel["n_mels"]),
            f_min=float(log_mel["f_min"]),
            f_max=(
                None if log_mel.get("f_max") is None else float(log_mel["f_max"])
            ),
            power=float(log_mel["power"]),
            top_db=(
                None if log_mel.get("top_db") is None else float(log_mel["top_db"])
            ),
            center=bool(log_mel.get("center", True)),
            pad_mode=str(log_mel.get("pad_mode", "reflect")),
            mel_scale=str(log_mel.get("mel_scale", "htk")),
            standardization_epsilon=float(normalization["epsilon"]),
        )

    @property
    def target_num_frames(self) -> int:
        """Return the exact waveform length produced by this preprocessor."""
        return round(self.sample_rate * self.clip_duration_seconds)

    def prepare_waveform(
        self,
        waveform: Tensor,
        source_sample_rate: int,
        *,
        training: bool | None = None,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Apply channel, sample-rate, and duration normalization only."""
        _validate_waveform(waveform)
        processed = convert_to_mono(waveform) if self.mono else waveform
        processed = resample_waveform(
            processed,
            source_sample_rate=source_sample_rate,
            target_sample_rate=self.sample_rate,
        )
        use_training_crop = self.training if training is None else training
        crop_mode = self.training_crop if use_training_crop else self.evaluation_crop
        return normalize_waveform_duration(
            processed,
            sample_rate=self.sample_rate,
            duration_seconds=self.clip_duration_seconds,
            crop_mode=crop_mode,
            padding_mode=self.padding_mode,
            generator=generator,
        )

    def extract_features(self, waveform: Tensor) -> Tensor:
        """Transform one normalized waveform into model-ready features."""
        return self.feature_extractor(waveform)

    def forward(
        self,
        waveform: Tensor,
        source_sample_rate: int,
        *,
        training: bool | None = None,
        generator: torch.Generator | None = None,
    ) -> PreprocessedAudio:
        """Apply the configured deterministic/evaluation or training pipeline."""
        _validate_waveform(waveform)
        source_num_channels = int(waveform.shape[0])
        source_num_frames = int(waveform.shape[1])
        processed = self.prepare_waveform(
            waveform,
            source_sample_rate,
            training=training,
            generator=generator,
        )
        features = self.extract_features(processed)
        return PreprocessedAudio(
            waveform=processed,
            features=features,
            source_sample_rate=source_sample_rate,
            target_sample_rate=self.sample_rate,
            source_num_channels=source_num_channels,
            source_num_frames=source_num_frames,
        )
