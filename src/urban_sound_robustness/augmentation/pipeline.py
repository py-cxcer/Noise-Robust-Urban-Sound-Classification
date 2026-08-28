"""Configurable training-only waveform and spectrogram augmentation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn

from urban_sound_robustness.audio import (
    AudioPreprocessor,
    PreprocessedAudio,
    convert_to_mono,
    load_audio,
    match_noise_length,
    mix_waveforms_at_snr,
    resample_waveform,
)
from urban_sound_robustness.evaluation.corruption import discover_noise_files
from urban_sound_robustness.utils.paths import resolve_project_path


class AugmentationError(ValueError):
    """Raised when an enabled augmentation cannot be applied safely."""


def _draw_probability(probability: float, generator: torch.Generator | None) -> bool:
    """Return a generator-controlled Bernoulli decision."""
    return float(torch.rand((), generator=generator).item()) < probability


def _draw_uniform(
    minimum: float,
    maximum: float,
    generator: torch.Generator | None,
) -> float:
    """Draw a reproducible uniform scalar from an inclusive numeric interval."""
    if minimum == maximum:
        return minimum
    unit_value = float(torch.rand((), generator=generator).item())
    return minimum + (maximum - minimum) * unit_value


def _draw_integer(
    minimum: int,
    maximum: int,
    generator: torch.Generator | None,
) -> int:
    """Draw an integer from the inclusive interval [minimum, maximum]."""
    if minimum == maximum:
        return minimum
    return int(
        torch.randint(
            minimum,
            maximum + 1,
            (),
            generator=generator,
        ).item()
    )


class WaveformAugmenter:
    """Apply configured time shift, gain, and real background noise."""

    def __init__(
        self,
        settings: Mapping[str, Any],
        *,
        sample_rate: int,
        project_root: str | Path,
        noise_paths: Sequence[str | Path] | None = None,
    ) -> None:
        self.settings = dict(settings)
        self.sample_rate = sample_rate
        self.project_root = Path(project_root).expanduser().resolve()
        self.time_shift = dict(self.settings.get("time_shift", {}))
        self.random_gain = dict(self.settings.get("random_gain", {}))
        self.background_noise = dict(self.settings.get("background_noise", {}))

        for optional_name in ("pitch_shift", "time_stretch"):
            optional_settings = dict(self.settings.get(optional_name, {}))
            if optional_settings.get("enabled", False):
                raise AugmentationError(
                    f"{optional_name} is optional and is not currently implemented."
                )

        self.noise_paths: tuple[Path, ...] = ()
        if self.background_noise.get("enabled", False):
            if noise_paths is None:
                configured_directory = self.background_noise.get("noise_directory")
                if not isinstance(configured_directory, str):
                    raise AugmentationError(
                        "Enabled background noise requires noise_directory."
                    )
                noise_directory = resolve_project_path(
                    configured_directory, self.project_root
                )
                discovered = discover_noise_files(noise_directory)
            else:
                discovered = [
                    Path(path).expanduser().resolve() for path in noise_paths
                ]
                if not discovered:
                    raise AugmentationError(
                        "Enabled background noise requires at least one noise file."
                    )
            self.noise_paths = tuple(discovered)

    def __call__(
        self,
        waveform: Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Return an augmented waveform without modifying the source tensor."""
        augmented = waveform
        if self.time_shift.get("enabled", False) and _draw_probability(
            float(self.time_shift["probability"]), generator
        ):
            maximum_shift = round(
                waveform.shape[1] * float(self.time_shift["max_shift_fraction"])
            )
            shift = _draw_integer(-maximum_shift, maximum_shift, generator)
            augmented = torch.roll(augmented, shifts=shift, dims=1)

        if self.random_gain.get("enabled", False) and _draw_probability(
            float(self.random_gain["probability"]), generator
        ):
            gain_db = _draw_uniform(
                float(self.random_gain["min_gain_db"]),
                float(self.random_gain["max_gain_db"]),
                generator,
            )
            gain = 10.0 ** (gain_db / 20.0)
            augmented = augmented * gain

        if self.background_noise.get("enabled", False) and _draw_probability(
            float(self.background_noise["probability"]), generator
        ):
            augmented = self._add_background_noise(augmented, generator)
        return augmented

    def _add_background_noise(
        self,
        waveform: Tensor,
        generator: torch.Generator | None,
    ) -> Tensor:
        """Select and mix one real noise recording at a random configured SNR."""
        path_index = _draw_integer(0, len(self.noise_paths) - 1, generator)
        loaded = load_audio(self.noise_paths[path_index])
        noise = convert_to_mono(loaded.waveform)
        noise = resample_waveform(noise, loaded.sample_rate, self.sample_rate)
        noise = match_noise_length(
            noise,
            int(waveform.shape[1]),
            generator=generator,
        )
        if waveform.shape[0] > 1:
            noise = noise.expand(waveform.shape[0], -1)
        target_snr_db = _draw_uniform(
            float(self.background_noise["min_snr_db"]),
            float(self.background_noise["max_snr_db"]),
            generator,
        )
        return mix_waveforms_at_snr(
            waveform,
            noise,
            target_snr_db,
        ).waveform


class SpectrogramAugmenter:
    """Apply generator-controlled frequency and time masking."""

    def __init__(self, settings: Mapping[str, Any]) -> None:
        self.frequency_mask = dict(settings.get("frequency_mask", {}))
        self.time_mask = dict(settings.get("time_mask", {}))

    def __call__(
        self,
        features: Tensor,
        *,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Mask standardized features with zeros, their approximate mean value."""
        augmented = features.clone()
        if self.frequency_mask.get("enabled", False) and _draw_probability(
            float(self.frequency_mask["probability"]), generator
        ):
            self._mask_dimension(
                augmented,
                dimension=1,
                maximum_width=int(self.frequency_mask["max_mask_bins"]),
                generator=generator,
            )
        if self.time_mask.get("enabled", False) and _draw_probability(
            float(self.time_mask["probability"]), generator
        ):
            self._mask_dimension(
                augmented,
                dimension=2,
                maximum_width=int(self.time_mask["max_mask_frames"]),
                generator=generator,
            )
        return augmented

    @staticmethod
    def _mask_dimension(
        features: Tensor,
        *,
        dimension: int,
        maximum_width: int,
        generator: torch.Generator | None,
    ) -> None:
        dimension_size = int(features.shape[dimension])
        width = _draw_integer(0, min(maximum_width, dimension_size), generator)
        if width == 0:
            return
        start = _draw_integer(0, dimension_size - width, generator)
        slices = [slice(None)] * features.ndim
        slices[dimension] = slice(start, start + width)
        features[tuple(slices)] = 0.0


class AugmentedAudioPreprocessor(nn.Module):
    """Insert configured training augmentations around reusable preprocessing."""

    def __init__(
        self,
        base_preprocessor: AudioPreprocessor,
        augmentation_settings: Mapping[str, Any],
        *,
        project_root: str | Path,
        noise_paths: Sequence[str | Path] | None = None,
    ) -> None:
        super().__init__()
        self.base_preprocessor = base_preprocessor
        self.sample_rate = base_preprocessor.sample_rate
        self.clip_duration_seconds = base_preprocessor.clip_duration_seconds
        self.waveform_augmenter = WaveformAugmenter(
            augmentation_settings.get("waveform", {}),
            sample_rate=self.sample_rate,
            project_root=project_root,
            noise_paths=noise_paths,
        )
        self.spectrogram_augmenter = SpectrogramAugmenter(
            augmentation_settings.get("spectrogram", {})
        )

    @property
    def target_num_frames(self) -> int:
        """Return the exact waveform size produced by the base preprocessor."""
        return self.base_preprocessor.target_num_frames

    def forward(
        self,
        waveform: Tensor,
        source_sample_rate: int,
        *,
        training: bool | None = None,
        generator: torch.Generator | None = None,
    ) -> PreprocessedAudio:
        """Apply augmentation only when the caller explicitly requests training."""
        source_num_channels = int(waveform.shape[0])
        source_num_frames = int(waveform.shape[1])
        use_training = self.training if training is None else training
        processed = self.base_preprocessor.prepare_waveform(
            waveform,
            source_sample_rate,
            training=use_training,
            generator=generator,
        )
        if use_training:
            processed = self.waveform_augmenter(
                processed,
                generator=generator,
            )
        features = self.base_preprocessor.extract_features(processed)
        if use_training:
            features = self.spectrogram_augmenter(
                features,
                generator=generator,
            )
        return PreprocessedAudio(
            waveform=processed,
            features=features,
            source_sample_rate=source_sample_rate,
            target_sample_rate=self.sample_rate,
            source_num_channels=source_num_channels,
            source_num_frames=source_num_frames,
        )


def create_configured_preprocessor(
    audio_settings: Mapping[str, Any],
    augmentation_settings: Mapping[str, Any] | None,
    *,
    project_root: str | Path,
    noise_paths: Sequence[str | Path] | None = None,
) -> AudioPreprocessor | AugmentedAudioPreprocessor:
    """Create the baseline or augmented preprocessing pipeline from configuration."""
    base = AudioPreprocessor(audio_settings)
    if not augmentation_settings or not augmentation_settings.get("enabled", False):
        return base
    return AugmentedAudioPreprocessor(
        base,
        augmentation_settings,
        project_root=project_root,
        noise_paths=noise_paths,
    )
