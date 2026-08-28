"""Deterministic external-noise selection for robustness evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from urban_sound_robustness.audio.loading import load_audio
from urban_sound_robustness.audio.noise import (
    SNRMixResult,
    match_noise_length,
    mix_waveforms_at_snr,
)
from urban_sound_robustness.audio.preprocessing import (
    convert_to_mono,
    resample_waveform,
)


SUPPORTED_NOISE_EXTENSIONS = frozenset({".wav", ".flac", ".ogg"})


class NoiseDatasetError(ValueError):
    """Raised when an external-noise collection cannot support corruption."""


@dataclass(frozen=True)
class RobustnessCondition:
    """One configured clean or noisy evaluation condition."""

    name: str
    snr_db: float | None


@dataclass(frozen=True)
class ControlledCorruption:
    """One deterministic condition applied to a clean waveform."""

    condition: RobustnessCondition
    waveform: Tensor
    mix_result: SNRMixResult | None
    noise_path: Path | None
    selection_seed: int | None


def stable_seed(base_seed: int, *parts: object) -> int:
    """Derive a stable non-negative PyTorch seed from semantic identifiers."""
    if isinstance(base_seed, bool) or not isinstance(base_seed, int) or base_seed < 0:
        raise NoiseDatasetError("base_seed must be a non-negative integer.")
    digest = hashlib.sha256()
    digest.update(str(base_seed).encode("utf-8"))
    for part in parts:
        digest.update(b"\x00")
        digest.update(str(part).encode("utf-8"))
    return int.from_bytes(digest.digest()[:8], byteorder="big") % (2**63 - 1)


def discover_noise_files(
    noise_directory: str | Path,
    *,
    extensions: Sequence[str] = tuple(SUPPORTED_NOISE_EXTENSIONS),
) -> list[Path]:
    """Return sorted supported audio files beneath an external-noise directory."""
    resolved_directory = Path(noise_directory).expanduser().resolve()
    if not resolved_directory.is_dir():
        raise NoiseDatasetError(
            f"External-noise directory does not exist: {resolved_directory}"
        )

    normalized_extensions = {extension.lower() for extension in extensions}
    noise_paths = sorted(
        path.resolve()
        for path in resolved_directory.rglob("*")
        if path.is_file() and path.suffix.lower() in normalized_extensions
    )
    if not noise_paths:
        supported = ", ".join(sorted(normalized_extensions))
        raise NoiseDatasetError(
            f"No supported noise files found under {resolved_directory}. "
            f"Expected extensions: {supported}."
        )
    return noise_paths


def parse_robustness_conditions(
    evaluation_settings: Mapping[str, Any],
) -> tuple[RobustnessCondition, ...]:
    """Convert validated condition mappings into immutable records."""
    conditions = evaluation_settings.get("conditions")
    if not isinstance(conditions, list) or not conditions:
        raise NoiseDatasetError("Evaluation conditions must be a non-empty list.")

    parsed: list[RobustnessCondition] = []
    for condition in conditions:
        if not isinstance(condition, dict):
            raise NoiseDatasetError("Each evaluation condition must be a mapping.")
        name = condition.get("name")
        snr_db = condition.get("snr_db")
        if not isinstance(name, str) or not name.strip():
            raise NoiseDatasetError("Every evaluation condition requires a name.")
        parsed.append(
            RobustnessCondition(
                name=name,
                snr_db=None if snr_db is None else float(snr_db),
            )
        )
    return tuple(parsed)


class DeterministicNoiseCorruptor:
    """Apply repeatable noise-file and segment choices per sample ID."""

    def __init__(
        self,
        noise_directory: str | Path,
        *,
        target_sample_rate: int,
        corruption_seed: int,
        power_epsilon: float = 1.0e-12,
    ) -> None:
        self.noise_paths = discover_noise_files(noise_directory)
        if (
            isinstance(target_sample_rate, bool)
            or not isinstance(target_sample_rate, int)
            or target_sample_rate <= 0
        ):
            raise NoiseDatasetError("target_sample_rate must be a positive integer.")
        self.target_sample_rate = target_sample_rate
        self.corruption_seed = corruption_seed
        self.power_epsilon = power_epsilon
        stable_seed(corruption_seed, "validation")

    def _prepare_noise(
        self,
        clean_waveform: Tensor,
        sample_id: str,
    ) -> tuple[Tensor, Path, int]:
        """Select, load, resample, and length-align one deterministic noise clip."""
        selection_seed = stable_seed(self.corruption_seed, sample_id)
        path_seed = stable_seed(selection_seed, "noise_file")
        noise_path = self.noise_paths[path_seed % len(self.noise_paths)]
        loaded_noise = load_audio(noise_path)
        noise_waveform = convert_to_mono(loaded_noise.waveform)
        noise_waveform = resample_waveform(
            noise_waveform,
            source_sample_rate=loaded_noise.sample_rate,
            target_sample_rate=self.target_sample_rate,
        )
        segment_generator = torch.Generator().manual_seed(
            stable_seed(selection_seed, "noise_segment")
        )
        noise_waveform = match_noise_length(
            noise_waveform,
            int(clean_waveform.shape[1]),
            generator=segment_generator,
        )
        if clean_waveform.shape[0] > 1:
            noise_waveform = noise_waveform.expand(clean_waveform.shape[0], -1)
        return noise_waveform, noise_path, selection_seed

    def corrupt(
        self,
        clean_waveform: Tensor,
        sample_id: str,
        condition: RobustnessCondition,
    ) -> ControlledCorruption:
        """Apply one clean/noisy condition without depending on iteration order."""
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise NoiseDatasetError("sample_id must be a non-empty string.")
        if condition.snr_db is None:
            return ControlledCorruption(
                condition=condition,
                waveform=clean_waveform.clone(),
                mix_result=None,
                noise_path=None,
                selection_seed=None,
            )

        noise_waveform, noise_path, selection_seed = self._prepare_noise(
            clean_waveform, sample_id
        )
        mix_result = mix_waveforms_at_snr(
            clean_waveform,
            noise_waveform,
            condition.snr_db,
            power_epsilon=self.power_epsilon,
        )
        return ControlledCorruption(
            condition=condition,
            waveform=mix_result.waveform,
            mix_result=mix_result,
            noise_path=noise_path,
            selection_seed=selection_seed,
        )
