"""Power-based background-noise mixing for controlled SNR experiments."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor


class NoiseMixingError(ValueError):
    """Raised when audio cannot be mixed at a mathematically valid SNR."""


@dataclass(frozen=True)
class SNRMixResult:
    """Numerical details and tensors produced by one target-SNR mixture."""

    waveform: Tensor
    scaled_noise: Tensor
    target_snr_db: float
    achieved_snr_db: float | None
    signal_power: float
    source_noise_power: float
    scaled_noise_power: float
    scale_factor: float
    applied: bool
    reason: str | None


def _validate_waveform(waveform: Tensor, name: str) -> None:
    """Validate the shared channels-first floating-point waveform contract."""
    if not isinstance(waveform, Tensor):
        raise NoiseMixingError(f"{name} must be a torch.Tensor.")
    if waveform.ndim != 2:
        raise NoiseMixingError(
            f"{name} must have shape [channels, frames]; "
            f"received {tuple(waveform.shape)}."
        )
    if waveform.shape[0] < 1 or waveform.shape[1] < 1:
        raise NoiseMixingError(f"{name} must contain channels and frames.")
    if not waveform.is_floating_point():
        raise NoiseMixingError(f"{name} must use a floating-point dtype.")
    if not bool(torch.isfinite(waveform).all()):
        raise NoiseMixingError(f"{name} contains non-finite samples.")


def _validate_power_epsilon(power_epsilon: float) -> float:
    """Return a finite positive power threshold."""
    if (
        isinstance(power_epsilon, bool)
        or not isinstance(power_epsilon, (int, float))
        or not math.isfinite(float(power_epsilon))
        or power_epsilon <= 0
    ):
        raise NoiseMixingError("power_epsilon must be a positive finite number.")
    return float(power_epsilon)


def waveform_power(waveform: Tensor) -> float:
    """Return mean-square waveform power using float64 accumulation."""
    _validate_waveform(waveform, "waveform")
    power = waveform.detach().to(dtype=torch.float64).square().mean()
    return float(power.item())


def match_noise_length(
    noise_waveform: Tensor,
    target_num_frames: int,
    *,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Crop or cyclically repeat noise to an exact frame count.

    Long noise is cropped at a generator-selected offset. Short noise is treated
    as a circular recording and repeated from a generator-selected phase. A
    seeded generator therefore makes either policy reproducible.
    """
    _validate_waveform(noise_waveform, "noise_waveform")
    if (
        isinstance(target_num_frames, bool)
        or not isinstance(target_num_frames, int)
        or target_num_frames <= 0
    ):
        raise NoiseMixingError("target_num_frames must be a positive integer.")

    source_num_frames = int(noise_waveform.shape[1])
    if source_num_frames == target_num_frames:
        return noise_waveform

    if source_num_frames > target_num_frames:
        maximum_offset = source_num_frames - target_num_frames
        start = int(
            torch.randint(
                low=0,
                high=maximum_offset + 1,
                size=(),
                generator=generator,
            ).item()
        )
        return noise_waveform[:, start : start + target_num_frames]

    phase_offset = int(
        torch.randint(
            low=0,
            high=source_num_frames,
            size=(),
            generator=generator,
        ).item()
    )
    required_frames = target_num_frames + phase_offset
    repeat_count = math.ceil(required_frames / source_num_frames)
    repeated = noise_waveform.repeat(1, repeat_count)
    return repeated[:, phase_offset : phase_offset + target_num_frames]


def measure_snr_db(
    clean_waveform: Tensor,
    corrupted_waveform: Tensor,
    *,
    power_epsilon: float = 1.0e-12,
) -> float:
    """Measure SNR from a clean reference and its additive corruption."""
    _validate_waveform(clean_waveform, "clean_waveform")
    _validate_waveform(corrupted_waveform, "corrupted_waveform")
    epsilon = _validate_power_epsilon(power_epsilon)
    if clean_waveform.shape != corrupted_waveform.shape:
        raise NoiseMixingError(
            "clean_waveform and corrupted_waveform must have identical shapes."
        )

    signal_power = waveform_power(clean_waveform)
    added_noise_power = waveform_power(corrupted_waveform - clean_waveform)
    if signal_power <= epsilon:
        raise NoiseMixingError("SNR is undefined for a silent clean waveform.")
    if added_noise_power <= epsilon:
        return math.inf
    return 10.0 * math.log10(signal_power / added_noise_power)


def mix_waveforms_at_snr(
    clean_waveform: Tensor,
    noise_waveform: Tensor,
    target_snr_db: float,
    *,
    power_epsilon: float = 1.0e-12,
) -> SNRMixResult:
    """Add noise scaled to a requested signal-to-noise ratio.

    Power is the mean squared amplitude. The desired noise power is signal power
    divided by 10 raised to snr_db / 10. No clipping or peak normalization is
    applied because either operation would alter the controlled relationship.

    A near-silent clean signal is preserved unchanged: a finite SNR is undefined
    when signal power is zero, and injecting arbitrary noise would create a
    mislabeled research condition. Near-silent noise raises an explicit error.
    """
    _validate_waveform(clean_waveform, "clean_waveform")
    _validate_waveform(noise_waveform, "noise_waveform")
    epsilon = _validate_power_epsilon(power_epsilon)
    if clean_waveform.shape != noise_waveform.shape:
        raise NoiseMixingError(
            "clean_waveform and noise_waveform must have identical shapes."
        )
    if (
        isinstance(target_snr_db, bool)
        or not isinstance(target_snr_db, (int, float))
        or not math.isfinite(float(target_snr_db))
    ):
        raise NoiseMixingError("target_snr_db must be a finite number.")

    target = float(target_snr_db)
    signal_power = waveform_power(clean_waveform)
    source_noise_power = waveform_power(noise_waveform)
    if signal_power <= epsilon:
        scaled_noise = torch.zeros_like(noise_waveform)
        return SNRMixResult(
            waveform=clean_waveform.clone(),
            scaled_noise=scaled_noise,
            target_snr_db=target,
            achieved_snr_db=None,
            signal_power=signal_power,
            source_noise_power=source_noise_power,
            scaled_noise_power=0.0,
            scale_factor=0.0,
            applied=False,
            reason="silent_signal",
        )
    if source_noise_power <= epsilon:
        raise NoiseMixingError(
            "Noise power is too close to zero to create a controlled SNR mixture."
        )

    desired_noise_power = signal_power / (10.0 ** (target / 10.0))
    scale_factor = math.sqrt(desired_noise_power / source_noise_power)
    scaled_noise = noise_waveform * scale_factor
    corrupted_waveform = clean_waveform + scaled_noise
    scaled_noise_power = waveform_power(scaled_noise)
    achieved_snr_db = 10.0 * math.log10(signal_power / scaled_noise_power)
    return SNRMixResult(
        waveform=corrupted_waveform,
        scaled_noise=scaled_noise,
        target_snr_db=target,
        achieved_snr_db=achieved_snr_db,
        signal_power=signal_power,
        source_noise_power=source_noise_power,
        scaled_noise_power=scaled_noise_power,
        scale_factor=scale_factor,
        applied=True,
        reason=None,
    )
