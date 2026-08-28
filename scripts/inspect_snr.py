"""Inspect configured SNR conditions on one real UrbanSound8K waveform."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import soundfile as sf
import torch

from urban_sound_robustness.audio import (
    AudioPreprocessor,
    NoiseMixingError,
    convert_to_mono,
    load_audio,
    match_noise_length,
    mix_waveforms_at_snr,
    resample_waveform,
)
from urban_sound_robustness.datasets import create_dataset_adapter
from urban_sound_robustness.evaluation import parse_robustness_conditions, stable_seed
from urban_sound_robustness.utils.config import load_experiment_config
from urban_sound_robustness.utils.logging_utils import configure_logging
from urban_sound_robustness.utils.paths import resolve_path_settings


def parse_arguments() -> argparse.Namespace:
    """Parse the bounded SNR-inspection command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "config",
        nargs="?",
        type=Path,
        default=Path("configs/experiment/development.yaml"),
        help="Composed experiment configuration.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "validation", "test"),
        default="test",
        help="Configured split containing the clean sample.",
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=0,
        help="Zero-based record index within the selected split.",
    )
    parser.add_argument(
        "--noise-file",
        type=Path,
        default=None,
        help="Optional real noise file; seeded white noise is used when omitted.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override evaluation.corruption_seed.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=None,
        help="Override results/snr_inspection.",
    )
    arguments = parser.parse_args()
    if arguments.sample_index < 0:
        parser.error("--sample-index cannot be negative")
    if arguments.seed is not None and arguments.seed < 0:
        parser.error("--seed cannot be negative")
    return arguments


def _prepare_reference_noise(
    noise_path: Path | None,
    *,
    target_sample_rate: int,
    target_shape: torch.Size,
    seed: int,
) -> tuple[torch.Tensor, str]:
    """Create or load one noise segment shared by every requested SNR."""
    generator = torch.Generator().manual_seed(seed)
    if noise_path is None:
        noise = torch.randn(target_shape, generator=generator)
        return noise, "synthetic_white_noise"

    loaded = load_audio(noise_path)
    noise = convert_to_mono(loaded.waveform)
    noise = resample_waveform(noise, loaded.sample_rate, target_sample_rate)
    noise = match_noise_length(
        noise,
        int(target_shape[1]),
        generator=generator,
    )
    if target_shape[0] > 1:
        noise = noise.expand(target_shape[0], -1)
    return noise, str(loaded.source_path)


def _save_listening_wav(
    path: Path,
    waveform: torch.Tensor,
    sample_rate: int,
    shared_scale_factor: float,
) -> None:
    """Save deterministic PCM-24 audio after shared peak-safety scaling."""
    scaled = waveform / shared_scale_factor
    frames_first = scaled.detach().cpu().transpose(0, 1).numpy()
    sf.write(path, frames_first, sample_rate, subtype="PCM_24")


def _save_comparison_plot(
    rows: list[dict],
    waveforms: list[torch.Tensor],
    sample_rate: int,
    output_path: Path,
) -> None:
    """Save aligned clean/noisy waveforms for visual SNR comparison."""
    duration = waveforms[0].shape[1] / sample_rate
    time_values = torch.arange(waveforms[0].shape[1]).numpy() / sample_rate
    maximum_amplitude = max(float(waveform.abs().max()) for waveform in waveforms)
    figure, axes = plt.subplots(
        len(waveforms),
        1,
        figsize=(12, 2.4 * len(waveforms)),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    if len(waveforms) == 1:
        axes = [axes]
    for axis, row, waveform in zip(axes, rows, waveforms):
        axis.plot(time_values, waveform[0].cpu().numpy(), linewidth=0.55)
        achieved = row["achieved_snr_db"]
        achieved_text = "clean" if achieved is None else f"achieved {achieved:.3f} dB"
        axis.set(
            title=f"{row['condition']} ({achieved_text})",
            ylabel="Amplitude",
            xlim=(0, duration),
            ylim=(-maximum_amplitude * 1.05, maximum_amplitude * 1.05),
        )
    axes[-1].set_xlabel("Time (seconds)")
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> int:
    """Create listening, plotting, and numerical SNR inspection artifacts."""
    arguments = parse_arguments()
    project_config = load_experiment_config(arguments.config)
    paths = resolve_path_settings(
        project_config.section("paths"), project_config.project_root
    )
    logger = configure_logging(paths["logs"] / "snr_inspection.log", level="INFO")
    try:
        output_directory = arguments.output_directory
        if output_directory is None:
            output_directory = paths["results"] / "snr_inspection"
        output_directory = output_directory.expanduser().resolve()
        output_directory.mkdir(parents=True, exist_ok=True)

        adapter = create_dataset_adapter(
            project_config.section("dataset"), project_config.project_root
        )
        records = adapter.records_for_split(arguments.split)
        if arguments.sample_index >= len(records):
            raise ValueError(
                f"--sample-index {arguments.sample_index} is outside split "
                f"size {len(records)}."
            )
        record = records[arguments.sample_index]
        loaded_clean = load_audio(record.audio_path)
        preprocessor = AudioPreprocessor(project_config.section("audio")).eval()
        clean = preprocessor(
            loaded_clean.waveform,
            loaded_clean.sample_rate,
            training=False,
        ).waveform

        evaluation_settings = project_config.section("evaluation")
        conditions = parse_robustness_conditions(evaluation_settings)
        corruption_seed = (
            int(evaluation_settings["corruption_seed"])
            if arguments.seed is None
            else arguments.seed
        )
        reference_noise, noise_source = _prepare_reference_noise(
            arguments.noise_file,
            target_sample_rate=preprocessor.sample_rate,
            target_shape=clean.shape,
            seed=stable_seed(corruption_seed, record.sample_id, "inspection"),
        )
        power_epsilon = float(evaluation_settings["mixing"]["power_epsilon"])

        rows: list[dict] = []
        condition_waveforms: list[torch.Tensor] = []
        for condition in conditions:
            if condition.snr_db is None:
                condition_waveform = clean.clone()
                row = {
                    "condition": condition.name,
                    "target_snr_db": None,
                    "achieved_snr_db": None,
                    "absolute_snr_error_db": None,
                    "applied": False,
                    "reason": "clean_condition",
                    "signal_power": None,
                    "scaled_noise_power": 0.0,
                    "scale_factor": 0.0,
                }
            else:
                mixture = mix_waveforms_at_snr(
                    clean,
                    reference_noise,
                    condition.snr_db,
                    power_epsilon=power_epsilon,
                )
                condition_waveform = mixture.waveform
                row = {
                    "condition": condition.name,
                    "target_snr_db": mixture.target_snr_db,
                    "achieved_snr_db": mixture.achieved_snr_db,
                    "absolute_snr_error_db": (
                        None
                        if mixture.achieved_snr_db is None
                        else abs(mixture.achieved_snr_db - mixture.target_snr_db)
                    ),
                    "applied": mixture.applied,
                    "reason": mixture.reason,
                    "signal_power": mixture.signal_power,
                    "scaled_noise_power": mixture.scaled_noise_power,
                    "scale_factor": mixture.scale_factor,
                }
            rows.append(row)
            condition_waveforms.append(condition_waveform)

        # One common factor prevents integer-WAV clipping while preserving the
        # SNR and relative amplitude relationship across every condition.
        maximum_amplitude = max(
            float(waveform.abs().max()) for waveform in condition_waveforms
        )
        listening_scale_factor = max(1.0, maximum_amplitude / 0.999)
        for condition, condition_waveform in zip(conditions, condition_waveforms):
            _save_listening_wav(
                output_directory / f"{condition.name}.wav",
                condition_waveform,
                preprocessor.sample_rate,
                listening_scale_factor,
            )

        results_path = output_directory / "condition_results.csv"
        pd.DataFrame(rows).to_csv(results_path, index=False)
        plot_path = output_directory / "waveform_comparison.png"
        _save_comparison_plot(
            rows,
            condition_waveforms,
            preprocessor.sample_rate,
            plot_path,
        )
        summary = {
            "sample_id": record.sample_id,
            "class_name": record.class_name,
            "split": arguments.split,
            "sample_index": arguments.sample_index,
            "sample_rate": preprocessor.sample_rate,
            "num_frames": int(clean.shape[1]),
            "corruption_seed": corruption_seed,
            "noise_source": noise_source,
            "synthetic_noise_for_inspection_only": arguments.noise_file is None,
            "listening_wav_scale_factor": listening_scale_factor,
            "conditions": rows,
        }
        summary_path = output_directory / "summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (NoiseMixingError, OSError, ValueError) as error:
        logger.error("SNR inspection failed: %s", error)
        return 1

    logger.info("Clean sample: %s (%s)", record.sample_id, record.class_name)
    logger.info("Noise source: %s", noise_source)
    logger.info("Saved summary: %s", summary_path)
    logger.info("Saved condition table: %s", results_path)
    logger.info("Saved comparison plot: %s", plot_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
