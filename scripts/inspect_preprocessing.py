"""Run configured audio preprocessing on real samples and save inspection artifacts."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from tqdm import tqdm

from urban_sound_robustness.audio import (
    AudioPreprocessingError,
    AudioPreprocessor,
    load_audio,
)
from urban_sound_robustness.datasets import (
    AudioSampleRecord,
    DatasetNotFoundError,
    DatasetValidationError,
    create_dataset_adapter,
)
from urban_sound_robustness.utils.config import ConfigurationError, load_experiment_config
from urban_sound_robustness.utils.logging_utils import configure_logging
from urban_sound_robustness.utils.paths import resolve_path_settings


def parse_arguments() -> argparse.Namespace:
    """Parse preprocessing inspection options."""
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
        choices=("all", "train", "validation", "test"),
        default="test",
        help="Official split from which samples are selected.",
    )
    parser.add_argument(
        "--mode",
        choices=("evaluation", "training"),
        default="evaluation",
        help="Use center cropping or training-time random cropping.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=12,
        help="Number of evenly spaced records to preprocess.",
    )
    parser.add_argument(
        "--num-plots",
        type=int,
        default=3,
        help="Number of selected samples for which PNG plots are saved.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random-crop seed; defaults to training.seed from configuration.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=None,
        help="Override results/preprocessing_inspection.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the sample progress bar.",
    )
    arguments = parser.parse_args()
    if arguments.num_samples < 1:
        parser.error("--num-samples must be at least 1")
    if arguments.num_plots < 0:
        parser.error("--num-plots cannot be negative")
    if arguments.seed is not None and arguments.seed < 0:
        parser.error("--seed cannot be negative")
    return arguments


def _select_evenly_spaced_records(
    records: Sequence[AudioSampleRecord],
    count: int,
) -> list[AudioSampleRecord]:
    """Select deterministic records across the complete requested split."""
    if not records:
        raise ValueError("The requested dataset split contains no records.")
    if count >= len(records):
        return list(records)
    if count == 1:
        return [records[0]]
    indices = [round(index * (len(records) - 1) / (count - 1)) for index in range(count)]
    return [records[index] for index in indices]


def _save_plot(
    sample_id: str,
    waveform: torch.Tensor,
    features: torch.Tensor,
    sample_rate: int,
    output_path: Path,
) -> None:
    """Save one normalized waveform and log-Mel feature visualization."""
    waveform_values = waveform[0].detach().cpu().numpy()
    feature_values = features[0].detach().cpu().numpy()
    duration = waveform.shape[1] / sample_rate
    time_values = torch.arange(waveform.shape[1]).numpy() / sample_rate

    figure, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True)
    axes[0].plot(time_values, waveform_values, linewidth=0.6)
    axes[0].set(
        title=f"Normalized waveform: {sample_id}",
        xlabel="Time (seconds)",
        ylabel="Amplitude",
        xlim=(0, duration),
    )
    image = axes[1].imshow(
        feature_values,
        origin="lower",
        aspect="auto",
        extent=(0, duration, 0, feature_values.shape[0]),
        cmap="magma",
    )
    axes[1].set(
        title="Standardized log-Mel spectrogram",
        xlabel="Time (seconds)",
        ylabel="Mel bin",
    )
    figure.colorbar(image, ax=axes[1], label="Standardized value")
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _process_record(
    record: AudioSampleRecord,
    preprocessor: AudioPreprocessor,
    *,
    training: bool,
    generator: torch.Generator,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    """Preprocess one record and return metrics plus plot-ready tensors."""
    loaded = load_audio(record.audio_path)
    processed = preprocessor(
        loaded.waveform,
        loaded.sample_rate,
        training=training,
        generator=generator,
    )
    features = processed.features
    row: dict[str, Any] = {
        "sample_id": record.sample_id,
        "label": record.label,
        "class_name": record.class_name,
        "fold": record.fold,
        "source_path": str(record.audio_path),
        "source_sample_rate": loaded.sample_rate,
        "source_channels": loaded.num_channels,
        "source_frames": loaded.num_frames,
        "source_duration_seconds": loaded.duration_seconds,
        "target_sample_rate": processed.target_sample_rate,
        "target_channels": int(processed.waveform.shape[0]),
        "target_frames": int(processed.waveform.shape[1]),
        "target_duration_seconds": (
            processed.waveform.shape[1] / processed.target_sample_rate
        ),
        "feature_channels": int(features.shape[0]),
        "feature_mels": int(features.shape[1]),
        "feature_time_frames": int(features.shape[2]),
        "feature_mean": float(features.mean()),
        "feature_std": float(features.std(unbiased=False)),
        "feature_min": float(features.min()),
        "feature_max": float(features.max()),
        "features_finite": bool(torch.isfinite(features).all()),
    }
    return row, processed.waveform, features


def main() -> int:
    """Run a bounded real-data preprocessing inspection."""
    arguments = parse_arguments()
    logger: logging.Logger | None = None
    try:
        project_config = load_experiment_config(arguments.config)
        resolved_paths = resolve_path_settings(
            project_config.section("paths"), project_config.project_root
        )
        logger = configure_logging(
            resolved_paths["logs"] / "preprocessing_inspection.log",
            level="INFO",
        )
        output_directory = arguments.output_directory
        if output_directory is None:
            output_directory = resolved_paths["results"] / "preprocessing_inspection"
        output_directory = output_directory.expanduser().resolve()
        output_directory.mkdir(parents=True, exist_ok=True)

        adapter = create_dataset_adapter(
            project_config.section("dataset"), project_config.project_root
        )
        records = (
            adapter.load_records()
            if arguments.split == "all"
            else adapter.records_for_split(arguments.split)
        )
        selected_records = _select_evenly_spaced_records(
            records, arguments.num_samples
        )
        preprocessor = AudioPreprocessor(project_config.section("audio"))
        training = arguments.mode == "training"
        seed = (
            int(project_config.section("training")["seed"])
            if arguments.seed is None
            else arguments.seed
        )
        generator = torch.Generator().manual_seed(seed)

        rows: list[dict[str, Any]] = []
        progress = tqdm(
            selected_records,
            desc="Preprocessing samples",
            unit="sample",
            disable=arguments.no_progress,
        )
        for index, record in enumerate(progress):
            row, waveform, features = _process_record(
                record,
                preprocessor,
                training=training,
                generator=generator,
            )
            rows.append(row)
            if index < min(arguments.num_plots, len(selected_records)):
                plot_path = output_directory / f"{Path(record.sample_id).stem}.png"
                _save_plot(
                    record.sample_id,
                    waveform,
                    features,
                    preprocessor.sample_rate,
                    plot_path,
                )

        inventory_path = output_directory / "sample_outputs.csv"
        pd.DataFrame(rows).to_csv(inventory_path, index=False)
        feature_shapes = sorted(
            {
                (row["feature_channels"], row["feature_mels"], row["feature_time_frames"])
                for row in rows
            }
        )
        summary = {
            "dataset": adapter.dataset_name,
            "split": arguments.split,
            "mode": arguments.mode,
            "seed": seed,
            "processed_samples": len(rows),
            "saved_plots": min(arguments.num_plots, len(rows)),
            "target_sample_rate": preprocessor.sample_rate,
            "target_duration_seconds": preprocessor.clip_duration_seconds,
            "target_num_frames": preprocessor.target_num_frames,
            "feature_shapes": [list(shape) for shape in feature_shapes],
            "all_features_finite": all(row["features_finite"] for row in rows),
            "source_sample_rates": sorted({row["source_sample_rate"] for row in rows}),
            "source_channel_counts": sorted({row["source_channels"] for row in rows}),
        }
        summary_path = output_directory / "summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (
        AudioPreprocessingError,
        ConfigurationError,
        DatasetNotFoundError,
        DatasetValidationError,
        OSError,
        ValueError,
    ) as error:
        if logger is None:
            logging.basicConfig(level=logging.ERROR)
            logger = logging.getLogger("preprocessing_inspection")
        logger.error("Preprocessing inspection failed: %s", error)
        return 1

    logger.info("Processed samples: %d", len(rows))
    logger.info("Feature shapes: %s", feature_shapes)
    logger.info("Saved summary: %s", summary_path)
    logger.info("Saved sample inventory: %s", inventory_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
