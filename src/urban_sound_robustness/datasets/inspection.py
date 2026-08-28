"""Reusable metadata and lightweight audio-header dataset inspection."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd
import soundfile as sf
from tqdm import tqdm

from urban_sound_robustness.datasets.records import AudioSampleRecord


@dataclass(frozen=True)
class DatasetInspectionResult:
    """Structured numerical outputs produced by dataset inspection."""

    summary: dict[str, Any]
    class_distribution: pd.DataFrame
    fold_distribution: pd.DataFrame
    sample_rate_distribution: pd.DataFrame
    sample_inventory: pd.DataFrame


def inspect_dataset(
    records: Iterable[AudioSampleRecord],
    class_names: Sequence[str],
    inspect_audio_headers: bool = True,
    imbalance_warning_ratio: float = 1.5,
    show_progress: bool = True,
) -> DatasetInspectionResult:
    """
    Inspect generic audio records without decoding full waveforms.

    Parameters
    ----------
    records : Iterable[AudioSampleRecord]
        Validated records from any single-label audio dataset adapter.
    class_names : Sequence[str]
        Complete class list ordered by numeric label.
    inspect_audio_headers : bool
        Read duration, sample rate, channels, and frame count using file headers.
        This is much cheaper than loading every waveform into memory.
    imbalance_warning_ratio : float
        Maximum-count divided by minimum-count threshold used for the summary flag.
    show_progress : bool
        Display a progress bar during audio-header inspection.

    Returns
    -------
    DatasetInspectionResult
        Summary, distributions, and per-sample inventory tables.
    """
    sample_records = list(records)

    if not sample_records:
        raise ValueError("Dataset inspection requires at least one sample record.")

    if imbalance_warning_ratio <= 1:
        raise ValueError("The imbalance warning ratio must be greater than one.")

    inventory_rows: list[dict[str, Any]] = []
    progress_description = "Inspecting audio headers" if inspect_audio_headers else None
    record_iterator = tqdm(
        sample_records,
        desc=progress_description,
        disable=not show_progress or not inspect_audio_headers,
        unit="file",
    )

    for record in record_iterator:
        inventory_rows.append(
            _inspect_one_record(record, inspect_audio_headers=inspect_audio_headers)
        )

    inventory = pd.DataFrame(inventory_rows)
    class_distribution = _build_class_distribution(sample_records, class_names)
    fold_distribution = _build_fold_distribution(sample_records)
    sample_rate_distribution = _build_sample_rate_distribution(inventory)
    summary = _build_summary(
        records=sample_records,
        inventory=inventory,
        class_distribution=class_distribution,
        inspect_audio_headers=inspect_audio_headers,
        imbalance_warning_ratio=imbalance_warning_ratio,
    )

    return DatasetInspectionResult(
        summary=summary,
        class_distribution=class_distribution,
        fold_distribution=fold_distribution,
        sample_rate_distribution=sample_rate_distribution,
        sample_inventory=inventory,
    )


def save_inspection_result(
    result: DatasetInspectionResult,
    output_directory: str | Path,
) -> dict[str, Path]:
    """
    Save a dataset inspection result to JSON and CSV files.

    Parameters
    ----------
    result : DatasetInspectionResult
        Inspection output to serialize.
    output_directory : str or Path
        Destination directory. It is created when necessary.

    Returns
    -------
    dict[str, Path]
        Names and resolved paths of all generated files.
    """
    resolved_output = Path(output_directory).expanduser().resolve()
    resolved_output.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "summary": resolved_output / "summary.json",
        "class_distribution": resolved_output / "class_distribution.csv",
        "fold_distribution": resolved_output / "fold_distribution.csv",
        "sample_rate_distribution": resolved_output / "sample_rate_distribution.csv",
        "sample_inventory": resolved_output / "sample_inventory.csv",
    }

    with output_paths["summary"].open("w", encoding="utf-8") as file_handle:
        json.dump(result.summary, file_handle, indent=2, sort_keys=True)
        file_handle.write("\n")

    result.class_distribution.to_csv(
        output_paths["class_distribution"], index=False
    )
    result.fold_distribution.to_csv(output_paths["fold_distribution"], index=False)
    result.sample_rate_distribution.to_csv(
        output_paths["sample_rate_distribution"], index=False
    )
    result.sample_inventory.to_csv(output_paths["sample_inventory"], index=False)
    return output_paths


def _inspect_one_record(
    record: AudioSampleRecord,
    inspect_audio_headers: bool,
) -> dict[str, Any]:
    metadata_start = record.metadata.get("start")
    metadata_end = record.metadata.get("end")
    metadata_duration = None

    if isinstance(metadata_start, (int, float)) and isinstance(
        metadata_end, (int, float)
    ):
        metadata_duration = float(metadata_end - metadata_start)

    file_exists = record.audio_path.is_file()
    inventory_row: dict[str, Any] = {
        "sample_id": record.sample_id,
        "file_path": str(record.audio_path),
        "label": record.label,
        "class_name": record.class_name,
        "fold": record.fold,
        "metadata_duration_seconds": metadata_duration,
        "file_exists": file_exists,
        "readable": None,
        "audio_duration_seconds": None,
        "sample_rate": None,
        "channels": None,
        "frames": None,
        "error": None,
    }

    if not file_exists:
        inventory_row["readable"] = False
        inventory_row["error"] = "file_not_found"
        return inventory_row

    if not inspect_audio_headers:
        return inventory_row

    try:
        audio_information = sf.info(record.audio_path)
    except (OSError, RuntimeError) as error:
        inventory_row["readable"] = False
        inventory_row["error"] = str(error)
        return inventory_row

    inventory_row["readable"] = True
    inventory_row["audio_duration_seconds"] = float(audio_information.duration)
    inventory_row["sample_rate"] = int(audio_information.samplerate)
    inventory_row["channels"] = int(audio_information.channels)
    inventory_row["frames"] = int(audio_information.frames)
    return inventory_row


def _build_class_distribution(
    records: Sequence[AudioSampleRecord],
    class_names: Sequence[str],
) -> pd.DataFrame:
    observed_counts = pd.Series(
        [record.class_name for record in records], dtype="object"
    ).value_counts()
    rows: list[dict[str, Any]] = []

    for label, class_name in enumerate(class_names):
        sample_count = int(observed_counts.get(class_name, 0))
        rows.append(
            {
                "label": label,
                "class_name": class_name,
                "sample_count": sample_count,
                "fraction": sample_count / len(records),
            }
        )

    return pd.DataFrame(rows)


def _build_fold_distribution(
    records: Sequence[AudioSampleRecord],
) -> pd.DataFrame:
    fold_counts = pd.Series(
        [record.fold for record in records], dtype="Int64"
    ).value_counts(dropna=False)
    rows: list[dict[str, Any]] = []

    for fold, sample_count in fold_counts.sort_index().items():
        rows.append(
            {
                "fold": None if pd.isna(fold) else int(fold),
                "sample_count": int(sample_count),
                "fraction": int(sample_count) / len(records),
            }
        )

    return pd.DataFrame(rows)


def _build_sample_rate_distribution(inventory: pd.DataFrame) -> pd.DataFrame:
    valid_sample_rates = inventory["sample_rate"].dropna().astype(int)
    sample_rate_counts = valid_sample_rates.value_counts().sort_index()
    rows = [
        {"sample_rate": int(sample_rate), "sample_count": int(sample_count)}
        for sample_rate, sample_count in sample_rate_counts.items()
    ]
    return pd.DataFrame(rows, columns=["sample_rate", "sample_count"])


def _build_summary(
    records: Sequence[AudioSampleRecord],
    inventory: pd.DataFrame,
    class_distribution: pd.DataFrame,
    inspect_audio_headers: bool,
    imbalance_warning_ratio: float,
) -> dict[str, Any]:
    class_counts = class_distribution["sample_count"]
    maximum_class_count = int(class_counts.max())
    minimum_class_count = int(class_counts.min())
    imbalance_ratio = None

    if minimum_class_count > 0:
        imbalance_ratio = maximum_class_count / minimum_class_count

    obvious_imbalance = (
        minimum_class_count == 0
        or imbalance_ratio is not None
        and imbalance_ratio >= imbalance_warning_ratio
    )

    readable_mask = inventory["readable"] == True  # noqa: E712
    missing_mask = inventory["error"] == "file_not_found"
    unreadable_mask = (inventory["readable"] == False) & ~missing_mask  # noqa: E712
    header_durations = inventory.loc[readable_mask, "audio_duration_seconds"].dropna()
    metadata_durations = inventory["metadata_duration_seconds"].dropna()
    durations = header_durations if not header_durations.empty else metadata_durations
    duration_source = "audio_header" if not header_durations.empty else "metadata"

    summary: dict[str, Any] = {
        "dataset_name": records[0].dataset_name,
        "total_samples": len(records),
        "num_classes": len(class_distribution),
        "class_names": class_distribution["class_name"].tolist(),
        "folds": sorted(
            {int(record.fold) for record in records if record.fold is not None}
        ),
        "audio_header_scan_performed": inspect_audio_headers,
        "missing_file_count": int(missing_mask.sum()),
        "unreadable_file_count": int(unreadable_mask.sum()),
        "readable_file_count": int(readable_mask.sum()),
        "minimum_class_count": minimum_class_count,
        "maximum_class_count": maximum_class_count,
        "class_imbalance_ratio": imbalance_ratio,
        "imbalance_warning_ratio": imbalance_warning_ratio,
        "obvious_class_imbalance": obvious_imbalance,
        "duration_source": duration_source,
    }

    if durations.empty:
        summary.update(
            {
                "minimum_duration_seconds": None,
                "maximum_duration_seconds": None,
                "mean_duration_seconds": None,
                "median_duration_seconds": None,
                "total_duration_hours": None,
            }
        )
    else:
        summary.update(
            {
                "minimum_duration_seconds": float(durations.min()),
                "maximum_duration_seconds": float(durations.max()),
                "mean_duration_seconds": float(durations.mean()),
                "median_duration_seconds": float(durations.median()),
                "total_duration_hours": float(durations.sum() / 3600),
            }
        )

    return summary

