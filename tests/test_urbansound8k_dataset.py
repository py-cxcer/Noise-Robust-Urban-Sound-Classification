"""Tests for UrbanSound8K metadata integration and generic inspection."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from urban_sound_robustness.datasets.inspection import (
    inspect_dataset,
    save_inspection_result,
)
from urban_sound_robustness.datasets.urbansound8k import (
    DatasetValidationError,
    UrbanSound8KAdapter,
)


@pytest.fixture
def urbansound_fixture(tmp_path: Path) -> tuple[UrbanSound8KAdapter, Path]:
    """Create a tiny UrbanSound8K-shaped dataset with valid and invalid files."""
    dataset_root = tmp_path / "UrbanSound8K"
    metadata_directory = dataset_root / "metadata"
    fold1_directory = dataset_root / "audio" / "fold1"
    fold2_directory = dataset_root / "audio" / "fold2"
    metadata_directory.mkdir(parents=True)
    fold1_directory.mkdir(parents=True)
    fold2_directory.mkdir(parents=True)

    first_waveform = np.linspace(-0.25, 0.25, 2000, dtype=np.float32)
    second_waveform = np.linspace(-0.5, 0.5, 8000, dtype=np.float32)
    sf.write(fold1_directory / "1000-0-0-0.wav", first_waveform, 8000)
    sf.write(fold2_directory / "1001-1-0-0.wav", second_waveform, 16000)
    (fold2_directory / "1003-1-0-1.wav").write_text(
        "not an audio file", encoding="utf-8"
    )

    metadata = pd.DataFrame(
        [
            _metadata_row("1000-0-0-0.wav", 1000, 0.0, 0.25, 1, 1, 0, "class_zero"),
            _metadata_row("1001-1-0-0.wav", 1001, 2.0, 2.5, 2, 2, 1, "class_one"),
            _metadata_row("1002-0-0-1.wav", 1002, 3.0, 3.2, 1, 1, 0, "class_zero"),
            _metadata_row("1003-1-0-1.wav", 1003, 4.0, 4.4, 2, 2, 1, "class_one"),
        ]
    )
    metadata.to_csv(metadata_directory / "UrbanSound8K.csv", index=False)
    settings = {
        "name": "urbansound8k",
        "adapter": "urbansound8k",
        "dataset_root": str(dataset_root),
        "audio_directory": "audio",
        "metadata_file": "metadata/UrbanSound8K.csv",
        "expected_num_samples": 4,
        "available_folds": [1, 2],
        "folds": {"train": [1], "validation": [2], "test": [2]},
        "num_classes": 2,
        "class_names": ["class_zero", "class_one"],
    }
    return UrbanSound8KAdapter(settings, tmp_path), dataset_root


def test_adapter_preserves_labels_paths_folds_and_metadata(
    urbansound_fixture: tuple[UrbanSound8KAdapter, Path],
) -> None:
    """Validated rows should become clearly structured common records."""
    adapter, dataset_root = urbansound_fixture

    records = adapter.load_records()

    assert len(records) == 4
    assert records[0].label == 0
    assert records[0].class_name == "class_zero"
    assert records[0].fold == 1
    assert records[0].metadata["fsID"] == 1000
    assert records[0].audio_path == (
        dataset_root / "audio" / "fold1" / "1000-0-0-0.wav"
    ).resolve()
    assert len(adapter.records_for_split("train")) == 2
    assert len(adapter.records_for_folds([2])) == 2


def test_adapter_reports_missing_audio_files(
    urbansound_fixture: tuple[UrbanSound8KAdapter, Path],
) -> None:
    """Missing files remain visible for inspection instead of disappearing."""
    adapter, _ = urbansound_fixture

    missing_files = adapter.missing_audio_files(adapter.load_records())

    assert [path.name for path in missing_files] == ["1002-0-0-1.wav"]


def test_missing_metadata_column_is_rejected(
    urbansound_fixture: tuple[UrbanSound8KAdapter, Path],
) -> None:
    """Schema errors should identify absent UrbanSound8K columns."""
    adapter, _ = urbansound_fixture
    metadata = pd.read_csv(adapter.metadata_path).drop(columns=["salience"])
    metadata.to_csv(adapter.metadata_path, index=False)

    with pytest.raises(DatasetValidationError, match="salience"):
        adapter.load_metadata()


def test_class_mapping_inconsistency_is_rejected(
    urbansound_fixture: tuple[UrbanSound8KAdapter, Path],
) -> None:
    """Configured class IDs and names must agree on every row."""
    adapter, _ = urbansound_fixture
    metadata = pd.read_csv(adapter.metadata_path)
    metadata.loc[0, "class"] = "class_one"
    metadata.to_csv(adapter.metadata_path, index=False)

    with pytest.raises(DatasetValidationError, match="Class IDs"):
        adapter.load_metadata()


def test_filename_metadata_inconsistency_is_rejected(
    urbansound_fixture: tuple[UrbanSound8KAdapter, Path],
) -> None:
    """Filename-derived source and class IDs provide an additional integrity check."""
    adapter, _ = urbansound_fixture
    metadata = pd.read_csv(adapter.metadata_path)
    metadata.loc[0, "fsID"] = 9999
    metadata.to_csv(adapter.metadata_path, index=False)

    with pytest.raises(DatasetValidationError, match="fsID mismatch"):
        adapter.load_metadata()


def test_inspection_reads_headers_and_saves_structured_outputs(
    urbansound_fixture: tuple[UrbanSound8KAdapter, Path],
    tmp_path: Path,
) -> None:
    """Inspection should report distributions, missing files, and corrupt audio."""
    adapter, _ = urbansound_fixture
    result = inspect_dataset(
        records=adapter.load_records(),
        class_names=adapter.class_names,
        inspect_audio_headers=True,
        imbalance_warning_ratio=1.5,
        show_progress=False,
    )

    assert result.summary["total_samples"] == 4
    assert result.summary["missing_file_count"] == 1
    assert result.summary["unreadable_file_count"] == 1
    assert result.summary["readable_file_count"] == 2
    assert result.summary["obvious_class_imbalance"] is False
    assert result.sample_rate_distribution["sample_rate"].tolist() == [8000, 16000]

    output_paths = save_inspection_result(result, tmp_path / "inspection")

    assert all(path.is_file() for path in output_paths.values())
    assert pd.read_csv(output_paths["sample_inventory"]).shape[0] == 4


def _metadata_row(
    file_name: str,
    freesound_id: int,
    start: float,
    end: float,
    salience: int,
    fold: int,
    class_id: int,
    class_name: str,
) -> dict[str, Any]:
    """Build one readable metadata row for the test fixture."""
    return {
        "slice_file_name": file_name,
        "fsID": freesound_id,
        "start": start,
        "end": end,
        "salience": salience,
        "fold": fold,
        "classID": class_id,
        "class": class_name,
    }

