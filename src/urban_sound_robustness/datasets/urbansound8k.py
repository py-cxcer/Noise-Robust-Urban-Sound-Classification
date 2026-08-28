"""UrbanSound8K metadata validation and dataset record construction."""

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from urban_sound_robustness.datasets.records import AudioSampleRecord
from urban_sound_robustness.utils.paths import resolve_project_path


REQUIRED_METADATA_COLUMNS = (
    "slice_file_name",
    "fsID",
    "start",
    "end",
    "salience",
    "fold",
    "classID",
    "class",
)


class DatasetNotFoundError(FileNotFoundError):
    """Raised when the configured dataset or metadata file is unavailable."""


class DatasetValidationError(ValueError):
    """Raised when dataset metadata is internally inconsistent."""


class UrbanSound8KAdapter:
    """Translate UrbanSound8K-specific metadata into common audio records."""

    def __init__(
        self,
        dataset_settings: Mapping[str, Any],
        project_root: str | Path,
    ) -> None:
        """
        Configure the adapter without loading metadata or audio.

        Parameters
        ----------
        dataset_settings : Mapping[str, Any]
            Resolved ``dataset`` configuration section.
        project_root : str or Path
            Repository root used for relative paths.
        """
        self._settings = dict(dataset_settings)
        self._dataset_name = str(self._settings.get("name", "urbansound8k"))
        self._project_root = Path(project_root).expanduser().resolve()
        self._dataset_root = resolve_project_path(
            str(self._settings.get("dataset_root", "")),
            self._project_root,
        )
        self._audio_directory = self._dataset_root / str(
            self._settings.get("audio_directory", "audio")
        )
        self._metadata_path = self._dataset_root / str(
            self._settings.get("metadata_file", "metadata/UrbanSound8K.csv")
        )
        self._class_names = tuple(self._settings.get("class_names", ()))
        self._available_folds = tuple(
            self._settings.get("available_folds", range(1, 11))
        )
        self._cached_records: list[AudioSampleRecord] | None = None

    @property
    def dataset_name(self) -> str:
        """Return the stable dataset identifier."""
        return self._dataset_name

    @property
    def class_names(self) -> Sequence[str]:
        """Return class names ordered by UrbanSound8K class ID."""
        return self._class_names

    @property
    def dataset_root(self) -> Path:
        """Return the resolved UrbanSound8K root directory."""
        return self._dataset_root

    @property
    def metadata_path(self) -> Path:
        """Return the resolved metadata CSV location."""
        return self._metadata_path

    def load_metadata(self) -> pd.DataFrame:
        """
        Load and validate the UrbanSound8K metadata CSV.

        Returns
        -------
        pandas.DataFrame
            Validated metadata with stable numeric column types.

        Raises
        ------
        DatasetNotFoundError
            If the dataset root or metadata CSV is missing.
        DatasetValidationError
            If columns, labels, folds, timestamps, or filenames are inconsistent.
        """
        if not self._dataset_root.is_dir():
            raise DatasetNotFoundError(
                f"UrbanSound8K directory not found: {self._dataset_root}. "
                "Place the extracted dataset under data/raw/UrbanSound8K or "
                "change dataset.dataset_root in configuration."
            )

        if not self._metadata_path.is_file():
            raise DatasetNotFoundError(
                f"UrbanSound8K metadata CSV not found: {self._metadata_path}"
            )

        try:
            metadata = pd.read_csv(self._metadata_path)
        except (OSError, pd.errors.ParserError) as error:
            raise DatasetValidationError(
                f"Could not read UrbanSound8K metadata '{self._metadata_path}': {error}"
            ) from error

        missing_columns = set(REQUIRED_METADATA_COLUMNS) - set(metadata.columns)

        if missing_columns:
            missing_names = ", ".join(sorted(missing_columns))
            raise DatasetValidationError(
                f"UrbanSound8K metadata is missing required columns: {missing_names}"
            )

        normalized_metadata = self._normalize_numeric_columns(metadata)
        self._validate_metadata(normalized_metadata)
        return normalized_metadata

    def load_records(self) -> list[AudioSampleRecord]:
        """
        Return all validated UrbanSound8K samples as common audio records.

        Returns
        -------
        list[AudioSampleRecord]
            Records retaining file paths, labels, folds, and original metadata.
        """
        if self._cached_records is not None:
            return list(self._cached_records)

        metadata = self.load_metadata()
        records: list[AudioSampleRecord] = []

        for row_values in metadata.to_dict(orient="records"):
            file_name = str(row_values["slice_file_name"])
            fold = int(row_values["fold"])
            label = int(row_values["classID"])
            audio_path = self._audio_directory / f"fold{fold}" / file_name
            record = AudioSampleRecord(
                sample_id=file_name,
                audio_path=audio_path.resolve(),
                label=label,
                class_name=str(row_values["class"]),
                fold=fold,
                dataset_name=self._dataset_name,
                metadata=dict(row_values),
            )
            records.append(record)

        self._cached_records = records
        return list(records)

    def records_for_split(self, split_name: str) -> list[AudioSampleRecord]:
        """
        Return records belonging to the configured folds for one split.

        Parameters
        ----------
        split_name : str
            ``train``, ``validation``, or ``test``.

        Returns
        -------
        list[AudioSampleRecord]
            Records whose official fold is assigned to the split.
        """
        fold_settings = self._settings.get("folds", {})

        if split_name not in fold_settings:
            raise ValueError(
                f"Unknown split '{split_name}'. Available splits: "
                f"{sorted(fold_settings)}"
            )

        return self.records_for_folds(fold_settings[split_name])

    def records_for_folds(self, folds: Iterable[int]) -> list[AudioSampleRecord]:
        """Return records from an explicit collection of official folds."""
        requested_folds = set(folds)
        invalid_folds = requested_folds - set(self._available_folds)

        if not requested_folds:
            raise ValueError("At least one fold must be requested.")

        if invalid_folds:
            raise ValueError(f"Unsupported UrbanSound8K folds: {sorted(invalid_folds)}")

        return [
            record for record in self.load_records() if record.fold in requested_folds
        ]

    @staticmethod
    def missing_audio_files(records: Iterable[AudioSampleRecord]) -> list[Path]:
        """Return audio paths referenced by metadata but absent from disk."""
        return [record.audio_path for record in records if not record.audio_path.is_file()]

    def _normalize_numeric_columns(self, metadata: pd.DataFrame) -> pd.DataFrame:
        """Convert numeric metadata columns and report non-numeric values clearly."""
        normalized = metadata.copy()
        numeric_columns = ("fsID", "start", "end", "salience", "fold", "classID")

        for column_name in numeric_columns:
            if column_name not in normalized.columns:
                continue

            converted_values = pd.to_numeric(normalized[column_name], errors="coerce")
            invalid_mask = converted_values.isna()

            if invalid_mask.any():
                row_numbers = (normalized.index[invalid_mask] + 2).tolist()[:10]
                raise DatasetValidationError(
                    f"Metadata column '{column_name}' contains missing or non-numeric "
                    f"values at CSV rows {row_numbers}."
                )

            normalized[column_name] = converted_values

        integer_columns = ("fsID", "salience", "fold", "classID")

        for column_name in integer_columns:
            values = normalized[column_name]
            non_integer_mask = values % 1 != 0

            if non_integer_mask.any():
                row_numbers = (normalized.index[non_integer_mask] + 2).tolist()[:10]
                raise DatasetValidationError(
                    f"Metadata column '{column_name}' must contain integers; invalid "
                    f"CSV rows: {row_numbers}."
                )

            normalized[column_name] = values.astype(int)

        normalized["start"] = normalized["start"].astype(float)
        normalized["end"] = normalized["end"].astype(float)
        return normalized

    def _validate_metadata(self, metadata: pd.DataFrame) -> None:
        """Validate the UrbanSound8K schema and its dataset-specific invariants."""
        missing_columns = set(REQUIRED_METADATA_COLUMNS) - set(metadata.columns)

        if missing_columns:
            missing_names = ", ".join(sorted(missing_columns))
            raise DatasetValidationError(
                f"UrbanSound8K metadata is missing required columns: {missing_names}"
            )

        if metadata.empty:
            raise DatasetValidationError("UrbanSound8K metadata contains no samples.")

        expected_num_samples = self._settings.get("expected_num_samples")

        if expected_num_samples is not None and len(metadata) != expected_num_samples:
            raise DatasetValidationError(
                f"Expected {expected_num_samples} UrbanSound8K metadata rows, "
                f"but found {len(metadata)}."
            )

        required_values = metadata.loc[:, REQUIRED_METADATA_COLUMNS]

        if required_values.isnull().any().any():
            columns_with_nulls = required_values.columns[
                required_values.isnull().any()
            ].tolist()
            raise DatasetValidationError(
                "UrbanSound8K metadata contains missing values in columns: "
                f"{columns_with_nulls}"
            )

        if not self._class_names:
            raise DatasetValidationError("No class names are configured for UrbanSound8K.")

        valid_class_ids = set(range(len(self._class_names)))
        observed_class_ids = set(metadata["classID"].unique())
        invalid_class_ids = observed_class_ids - valid_class_ids

        if invalid_class_ids:
            raise DatasetValidationError(
                f"Metadata contains unsupported class IDs: {sorted(invalid_class_ids)}"
            )

        expected_names_by_id = dict(enumerate(self._class_names))
        expected_names = metadata["classID"].map(expected_names_by_id)
        class_mismatch_mask = metadata["class"].astype(str) != expected_names

        if class_mismatch_mask.any():
            examples = metadata.loc[
                class_mismatch_mask, ["slice_file_name", "classID", "class"]
            ].head(10)
            raise DatasetValidationError(
                "Class IDs do not match configured class names. Examples: "
                f"{examples.to_dict(orient='records')}"
            )

        observed_folds = set(metadata["fold"].unique())
        invalid_folds = observed_folds - set(self._available_folds)

        if invalid_folds:
            raise DatasetValidationError(
                f"Metadata contains unsupported folds: {sorted(invalid_folds)}"
            )

        invalid_time_mask = (
            (metadata["start"] < 0)
            | (metadata["end"] <= metadata["start"])
        )

        if invalid_time_mask.any():
            examples = metadata.loc[
                invalid_time_mask, ["slice_file_name", "start", "end"]
            ].head(10)
            raise DatasetValidationError(
                "Metadata contains invalid start/end times. Examples: "
                f"{examples.to_dict(orient='records')}"
            )

        invalid_salience = set(metadata["salience"].unique()) - {1, 2}

        if invalid_salience:
            raise DatasetValidationError(
                f"Metadata contains unsupported salience values: {sorted(invalid_salience)}"
            )

        duplicate_mask = metadata["slice_file_name"].duplicated(keep=False)

        if duplicate_mask.any():
            duplicate_names = metadata.loc[duplicate_mask, "slice_file_name"].head(10)
            raise DatasetValidationError(
                f"Metadata contains duplicate filenames: {duplicate_names.tolist()}"
            )

        self._validate_filenames(metadata)

    def _validate_filenames(self, metadata: pd.DataFrame) -> None:
        """Check filename-derived Freesound and class IDs against CSV metadata."""
        inconsistencies: list[str] = []

        for row in metadata.itertuples(index=False):
            file_name = str(row.slice_file_name)
            file_path = Path(file_name)

            if file_path.name != file_name or file_path.suffix.lower() != ".wav":
                inconsistencies.append(f"unsafe or non-WAV filename '{file_name}'")
                continue

            name_parts = file_path.stem.split("-")

            if len(name_parts) != 4:
                inconsistencies.append(f"unexpected filename format '{file_name}'")
                continue

            try:
                filename_freesound_id = int(name_parts[0])
                filename_class_id = int(name_parts[1])
            except ValueError:
                inconsistencies.append(f"non-numeric IDs in filename '{file_name}'")
                continue

            if filename_freesound_id != int(row.fsID):
                inconsistencies.append(
                    f"fsID mismatch for '{file_name}': filename={filename_freesound_id}, "
                    f"metadata={row.fsID}"
                )

            if filename_class_id != int(row.classID):
                inconsistencies.append(
                    f"classID mismatch for '{file_name}': filename={filename_class_id}, "
                    f"metadata={row.classID}"
                )

            if len(inconsistencies) >= 10:
                break

        if inconsistencies:
            raise DatasetValidationError(
                "UrbanSound8K filename/metadata inconsistencies: "
                + "; ".join(inconsistencies)
            )
