"""Small structural interface shared by single-label audio dataset adapters."""

from typing import Protocol, Sequence

from urban_sound_robustness.datasets.records import AudioSampleRecord


class AudioDatasetAdapter(Protocol):
    """Behavior required by generic inspection and future Dataset code."""

    @property
    def dataset_name(self) -> str:
        """Return the stable dataset identifier."""

    @property
    def class_names(self) -> Sequence[str]:
        """Return class names ordered by numeric label."""

    def load_records(self) -> list[AudioSampleRecord]:
        """Validate dataset metadata and return all sample records."""

    def records_for_split(self, split_name: str) -> list[AudioSampleRecord]:
        """Return records from folds assigned to one configured split."""

