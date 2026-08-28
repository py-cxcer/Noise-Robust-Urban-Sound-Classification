"""Common records exchanged between dataset adapters and reusable pipeline code."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class AudioSampleRecord:
    """
    Describe one labeled audio clip without loading its waveform.

    Parameters
    ----------
    sample_id : str
        Dataset-unique identifier, normally the original filename.
    audio_path : Path
        Resolved location of the audio file.
    label : int
        Numeric class ID expected by classification models.
    class_name : str
        Human-readable class name corresponding to ``label``.
    fold : int or None
        Official dataset fold when one exists.
    dataset_name : str
        Name of the adapter that produced the record.
    metadata : Mapping[str, Any]
        Original dataset metadata retained for analysis and error inspection.
    """

    sample_id: str
    audio_path: Path
    label: int
    class_name: str
    fold: int | None
    dataset_name: str
    metadata: Mapping[str, Any]

