"""Dataset contracts, adapters, and inspection utilities."""

from urban_sound_robustness.datasets.factory import create_dataset_adapter
from urban_sound_robustness.datasets.inspection import (
    DatasetInspectionResult,
    inspect_dataset,
    save_inspection_result,
)
from urban_sound_robustness.datasets.records import AudioSampleRecord
from urban_sound_robustness.datasets.torch_dataset import (
    AudioDatasetItem,
    PreprocessedAudioDataset,
    create_preprocessed_dataset,
)
from urban_sound_robustness.datasets.urbansound8k import (
    DatasetNotFoundError,
    DatasetValidationError,
    UrbanSound8KAdapter,
)

__all__ = [
    "AudioSampleRecord",
    "AudioDatasetItem",
    "DatasetInspectionResult",
    "DatasetNotFoundError",
    "DatasetValidationError",
    "PreprocessedAudioDataset",
    "UrbanSound8KAdapter",
    "create_dataset_adapter",
    "create_preprocessed_dataset",
    "inspect_dataset",
    "save_inspection_result",
]
