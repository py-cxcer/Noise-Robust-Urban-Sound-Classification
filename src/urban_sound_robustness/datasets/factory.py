"""Construct dataset adapters without spreading dataset-specific conditions."""

from pathlib import Path
from typing import Any, Mapping

from urban_sound_robustness.datasets.base import AudioDatasetAdapter
from urban_sound_robustness.datasets.urbansound8k import UrbanSound8KAdapter


def create_dataset_adapter(
    dataset_settings: Mapping[str, Any],
    project_root: str | Path,
) -> AudioDatasetAdapter:
    """
    Create the adapter selected by dataset configuration.

    Parameters
    ----------
    dataset_settings : Mapping[str, Any]
        Resolved dataset configuration section.
    project_root : str or Path
        Repository root used for relative dataset paths.

    Returns
    -------
    AudioDatasetAdapter
        Configured dataset adapter.

    Raises
    ------
    ValueError
        If the adapter name is unsupported.
    """
    adapter_name = str(dataset_settings.get("adapter", "")).strip().lower()

    if adapter_name == "urbansound8k":
        return UrbanSound8KAdapter(dataset_settings, project_root)

    raise ValueError(
        f"Unsupported dataset adapter '{adapter_name}'. "
        "Add a new adapter module and register it in datasets/factory.py."
    )

