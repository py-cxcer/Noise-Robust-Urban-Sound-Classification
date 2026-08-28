"""Select and describe the PyTorch execution device."""

from typing import Any

import torch


def select_device(preference: str = "auto") -> torch.device:
    """
    Select a CPU or CUDA device from a readable configuration value.

    Parameters
    ----------
    preference : str
        ``auto`` selects CUDA when available and otherwise falls back to CPU.
        ``cpu``, ``cuda``, and indexed values such as ``cuda:0`` are also valid.

    Returns
    -------
    torch.device
        Validated device for model and tensor placement.

    Raises
    ------
    ValueError
        If the preference is unsupported or a CUDA index is invalid.
    RuntimeError
        If CUDA is explicitly requested but unavailable.
    """
    normalized_preference = preference.strip().lower()

    if normalized_preference == "auto":
        normalized_preference = "cuda" if torch.cuda.is_available() else "cpu"

    if normalized_preference == "cpu":
        return torch.device("cpu")

    if normalized_preference == "cuda":
        normalized_preference = "cuda:0"

    if not normalized_preference.startswith("cuda:"):
        raise ValueError(
            f"Unsupported device preference '{preference}'. Use auto, cpu, cuda, or cuda:N."
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but PyTorch cannot access a CUDA device. "
            "Use device=auto for CPU fallback or install a CUDA-enabled PyTorch build."
        )

    try:
        device_index = int(normalized_preference.split(":", maxsplit=1)[1])
    except ValueError as error:
        raise ValueError(f"Invalid CUDA device preference: '{preference}'.") from error

    if device_index < 0 or device_index >= torch.cuda.device_count():
        raise ValueError(
            f"CUDA device index {device_index} is unavailable; "
            f"detected {torch.cuda.device_count()} CUDA device(s)."
        )

    return torch.device(f"cuda:{device_index}")


def describe_device(device: torch.device) -> dict[str, Any]:
    """
    Return serializable hardware information for experiment records.

    Parameters
    ----------
    device : torch.device
        Selected execution device.

    Returns
    -------
    dict[str, Any]
        Device type and, for CUDA, name, capability, and memory capacity.
    """
    description: dict[str, Any] = {"type": device.type}

    if device.type != "cuda":
        return description

    device_index = 0 if device.index is None else device.index
    properties = torch.cuda.get_device_properties(device_index)
    description.update(
        {
            "index": device_index,
            "name": properties.name,
            "compute_capability": list(torch.cuda.get_device_capability(device_index)),
            "total_memory_bytes": properties.total_memory,
        }
    )
    return description

