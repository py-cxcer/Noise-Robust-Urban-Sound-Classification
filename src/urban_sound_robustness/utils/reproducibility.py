"""Reproducibility helpers for Python, NumPy, PyTorch, CUDA, and DataLoaders."""

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """
    Seed all random-number generators used by the project.

    Parameters
    ----------
    seed : int
        Non-negative experiment seed.
    deterministic : bool
        Request deterministic PyTorch algorithms where available. This improves
        repeatability but can reduce training speed.

    Returns
    -------
    None

    Notes
    -----
    ``PYTHONHASHSEED`` affects newly started processes. Setting it here records the
    intended value, while ``random.seed`` controls the current process.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("The random seed must be a non-negative integer.")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def seed_data_loader_worker(worker_id: int) -> None:
    """
    Seed NumPy and Python state inside one DataLoader worker.

    Parameters
    ----------
    worker_id : int
        Worker ID supplied by PyTorch. PyTorch already incorporates this value in
        ``torch.initial_seed()``, so it is not added a second time.

    Returns
    -------
    None
    """
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_data_loader_generator(seed: int) -> torch.Generator:
    """
    Create a seeded generator for deterministic DataLoader sampling.

    Parameters
    ----------
    seed : int
        Non-negative experiment seed.

    Returns
    -------
    torch.Generator
        Generator suitable for the DataLoader ``generator`` argument.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("The random seed must be a non-negative integer.")

    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator

