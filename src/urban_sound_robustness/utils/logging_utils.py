"""Consistent console and file logging for project scripts and experiments."""

import logging
from pathlib import Path


PROJECT_LOGGER_NAME = "urban_sound_robustness"


def configure_logging(
    log_file: str | Path | None = None,
    level: str = "INFO",
) -> logging.Logger:
    """
    Configure the project logger without adding duplicate handlers.

    Parameters
    ----------
    log_file : str or Path or None
        Optional file that should receive the same records as the console.
    level : str
        Standard logging level name such as ``INFO`` or ``DEBUG``.

    Returns
    -------
    logging.Logger
        Configured package-level logger.

    Raises
    ------
    ValueError
        If ``level`` is not recognized.
    """
    normalized_level = level.strip().upper()
    numeric_level = logging.getLevelNamesMapping().get(normalized_level)

    if numeric_level is None:
        raise ValueError(f"Unsupported logging level: '{level}'.")

    logger = logging.getLogger(PROJECT_LOGGER_NAME)
    logger.setLevel(numeric_level)
    logger.propagate = False

    # Reconfiguration is common in tests and notebooks. Removing only this
    # package logger's handlers prevents duplicated messages.
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file is not None:
        resolved_log_file = Path(log_file).expanduser().resolve()
        resolved_log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(resolved_log_file, encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

