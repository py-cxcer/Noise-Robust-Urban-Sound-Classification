"""Path discovery and resolution helpers for local and cloud environments."""

from pathlib import Path
from typing import Mapping


class ProjectRootNotFoundError(FileNotFoundError):
    """Raised when the repository root cannot be located from a starting path."""


def find_project_root(start_path: str | Path | None = None) -> Path:
    """
    Find the repository root by looking for package metadata and configurations.

    Parameters
    ----------
    start_path : str or Path or None
        File or directory from which to start. Defaults to the working directory.

    Returns
    -------
    Path
        Absolute repository root.

    Raises
    ------
    ProjectRootNotFoundError
        If no parent contains both ``pyproject.toml`` and ``configs/``.
    """
    candidate = Path.cwd() if start_path is None else Path(start_path)
    candidate = candidate.expanduser().resolve()

    if candidate.is_file():
        candidate = candidate.parent

    for directory in (candidate, *candidate.parents):
        has_package_metadata = (directory / "pyproject.toml").is_file()
        has_configurations = (directory / "configs").is_dir()

        if has_package_metadata and has_configurations:
            return directory

    raise ProjectRootNotFoundError(
        f"Could not find a project root above '{candidate}'. Expected both "
        "pyproject.toml and a configs directory."
    )


def resolve_project_path(path_value: str | Path, project_root: str | Path) -> Path:
    """
    Resolve a configured path without assuming a Windows-specific directory.

    Parameters
    ----------
    path_value : str or Path
        Relative project path or an absolute path supplied by the user.
    project_root : str or Path
        Repository root used for relative paths.

    Returns
    -------
    Path
        Absolute normalized path. The path does not need to exist yet.
    """
    configured_path = Path(path_value).expanduser()

    if configured_path.is_absolute():
        return configured_path.resolve()

    root = Path(project_root).expanduser().resolve()
    return (root / configured_path).resolve()


def resolve_path_settings(
    path_settings: Mapping[str, str | Path],
    project_root: str | Path,
) -> dict[str, Path]:
    """
    Resolve every entry from the ``paths`` configuration section.

    Parameters
    ----------
    path_settings : Mapping[str, str or Path]
        Named paths read from configuration.
    project_root : str or Path
        Repository root used to resolve relative entries.

    Returns
    -------
    dict[str, Path]
        Mapping with the same keys and absolute ``Path`` values.
    """
    resolved_paths: dict[str, Path] = {}

    for name, path_value in path_settings.items():
        resolved_paths[name] = resolve_project_path(path_value, project_root)

    return resolved_paths
