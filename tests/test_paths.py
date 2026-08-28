"""Tests for operating-system-independent project path handling."""

from pathlib import Path

from urban_sound_robustness.utils.paths import (
    find_project_root,
    resolve_path_settings,
    resolve_project_path,
)


def test_find_project_root_from_nested_directory(tmp_path: Path) -> None:
    """Root discovery should work from scripts, tests, or notebook subfolders."""
    (tmp_path / "pyproject.toml").touch()
    (tmp_path / "configs").mkdir()
    nested_directory = tmp_path / "one" / "two"
    nested_directory.mkdir(parents=True)

    assert find_project_root(nested_directory) == tmp_path.resolve()


def test_resolve_relative_project_path(tmp_path: Path) -> None:
    """Relative values should be anchored at the project rather than the shell."""
    resolved_path = resolve_project_path("data/raw", tmp_path)

    assert resolved_path == (tmp_path / "data" / "raw").resolve()


def test_resolve_path_settings_preserves_names(tmp_path: Path) -> None:
    """All configured path names should be retained after resolution."""
    path_settings = {"data": "data", "logs": "outputs/logs"}

    resolved_paths = resolve_path_settings(path_settings, tmp_path)

    assert resolved_paths["data"] == (tmp_path / "data").resolve()
    assert resolved_paths["logs"] == (tmp_path / "outputs" / "logs").resolve()

