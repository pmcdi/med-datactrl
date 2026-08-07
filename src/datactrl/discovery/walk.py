"""Filesystem walk helpers."""

from __future__ import annotations

from pathlib import Path


def iter_files(root: Path) -> list[Path]:
    """Return visible files under ``root``, excluding hidden path parts."""
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"Root is not a directory: {root}")

    files: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if _is_hidden(path, root):
            continue
        files.append(path)
    return sorted(files)


def _is_hidden(path: Path, root: Path) -> bool:
    """Return True when any path component under ``root`` starts with ``.``."""
    relative = path.relative_to(root)
    return any(part.startswith(".") for part in relative.parts)
