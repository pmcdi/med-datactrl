"""Clinical notes adapter."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from datactrl.adapters.base import resolve_fetch_options
from datactrl.errors import ResourceNotFound
from datactrl.limits import (
    DEFAULT_NOTE_CHARS,
    DEFAULT_NOTE_PREVIEW_CHARS,
    MAX_NOTE_CHARS,
)
from datactrl.types import (
    DescribeResult,
    FetchResult,
    Modality,
    ResourceRef,
    ViewSpec,
    make_resource_id,
)


@dataclass
class _NoteRecord:
    ref: ResourceRef
    path: Path
    format: str


_VIEWS = (
    ViewSpec(
        name="text",
        description="Bounded clinical note text",
        defaults={"max_chars": DEFAULT_NOTE_CHARS},
        max={"max_chars": MAX_NOTE_CHARS},
    ),
)


class NotesAdapter:
    """Clinical note modality adapter."""

    modality = Modality.NOTE
    version = "1"

    def __init__(self) -> None:
        """Create an empty adapter; call ``ingest`` before describe/fetch."""
        self._notes: dict[str, _NoteRecord] = {}

    def views(self) -> Sequence[ViewSpec]:
        """Return supported note fetch views."""
        return _VIEWS

    def ingest(self, root: Path, paths: Sequence[Path]) -> list[ResourceRef]:
        """Index note files under ``root`` as text resources."""
        self._notes.clear()
        refs: list[ResourceRef] = []
        for path in sorted(paths):
            rel = _rel(path, root)
            note_id = make_resource_id(Modality.NOTE, _path_key(rel))
            fmt = path.suffix.lower().lstrip(".") or "txt"
            # Handle double extensions like .pdf.pdf
            if path.name.lower().endswith(".pdf.pdf"):
                fmt = "pdf"
            ref = ResourceRef(
                id=note_id,
                modality=Modality.NOTE,
                path=rel,
                label=path.name,
                meta={"format": fmt},
            )
            self._notes[note_id] = _NoteRecord(ref=ref, path=path, format=fmt)
            refs.append(ref)
        return refs

    def describe(self, resource_id: str) -> DescribeResult:
        """Return format, char count, and a short preview for one note."""
        record = self._require(resource_id)
        text = _read_text(record.path, record.format)
        preview = text[:DEFAULT_NOTE_PREVIEW_CHARS]
        if len(text) > DEFAULT_NOTE_PREVIEW_CHARS:
            preview += "..."
        return DescribeResult(
            id=resource_id,
            modality=Modality.NOTE,
            label=record.ref.label,
            path=record.ref.path,
            views=[spec.name for spec in _VIEWS],
            view_specs=list(_VIEWS),
            summary={
                "format": record.format,
                "char_count": len(text),
            },
            preview=preview,
        )

    def fetch(
        self, resource_id: str, view: str | None = None, **opts: Any
    ) -> FetchResult:
        """Fetch bounded note text."""
        resolved_view, merged = resolve_fetch_options(
            _VIEWS, view, opts, default_view="text"
        )
        record = self._require(resource_id)
        max_chars = int(merged.get("max_chars", DEFAULT_NOTE_CHARS))
        text = _read_text(record.path, record.format)
        truncated = len(text) > max_chars
        return FetchResult(
            id=resource_id,
            modality=Modality.NOTE,
            view=resolved_view,
            data={
                "format": record.format,
                "text": text[:max_chars],
                "char_count": len(text),
            },
            bounds={
                "max_chars": max_chars,
                "truncated": truncated,
            },
            provenance={
                "path": record.ref.path,
                "adapter": "notes",
                "adapter_version": self.version,
            },
        )

    def _require(self, resource_id: str) -> _NoteRecord:
        """Look up an indexed note or raise ResourceNotFound."""
        try:
            return self._notes[resource_id]
        except KeyError as exc:
            raise ResourceNotFound(f"Unknown note: {resource_id!r}") from exc


def _read_text(path: Path, fmt: str) -> str:
    """Read full note text from plain text or PDF."""
    if fmt == "pdf":
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages).strip()
    return path.read_text(encoding="utf-8", errors="replace")


def _rel(path: Path, root: Path) -> str:
    """Return ``path`` relative to ``root``, or its string form if outside."""
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _path_key(relative_path: str) -> str:
    """Hash a relative path into a stable note resource key."""
    return hashlib.sha1(relative_path.encode("utf-8")).hexdigest()[:12]
