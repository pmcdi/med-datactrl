from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from datactrl.adapters.discovery import NOTE_EXTENSIONS, iter_patient_files

PREVIEW_LEN = 200


@dataclass(frozen=True)
class ClinicalNoteRecord:
    note_id: str
    path: Path
    filename: str
    format: str


class ClinicalAdapter:
    """Discover and read clinical notes for one patient directory."""

    def __init__(self, patient_dir: Path, notes: dict[str, ClinicalNoteRecord]):
        self.patient_dir = patient_dir
        self.notes = notes

    @classmethod
    def from_patient_directory(cls, patient_dir: Path) -> ClinicalAdapter | None:
        """Return an adapter when note files are present, otherwise ``None``."""

        notes = _discover_notes(patient_dir)
        if not notes:
            return None
        return cls(patient_dir, notes)

    def list_notes(self) -> list[dict[str, Any]]:
        """Return one summary dict per indexed clinical note."""

        return [_note_summary(self, note_id) for note_id in sorted(self.notes)]

    def get_note(self, note_id: str) -> str | dict[str, Any]:
        """Return note text or parsed JSON for one note ID."""

        record = self._require_note(note_id)
        if record.format == "json":
            return json.loads(record.path.read_text())
        return _read_note_text(record.path, record.format)

    def _require_note(self, note_id: str) -> ClinicalNoteRecord:
        try:
            return self.notes[note_id]
        except KeyError:
            raise ValueError(f"Unknown clinical note: {note_id!r}") from None


def _discover_notes(patient_dir: Path) -> dict[str, ClinicalNoteRecord]:
    notes: dict[str, ClinicalNoteRecord] = {}
    for path in sorted(iter_patient_files(patient_dir)):
        suffix = path.suffix.lower()
        if suffix not in NOTE_EXTENSIONS:
            continue
        notes[path.stem] = ClinicalNoteRecord(
            note_id=path.stem,
            path=path,
            filename=path.name,
            format=suffix.lstrip("."),
        )
    return notes


def _read_note_text(path: Path, fmt: str) -> str:
    if fmt == "pdf":
        reader = PdfReader(path)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages).strip()
    return path.read_text()


def _note_summary(adapter: ClinicalAdapter, note_id: str) -> dict[str, Any]:
    record = adapter.notes[note_id]
    text = _read_note_text(record.path, record.format)
    preview = text[:PREVIEW_LEN]
    if len(text) > PREVIEW_LEN:
        preview += "..."

    return {
        "note_id": note_id,
        "filename": record.filename,
        "path": str(record.path.relative_to(adapter.patient_dir)),
        "format": record.format,
        "char_count": len(text),
        "preview": preview,
    }
