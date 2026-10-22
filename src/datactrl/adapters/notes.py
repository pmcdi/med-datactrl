"""Clinical notes adapter."""

from __future__ import annotations

import base64
import hashlib
import io
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pypdf import PdfReader, PdfWriter

from datactrl.adapters.base import resolve_fetch_options
from datactrl.errors import (
    InvalidFetchOptions,
    LimitExceeded,
    ResourceNotFound,
    UnknownView,
)
from datactrl.limits import (
    DEFAULT_NOTE_CHARS,
    DEFAULT_NOTE_PREVIEW_CHARS,
    DEFAULT_NOTE_SEARCH_HITS,
    MAX_NOTE_CHARS,
    MAX_NOTE_SEARCH_HITS,
    MAX_NOTE_SECTIONS,
    MAX_SPAN_CHARS,
    MAX_SPAN_CONTEXT_CHARS,
    NOTE_META_SNIFF_CHARS,
)
from datactrl.types import (
    DescribeResult,
    FetchResult,
    Modality,
    ResourceRef,
    ViewSpec,
    make_resource_id,
)
from datactrl.utils.notes_index import (
    BM25Index,
    Section,
    build_chunks,
    guess_note_type,
    resolve_note_date,
    section_at,
    sectionize,
    snippet,
    text_quality,
)


@dataclass
class _NoteRecord:
    ref: ResourceRef
    path: Path
    format: str
    _text: str | None = field(default=None, repr=False)
    _sections: list[Section] | None = field(default=None, repr=False)


_VIEWS = (
    ViewSpec(
        name="text",
        description="Bounded clinical note text",
        defaults={"max_chars": DEFAULT_NOTE_CHARS},
        max={"max_chars": MAX_NOTE_CHARS},
    ),
    ViewSpec(
        name="sections",
        description="Section headers with character offsets (no body text)",
    ),
    ViewSpec(
        name="span",
        description=(
            "Exact text between character offsets start/end, with optional "
            "surrounding context_chars"
        ),
        defaults={"start": 0, "end": 0, "context_chars": 0},
        max={"context_chars": MAX_SPAN_CONTEXT_CHARS},
    ),
    ViewSpec(
        name="page",
        description="PDF notes only: one page as a standalone PDF plus its text",
        defaults={"page_index": 0},
    ),
)


class NotesAdapter:
    """Clinical note modality adapter."""

    modality = Modality.NOTE
    version = "2"

    def __init__(self) -> None:
        """Create an empty adapter; call ``ingest`` before describe/fetch."""
        self._notes: dict[str, _NoteRecord] = {}
        self._bm25: BM25Index | None = None

    def views(self) -> Sequence[ViewSpec]:
        """Return supported note fetch views."""
        return _VIEWS

    # -----------------------------------------------------------------------
    # Ingest
    # -----------------------------------------------------------------------

    def ingest(self, root: Path, paths: Sequence[Path]) -> list[ResourceRef]:
        """Index note files under ``root`` with date/type/quality metadata."""
        self._notes.clear()
        self._bm25 = None
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
                meta={"format": fmt, **_sniff_meta(path, fmt)},
            )
            self._notes[note_id] = _NoteRecord(ref=ref, path=path, format=fmt)
            refs.append(ref)
        return refs

    # -----------------------------------------------------------------------
    # Describe, fetch, and search
    # -----------------------------------------------------------------------

    def describe(self, resource_id: str) -> DescribeResult:
        """Return format, char count, date/type metadata and a short preview."""
        record = self._require(resource_id)
        text = self._full_text(record)
        preview = text[:DEFAULT_NOTE_PREVIEW_CHARS]
        if len(text) > DEFAULT_NOTE_PREVIEW_CHARS:
            preview += "..."
        summary: dict[str, Any] = {
            "format": record.format,
            "char_count": len(text),
            "sections": [s.name for s in self._sections_of(record)][:MAX_NOTE_SECTIONS],
        }
        for key in ("note_date", "note_date_confidence", "note_type", "text_quality"):
            if key in record.ref.meta:
                summary[key] = record.ref.meta[key]
        if record.format == "pdf":
            summary["pages"] = record.ref.meta.get("pages")
            summary["page_quality"] = record.ref.meta.get("page_quality")
        views = [
            spec.name
            for spec in _VIEWS
            if spec.name != "page" or record.format == "pdf"
        ]
        return DescribeResult(
            id=resource_id,
            modality=Modality.NOTE,
            label=record.ref.label,
            path=record.ref.path,
            views=views,
            view_specs=[spec for spec in _VIEWS if spec.name in views],
            summary=summary,
            preview=preview,
        )

    def fetch(
        self, resource_id: str, view: str | None = None, **opts: Any
    ) -> FetchResult:
        """Fetch a bounded note view (text, sections, span, page)."""
        resolved_view, merged = resolve_fetch_options(
            _VIEWS, view, opts, default_view="text"
        )
        record = self._require(resource_id)
        if resolved_view == "sections":
            data = self._fetch_sections(record)
            bounds: dict[str, Any] = {"max_sections": MAX_NOTE_SECTIONS}
        elif resolved_view == "span":
            data, bounds = self._fetch_span(record, merged)
        elif resolved_view == "page":
            data, bounds = self._fetch_page(record, merged)
        else:
            max_chars = int(merged.get("max_chars", DEFAULT_NOTE_CHARS))
            raw = _read_text(record.path, record.format, max_chars=max_chars + 1)
            truncated = len(raw) > max_chars
            text = raw[:max_chars]
            data = {
                "format": record.format,
                "text": text,
                "char_count": len(text) if not truncated else max_chars,
            }
            bounds = {"max_chars": max_chars, "truncated": truncated}
        return FetchResult(
            id=resource_id,
            modality=Modality.NOTE,
            view=resolved_view,
            data=data,
            bounds=bounds,
            provenance={
                "path": record.ref.path,
                "adapter": "notes",
                "adapter_version": self.version,
            },
        )

    def search(
        self, query: str, limit: int = DEFAULT_NOTE_SEARCH_HITS
    ) -> dict[str, Any]:
        """Lexical BM25 search over note sections; returns cited snippets."""
        if limit < 1:
            raise InvalidFetchOptions(f"limit must be >= 1, got {limit}")
        if limit > MAX_NOTE_SEARCH_HITS:
            raise LimitExceeded(
                f"limit={limit} exceeds max {MAX_NOTE_SEARCH_HITS} for notes_search; "
                f"retry with limit<={MAX_NOTE_SEARCH_HITS}"
            )
        index = self._index()
        hits = []
        for chunk, score, terms in index.search(query, limit):
            record = self._notes[chunk.note_id]
            text = self._full_text(record)
            snip = snippet(text, chunk.start, chunk.end, terms)
            hits.append(
                {
                    "note_id": chunk.note_id,
                    "label": record.ref.label,
                    "note_date": record.ref.meta.get("note_date"),
                    "note_type": record.ref.meta.get("note_type"),
                    "section": chunk.section,
                    "start": chunk.start,
                    "end": chunk.end,
                    "score": score,
                    "matched_terms": terms,
                    "snippet": snip["text"],
                    "snippet_start": snip["start"],
                    "snippet_end": snip["end"],
                }
            )
        return {
            "query": query,
            "hits": hits,
            "bounds": {"limit": limit, "chunks_indexed": len(index.chunks)},
        }

    def total_chars(self) -> int:
        """Total extracted characters across all notes."""
        return sum(len(self._full_text(record)) for record in self._notes.values())

    # --- internals ---

    def _fetch_sections(self, record: _NoteRecord) -> dict[str, Any]:
        sections = self._sections_of(record)
        return {
            "format": record.format,
            "char_count": len(self._full_text(record)),
            "note_date": record.ref.meta.get("note_date"),
            "note_type": record.ref.meta.get("note_type"),
            "sections": [s.to_dict() for s in sections[:MAX_NOTE_SECTIONS]],
            "truncated": len(sections) > MAX_NOTE_SECTIONS,
        }

    def _fetch_span(
        self, record: _NoteRecord, opts: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        text = self._full_text(record)
        start, end = int(opts.get("start") or 0), int(opts.get("end") or 0)
        context = int(opts.get("context_chars") or 0)
        if start < 0 or end < 0:
            raise InvalidFetchOptions("start/end must be non-negative")
        if end <= start:
            raise InvalidFetchOptions(
                f"end ({end}) must be greater than start ({start})"
            )
        if start >= len(text):
            raise InvalidFetchOptions(
                f"start={start} is beyond note length {len(text)}"
            )
        if end - start > MAX_SPAN_CHARS:
            raise LimitExceeded(
                f"span length {end - start} exceeds max {MAX_SPAN_CHARS}; "
                f"retry with end-start<={MAX_SPAN_CHARS}"
            )
        end = min(end, len(text))
        section = section_at(self._sections_of(record), start)
        data = {
            "format": record.format,
            "text": text[start:end],
            "start": start,
            "end": end,
            "context_before": text[max(0, start - context) : start],
            "context_after": text[end : end + context],
            "section": section.name if section else None,
            "note_char_count": len(text),
        }
        return data, {"max_span_chars": MAX_SPAN_CHARS, "context_chars": context}

    def _fetch_page(
        self, record: _NoteRecord, opts: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if record.format != "pdf":
            raise UnknownView("view 'page' is only available for PDF notes")
        reader = PdfReader(str(record.path))
        page_index = int(opts.get("page_index") or 0)
        if page_index < 0 or page_index >= len(reader.pages):
            raise InvalidFetchOptions(
                f"page_index={page_index} out of range (0..{len(reader.pages) - 1})"
            )
        writer = PdfWriter()
        writer.add_page(reader.pages[page_index])
        buffer = io.BytesIO()
        writer.write(buffer)
        page_text = reader.pages[page_index].extract_text() or ""
        data = {
            "format": "pdf",
            "mime_type": "application/pdf",
            "pdf_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
            "page_index": page_index,
            "n_pages": len(reader.pages),
            "text": page_text,
            "text_quality": text_quality(page_text),
            "page_offset": self._page_offsets(record)[page_index],
        }
        return data, {"pages": 1}

    def _full_text(self, record: _NoteRecord) -> str:
        if record._text is None:
            record._text = _read_text(
                record.path, record.format, max_chars=MAX_NOTE_CHARS
            )
        return record._text

    def _sections_of(self, record: _NoteRecord) -> list[Section]:
        if record._sections is None:
            record._sections = sectionize(self._full_text(record))
        return record._sections

    def _page_offsets(self, record: _NoteRecord) -> list[int]:
        """Character offset of each PDF page inside the full extracted text."""
        pages = _pdf_page_texts(record.path)
        joined = "\n\n".join(pages)
        lead = len(joined) - len(joined.lstrip())
        offsets, pos = [], 0
        for page in pages:
            offsets.append(max(0, pos - lead))
            pos += len(page) + 2
        return offsets

    def _index(self) -> BM25Index:
        if self._bm25 is None:
            chunks = []
            for note_id, record in self._notes.items():
                chunks.extend(
                    build_chunks(
                        note_id, self._full_text(record), self._sections_of(record)
                    )
                )
            self._bm25 = BM25Index(chunks)
        return self._bm25

    def _require(self, resource_id: str) -> _NoteRecord:
        """Look up an indexed note or raise ResourceNotFound."""
        try:
            return self._notes[resource_id]
        except KeyError as exc:
            raise ResourceNotFound(f"Unknown note: {resource_id!r}") from exc


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------


def _sniff_meta(path: Path, fmt: str) -> dict[str, Any]:
    """Cheap discovery-time metadata: date, note type, text quality."""
    try:
        if fmt == "pdf":
            pages = _pdf_page_texts(path, max_pages=5)
            head = "\n\n".join(pages)[:NOTE_META_SNIFF_CHARS]
            page_quality = [text_quality(page)["level"] for page in pages]
        else:
            head = _read_text(path, fmt, max_chars=NOTE_META_SNIFF_CHARS)
            page_quality = None
    except Exception:  # noqa: BLE001 - unreadable notes stay indexed without meta
        return {"text_quality": "unreadable"}
    dated = resolve_note_date(head, path.name)
    meta: dict[str, Any] = {
        "note_date": dated["date"],
        "note_date_source": dated["date_source"],
        "note_date_confidence": dated["date_confidence"],
        "note_type": guess_note_type(head, path.name),
        "text_quality": text_quality(head)["level"],
    }
    if dated["date_flags"]:
        meta["note_date_flags"] = dated["date_flags"]
    if page_quality is not None:
        meta["pages"] = len(page_quality)
        meta["page_quality"] = page_quality
        rank = {"good": 0, "mixed": 1, "poor": 2, "empty": 3}
        meta["text_quality"] = max(
            page_quality, key=lambda lvl: rank.get(lvl, 3), default="empty"
        )
    return meta


def _pdf_page_texts(path: Path, max_pages: int | None = None) -> list[str]:
    reader = PdfReader(str(path))
    pages = reader.pages if max_pages is None else reader.pages[:max_pages]
    return [page.extract_text() or "" for page in pages]


def _read_text(path: Path, fmt: str, max_chars: int | None = None) -> str:
    """Read note text from plain text or PDF, optionally stopping early."""
    if fmt == "pdf":
        reader = PdfReader(str(path))
        chunks: list[str] = []
        total = 0
        for page in reader.pages:
            chunk = page.extract_text() or ""
            chunks.append(chunk)
            total += len(chunk) + 2
            if max_chars is not None and total >= max_chars:
                break
        text = "\n\n".join(chunks).strip()
        return text if max_chars is None else text[:max_chars]
    if max_chars is None:
        return path.read_text(encoding="utf-8", errors="replace")
    with path.open(encoding="utf-8", errors="replace") as handle:
        return handle.read(max_chars)


def _rel(path: Path, root: Path) -> str:
    """Return ``path`` relative to ``root``, or its string form if outside."""
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _path_key(relative_path: str) -> str:
    """Hash a relative path into a stable note resource key."""
    return hashlib.sha1(relative_path.encode("utf-8")).hexdigest()[:12]
