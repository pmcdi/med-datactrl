"""Compile step: notes -> grounded fact ledger (cached per patient, per note hash).

Each note is extracted independently (a parallel map), so large charts scale by
note count and unchanged notes are never re-extracted.
"""

from __future__ import annotations

import base64
import hashlib
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import src.retrieval.case_store  # noqa: F401  (puts datactrl on sys.path)
from datactrl.limits import MAX_NOTE_CHARS
from datactrl.notes_index import parse_date

from src.agents.notes import llm
from src.agents.notes.grounding import locate, value_supported
from src.agents.notes.ledger import LEDGER_VERSION, Ledger
from src.agents.notes.schema import (
    ASSERTIONS,
    CATEGORIES,
    SOURCE_TYPES,
    ExtractionResult,
    Fact,
    make_fact_id,
    normalize_attribute,
    normalize_choice,
    normalize_value,
)

EXTRACTOR_VERSION = "2"
LEDGER_DIR = Path(__file__).resolve().parents[3] / "workspace" / "notes_ledger"
EXTRACT_CHUNK_CHARS = 24_000
MAX_WORKERS = 8
MAX_OCR_PAGES = 5

_MEMORY: dict[str, Ledger] = {}
_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


@dataclass
class NoteDoc:
    """A text document the extractor reads: a native note or an OCR transcript."""

    doc_id: str
    note_id: str
    label: str
    note_date: str | None
    date_confidence: str
    note_type: str
    text: str
    sections: list[dict[str, Any]] = field(default_factory=list)
    derived: str = "native"
    page_index: int | None = None

    @property
    def hash(self) -> str:
        """Content hash that invalidates cached facts when text or extractor changes."""
        from src.core.gemini_client import NOTES_EXTRACT_MODEL

        raw = f"{EXTRACTOR_VERSION}|{NOTES_EXTRACT_MODEL}|{self.doc_id}|{self.text}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()


_EXTRACT_PROMPT = """You are a clinical information extraction system. Extract every clinically relevant fact from ONE clinical document.

Document: {label} (id {doc_id})
Note date: {note_date} (confidence: {date_confidence}); note type guess: {note_type}{ocr_hint}
Sections: {sections}
{focus}
Rules:
- One fact per atomic finding (diagnosis, stage, tumor size/location/grade, lab value, dose, fractions, procedure, performance status, outcome, comorbidity, medication, allergy, negated finding, etc.).
- attribute: use tumor.* only for the primary tumor; use recurrence.* for recurrences and metastasis.site for distant disease.
- value: copy the value as written (do not convert units).
- quote: copy VERBATIM from the document text, the shortest span (<= 200 chars) that proves the fact. Never paraphrase in quote.
- assertion: negated for "no X", "denies X", "not identified"; possible for "?", "vs", "suspicious for", "maybe"; historical for past events; family for family history; planned for plans/recommendations; otherwise present.
- source_type: pathology (final pathology report), radiology_final (final/attending/corrected read), radiology_prelim (preliminary or voice-recognition read), operative, physician, nursing_phone, research_form. Judge from the document and the section (a telephone preliminary pathology result inside an operative note is operative).
- event_date: only when the text states when the event happened. Use relative_time for relative expressions (e.g. "~22 mo post-op").
- When the document itself records uncertainty or disagreement (e.g. "72 y/o (chart says 71??)"), extract each stated value as its own fact.
- Do not infer anything that is not written.

DOCUMENT TEXT:
<<<
{text}
>>>"""

_OCR_PROMPT = """Transcribe this scanned clinical document page verbatim as plain text.
Preserve line breaks and the original wording. Mark unreadable words as [illegible].
Do not add commentary, interpretation or formatting."""


def ledger_path(patient_id: str) -> Path:
    """Disk cache location for one patient's ledger."""
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", patient_id)
    return LEDGER_DIR / f"{safe}.json"


def load_native_docs(case) -> list[NoteDoc]:
    """Read every discovered note (bounded) with its sections and metadata."""
    docs = []
    for ref in case.report.notes:
        text = case.fetch(ref.id, view="text", max_chars=MAX_NOTE_CHARS)["data"]["text"]
        sections = case.fetch(ref.id, view="sections")["data"]["sections"]
        docs.append(
            NoteDoc(
                doc_id=ref.id,
                note_id=ref.id,
                label=ref.label,
                note_date=ref.meta.get("note_date"),
                date_confidence=ref.meta.get("note_date_confidence", "none"),
                note_type=ref.meta.get("note_type", "unknown"),
                text=text,
                sections=sections,
            )
        )
    return docs


def doc_text(ledger: Ledger, doc_id: str, case) -> str | None:
    """Text for a ledger document id (native note via datactrl, or stored OCR)."""
    doc = ledger.docs.get(doc_id)
    if doc and doc.get("derived") == "ocr":
        return doc.get("text")
    try:
        return case.fetch(doc_id, view="text", max_chars=MAX_NOTE_CHARS)["data"]["text"]
    except Exception:  # noqa: BLE001 - unknown ids are reported as missing
        return None


def _section_for(sections: list[dict[str, Any]], offset: int) -> str | None:
    for section in sections:
        if section["start"] <= offset < section["end"]:
            return section["name"]
    return None


def _chunks(doc: NoteDoc) -> list[tuple[int, str]]:
    """Split long docs on section boundaries into <= EXTRACT_CHUNK_CHARS pieces."""
    text = doc.text
    bounds = sorted({s["start"] for s in doc.sections})
    pieces: list[tuple[int, str]] = []
    start = 0
    while start < len(text):
        limit = start + EXTRACT_CHUNK_CHARS
        if limit >= len(text):
            pieces.append((start, text[start:]))
            break
        section_cuts = [b for b in bounds if start < b <= limit]
        if section_cuts:
            cut = max(section_cuts)
        else:
            newline = text.rfind("\n", start, limit)
            cut = newline if newline > start else limit
        pieces.append((start, text[start:cut]))
        start = cut
    return pieces or [(0, text)]


def _clean_date(value: str | None) -> str | None:
    if not value:
        return None
    value = value.strip()
    if re.fullmatch(r"\d{4}(-\d{2}(-\d{2})?)?", value):
        return value
    return parse_date(value)


def facts_from_payload(
    items: list[dict[str, Any]], doc: NoteDoc, text: str, offset: int = 0
) -> tuple[list[Fact], list[dict[str, Any]]]:
    """Normalize and ground raw extracted items; return (facts, dropped)."""
    facts: list[Fact] = []
    dropped: list[dict[str, Any]] = []
    seen: set[tuple] = set()
    for raw in items:
        quote = str(raw.get("quote") or "")
        value = str(raw.get("value") or "").strip()
        category = normalize_choice(raw.get("category"), CATEGORIES, "diagnosis")
        attribute = normalize_attribute(str(raw.get("attribute") or category), category)
        span = locate(quote, text)
        if span is None:
            dropped.append(
                {"doc_id": doc.doc_id, "reason": "quote_not_found",
                 "attribute": attribute, "value": value, "quote": quote[:200]}
            )
            continue
        located = text[span.start : span.end]
        if not value_supported(value, located):
            dropped.append(
                {"doc_id": doc.doc_id, "reason": "value_not_in_quote",
                 "attribute": attribute, "value": value, "quote": located[:200]}
            )
            continue
        start, end = span.start + offset, span.end + offset
        unit = str(raw.get("unit") or "").strip() or None
        event_date = _clean_date(raw.get("event_date"))
        relative = (raw.get("relative_time") or "").strip() or None
        assertion = normalize_choice(raw.get("assertion"), ASSERTIONS, "present")
        normalized = normalize_value(attribute, value, unit)
        key = (attribute, str(normalized), assertion, start)
        if key in seen:
            continue
        seen.add(key)
        flags = []
        if span.method == "fuzzy":
            flags.append("fuzzy_grounding")
        if doc.derived == "ocr":
            flags.append("ocr_transcribed")
        facts.append(
            Fact(
                fact_id=make_fact_id(doc.doc_id, start, end, attribute, value),
                category=category,
                attribute=attribute,
                value=value,
                unit=unit,
                normalized_value=normalized,
                event_date=event_date,
                date_kind="explicit" if event_date else "relative" if relative else "note_date",
                relative_time=relative,
                assertion=assertion,
                source_type=normalize_choice(raw.get("source_type"), SOURCE_TYPES, "unknown"),
                note_id=doc.doc_id,
                note_label=doc.label,
                note_date=doc.note_date,
                section=_section_for(doc.sections, start),
                char_start=start,
                char_end=end,
                quote=located,
                derived=doc.derived,
                grounding=span.method,
                grounding_score=span.score,
                flags=flags,
            )
        )
    return facts, dropped


def extract_doc(
    doc: NoteDoc, *, focus: str | None = None, model: str | None = None
) -> tuple[list[Fact], list[dict[str, Any]]]:
    """Run structured extraction over one document and ground every fact."""
    from src.core.gemini_client import NOTES_EXTRACT_MODEL

    facts: list[Fact] = []
    dropped: list[dict[str, Any]] = []
    sections = ", ".join(dict.fromkeys(s["name"] for s in doc.sections)) or "(none)"
    for offset, text in _chunks(doc):
        prompt = _EXTRACT_PROMPT.format(
            label=doc.label,
            doc_id=doc.doc_id,
            note_date=doc.note_date or "unknown",
            date_confidence=doc.date_confidence,
            note_type=doc.note_type,
            ocr_hint=(
                f"; this is an OCR transcript of PDF page {doc.page_index}"
                if doc.derived == "ocr" else ""
            ),
            sections=sections,
            focus=f"Focus: only extract facts relevant to: {focus}\n" if focus else "",
            text=text,
        )
        payload = llm.generate_json(
            prompt, model=model or NOTES_EXTRACT_MODEL, schema=ExtractionResult
        )
        items = payload.get("facts", []) if isinstance(payload, dict) else payload
        got, lost = facts_from_payload(list(items or []), doc, text, offset)
        facts.extend(got)
        dropped.extend(lost)
    return facts, dropped


def transcribe_pdf_pages(case, ref, cached_docs: dict[str, dict]) -> list[NoteDoc]:
    """Vision-transcribe low-quality PDF pages; reuse cached transcripts."""
    from google.genai import types

    from src.core.gemini_client import NOTES_EXTRACT_MODEL

    qualities = ref.meta.get("page_quality") or []
    docs = []
    for page_index, level in enumerate(qualities[:MAX_OCR_PAGES]):
        if level == "good":
            continue
        doc_id = f"{ref.id}#ocr{page_index}"
        page = case.fetch(ref.id, view="page", page_index=page_index)["data"]
        source_hash = hashlib.sha1(page["pdf_base64"].encode("ascii")).hexdigest()
        cached = cached_docs.get(doc_id)
        if cached and cached.get("source_hash") == source_hash and cached.get("text"):
            transcript = cached["text"]
        else:
            transcript = llm.generate_text(
                [
                    types.Part.from_bytes(
                        data=base64.b64decode(page["pdf_base64"]),
                        mime_type="application/pdf",
                    ),
                    types.Part.from_text(text=_OCR_PROMPT),
                ],
                model=NOTES_EXTRACT_MODEL,
            )
        doc = NoteDoc(
            doc_id=doc_id,
            note_id=ref.id,
            label=f"{ref.label} (OCR p{page_index + 1})",
            note_date=ref.meta.get("note_date"),
            date_confidence=ref.meta.get("note_date_confidence", "none"),
            note_type=ref.meta.get("note_type", "unknown"),
            text=transcript,
            derived="ocr",
            page_index=page_index,
        )
        doc.sections = []
        docs.append(doc)
        cached_docs[doc_id] = {"source_hash": source_hash}
    return docs


def _dedupe_ocr(facts: list[Fact]) -> list[Fact]:
    """Drop OCR facts that repeat a native fact (same attribute and value)."""
    native = {(f.attribute, str(f.normalized_value), f.assertion) for f in facts if f.derived == "native"}
    return [
        f for f in facts
        if f.derived == "native" or (f.attribute, str(f.normalized_value), f.assertion) not in native
    ]


def finalize(ledger: Ledger) -> Ledger:
    """Recompute the deterministic timeline and conflicts from facts."""
    from src.agents.notes.conflicts import detect_conflicts
    from src.agents.notes.timeline import build_timeline

    ledger.timeline = build_timeline(ledger.facts, ledger.notes)
    ledger.conflicts = detect_conflicts(ledger.facts, ledger.notes, ledger.timeline)
    return ledger


def build_ledger(
    patient_id: str, case, *, previous: Ledger | None = None, ocr: bool = True
) -> Ledger:
    """Extract (or reuse cached) facts for every note and assemble the ledger."""
    from src.core.gemini_client import NOTES_EXTRACT_MODEL

    native = load_native_docs(case)
    cached_docs = dict(previous.docs) if previous else {}
    derived: list[NoteDoc] = []
    if ocr:
        for ref in case.report.notes:
            if ref.meta.get("format") == "pdf" and ref.meta.get("text_quality") not in {"good", None}:
                try:
                    derived.extend(transcribe_pdf_pages(case, ref, cached_docs))
                except Exception as exc:  # noqa: BLE001 - OCR is best-effort
                    print(f"[notes] OCR failed for {ref.label}: {exc}")
    all_docs = native + derived

    reuse: dict[str, list[Fact]] = {}
    if previous:
        by_doc: dict[str, list[Fact]] = {}
        for item in previous.facts:
            by_doc.setdefault(item.note_id, []).append(item)
        for doc in all_docs:
            if previous.docs.get(doc.doc_id, {}).get("hash") == doc.hash:
                reuse[doc.doc_id] = by_doc.get(doc.doc_id, [])

    todo = [doc for doc in all_docs if doc.doc_id not in reuse]
    results: dict[str, tuple[list[Fact], list[dict]]] = {}
    if todo:
        with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(todo))) as pool:
            for doc, result in zip(todo, pool.map(_safe_extract, todo)):
                results[doc.doc_id] = result

    facts: list[Fact] = []
    dropped: list[dict] = [d for d in (previous.dropped if previous else []) if d.get("doc_id") in reuse]
    docs_meta: dict[str, dict[str, Any]] = {}
    for doc in all_docs:
        got, lost = (reuse[doc.doc_id], []) if doc.doc_id in reuse else results[doc.doc_id]
        facts.extend(got)
        dropped.extend(lost)
        meta = {
            "note_id": doc.note_id,
            "label": doc.label,
            "derived": doc.derived,
            "page_index": doc.page_index,
            "hash": doc.hash,
            "chars": len(doc.text),
        }
        if doc.derived == "ocr":
            meta["text"] = doc.text
            meta["source_hash"] = cached_docs.get(doc.doc_id, {}).get("source_hash")
        if any(d.get("doc_id") == doc.doc_id and d.get("reason") == "extraction_failed" for d in lost):
            meta["hash"] = None  # retry next build
        docs_meta[doc.doc_id] = meta

    notes = [
        {
            "note_id": ref.id,
            "label": ref.label,
            "note_date": ref.meta.get("note_date"),
            "note_date_confidence": ref.meta.get("note_date_confidence"),
            "note_date_flags": ref.meta.get("note_date_flags", []),
            "note_type": ref.meta.get("note_type"),
            "text_quality": ref.meta.get("text_quality"),
            "format": ref.meta.get("format"),
        }
        for ref in case.report.notes
    ]
    ledger = Ledger(
        patient_id=patient_id,
        model=NOTES_EXTRACT_MODEL,
        built_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        notes=notes,
        docs=docs_meta,
        facts=_dedupe_ocr(facts),
        dropped=dropped,
    )
    finalize(ledger)
    if os.getenv("NOTES_ADJUDICATE", "1") != "0":
        from src.agents.notes.conflicts import adjudicate
        from src.core.gemini_client import NOTES_REASON_MODEL

        cached = {
            c["conflict_id"]: c["adjudication"]
            for c in (previous.conflicts if previous else [])
            if c.get("adjudication")
        }
        adjudicate(ledger.conflicts, model=NOTES_REASON_MODEL, cached=cached)
    return ledger


def _safe_extract(doc: NoteDoc) -> tuple[list[Fact], list[dict]]:
    try:
        return extract_doc(doc)
    except Exception as exc:  # noqa: BLE001 - one bad note must not sink the ledger
        return [], [{"doc_id": doc.doc_id, "reason": "extraction_failed", "error": str(exc)[:300]}]


def _lock_for(patient_id: str) -> threading.Lock:
    with _LOCKS_GUARD:
        return _LOCKS.setdefault(patient_id, threading.Lock())


def get_ledger(patient_id: str, case=None, *, force: bool = False) -> Ledger:
    """Return the patient's ledger, building or incrementally refreshing it."""
    from src.retrieval.case_store import get_case

    case = case or get_case(patient_id)
    with _lock_for(patient_id):
        cached = _MEMORY.get(patient_id)
        path = ledger_path(patient_id)
        if cached is None and path.exists():
            try:
                cached = Ledger.load(path)
                if cached.version != LEDGER_VERSION:
                    cached = None
            except Exception:  # noqa: BLE001 - corrupt cache is rebuilt
                cached = None
        if cached is not None and not force and _is_fresh(cached, case):
            _MEMORY[patient_id] = cached
            return cached
        ledger = build_ledger(patient_id, case, previous=None if force else cached)
        ledger.save(path)
        _MEMORY[patient_id] = ledger
        return ledger


def cached_ledger(patient_id: str) -> Ledger | None:
    """The in-memory or on-disk ledger without building one (None if absent)."""
    ledger = _MEMORY.get(patient_id)
    if ledger is not None:
        return ledger
    path = ledger_path(patient_id)
    if not path.exists():
        return None
    try:
        ledger = Ledger.load(path)
    except Exception:  # noqa: BLE001 - corrupt cache is treated as absent
        return None
    return ledger if ledger.version == LEDGER_VERSION else None


def _is_fresh(ledger: Ledger, case) -> bool:
    """True when every native note's hash matches the cached ledger."""
    native_ids = {ref.id for ref in case.report.notes}
    cached_native = {k for k, v in ledger.docs.items() if v.get("derived") == "native"}
    if native_ids != cached_native:
        return False
    for doc in load_native_docs(case):
        if ledger.docs.get(doc.doc_id, {}).get("hash") != doc.hash:
            return False
    return True


def reset_memory() -> None:
    """Drop in-memory ledgers (tests)."""
    _MEMORY.clear()
