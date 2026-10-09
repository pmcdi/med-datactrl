"""Deterministic patient timeline with relative-date anchoring.

Relative expressions ("~22 mo post-op", "656 days from index") are resolved
against anchor events when one exists and marked ``inferred``; otherwise they
stay ``relative`` and are placed at their note date. Uncertain dates are kept
visible, never silently corrected.
"""

from __future__ import annotations

import re
import statistics
from datetime import date, timedelta
from typing import Any

from dateutil.relativedelta import relativedelta

from src.agents.notes.schema import Fact

TIMELINE_CATEGORIES = {
    "diagnosis", "staging", "tumor", "pathology", "imaging_finding", "procedure",
    "treatment", "response", "toxicity", "outcome", "performance_status", "lab",
}
OUTLIER_YEARS = 5

_REL_RE = re.compile(
    r"(?P<approx>~|about|approx\.?|approximately)?\s*(?P<n>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>days?|d|wks?|weeks?|mos?|months?|yrs?|years?)\b"
    r"(?P<tail>[^.;\n]{0,40})",
    re.IGNORECASE,
)
_SURGERY_RE = re.compile(r"resect|surg|ectomy|excision|operat|post[\s-]?op", re.I)
_INDEX_RE = re.compile(r"index|diagnos", re.I)
_TREATMENT_START_RE = re.compile(r"treatment start|rt start|start of (rt|treatment)|radiation start", re.I)


def parse_partial(value: str | None) -> tuple[date, str] | None:
    """Parse ``YYYY``, ``YYYY-MM`` or ``YYYY-MM-DD`` into (date, precision)."""
    if not value:
        return None
    parts = value.split("-")
    try:
        if len(parts) == 1 and len(parts[0]) == 4:
            return date(int(parts[0]), 1, 1), "year"
        if len(parts) == 2:
            return date(int(parts[0]), int(parts[1]), 1), "month"
        if len(parts) == 3:
            return date(int(parts[0]), int(parts[1]), int(parts[2])), "day"
    except ValueError:
        return None
    return None


def _format(d: date, precision: str) -> str:
    if precision == "year":
        return f"{d.year}"
    if precision == "month":
        return f"{d.year}-{d.month:02d}"
    return d.isoformat()


def _fact_date(fact: Fact) -> tuple[date, str] | None:
    return parse_partial(fact.event_date) or parse_partial(fact.note_date)


def _find_anchor(kind: str, facts: list[Fact], is_outlier=None) -> Fact | None:
    """Earliest dated fact (not a date outlier) that can anchor a relative expression."""
    candidates: list[tuple[date, Fact]] = []
    for fact in facts:
        if fact.relative_time:
            continue
        hay = f"{fact.attribute} {fact.value} {fact.quote}"
        if kind == "surgery":
            ok = fact.category in {"procedure", "treatment"} and _SURGERY_RE.search(hay)
        elif kind == "treatment_start":
            ok = fact.category == "treatment" or fact.attribute.startswith("rt.")
        else:
            ok = fact.category in {"diagnosis", "pathology", "tumor"}
        if not ok:
            continue
        parsed = _fact_date(fact)
        if parsed and not (is_outlier and is_outlier(parsed[0])):
            candidates.append((parsed[0], fact))
    if not candidates:
        return None
    return min(candidates, key=lambda pair: pair[0])[1]


def resolve_relative(fact: Fact, facts: list[Fact], is_outlier=None) -> dict[str, Any] | None:
    """Resolve a fact's relative_time against an anchor event, if possible."""
    match = _REL_RE.search(fact.relative_time or "")
    if not match:
        return None
    text = fact.relative_time or ""
    tail = match.group("tail") or ""
    if _SURGERY_RE.search(tail) or _SURGERY_RE.search(text):
        kind = "surgery"
    elif _TREATMENT_START_RE.search(text):
        kind = "treatment_start"
    elif _INDEX_RE.search(text):
        kind = "index"
    else:
        return None
    anchor = _find_anchor(kind, facts, is_outlier)
    if anchor is None:
        return None
    parsed = _fact_date(anchor)
    if parsed is None:
        return None
    base, precision = parsed
    n = float(match.group("n"))
    unit = match.group("unit").lower()
    if unit.startswith("d"):
        resolved = base + timedelta(days=round(n))
        out_precision = "day" if precision == "day" else "month"
    elif unit.startswith("w"):
        resolved = base + timedelta(weeks=n)
        out_precision = "day" if precision == "day" else "month"
    elif unit.startswith("m"):
        resolved = base + relativedelta(months=round(n))
        out_precision = "month"
    else:
        resolved = base + timedelta(days=round(n * 365.25))
        out_precision = "month"
    if precision == "year":
        out_precision = "year" if unit.startswith("y") else "month"
    approx = bool(match.group("approx")) or precision != "day"
    return {
        "date": _format(resolved, out_precision),
        "anchor_fact_id": anchor.fact_id,
        "anchor_kind": kind,
        "approximate": approx,
    }


def _label(fact: Fact) -> str:
    neg = "NO " if fact.assertion == "negated" else ""
    qual = "" if fact.assertion in {"present", "negated"} else f" ({fact.assertion})"
    return f"{neg}{fact.attribute} = {fact.value}{qual} [{fact.note_label}]"


def build_timeline(facts: list[Fact], notes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ordered events: one per note plus clinically timed facts."""
    events: list[dict[str, Any]] = []
    # Reference point for outliers: one vote per dated note (fact dates include
    # legitimately old history such as prior surgeries).
    reference = [parsed[0] for parsed in (parse_partial(n.get("note_date")) for n in notes) if parsed]
    median = (
        date.fromordinal(int(statistics.median(d.toordinal() for d in reference)))
        if len(reference) >= 3 else None
    )

    def outlier(d: date | None) -> bool:
        return bool(d and median and abs((d - median).days) > OUTLIER_YEARS * 365)

    for note in notes:
        parsed = parse_partial(note.get("note_date"))
        flags = list(note.get("note_date_flags") or [])
        if note.get("note_date_confidence") in {"low", "none"}:
            flags.append(f"note_date_confidence_{note.get('note_date_confidence')}")
        if parsed and outlier(parsed[0]):
            flags.append("date_outlier")
        events.append({
            "kind": "note",
            "date": note.get("note_date"),
            "date_kind": "note",
            "label": f"Note: {note['label']} ({note.get('note_type') or 'unknown'})",
            "note_id": note["note_id"],
            "note_label": note["label"],
            "fact_id": None,
            "flags": flags,
        })

    for fact in facts:
        if fact.category not in TIMELINE_CATEGORIES:
            continue
        flags = list(fact.flags)
        resolved = None
        if fact.event_date:
            when, kind = fact.event_date, "explicit"
        elif fact.relative_time:
            resolved = resolve_relative(fact, facts, outlier)
            if resolved:
                when, kind = resolved["date"], "inferred"
                if resolved["approximate"]:
                    flags.append("approximate")
            else:
                when, kind = fact.note_date, "relative"
        else:
            when, kind = fact.note_date, "note_date"
        parsed = parse_partial(when)
        if parsed and outlier(parsed[0]) and fact.assertion != "historical":
            flags.append("date_outlier")
        events.append({
            "kind": "fact",
            "date": when,
            "date_kind": kind,
            "label": _label(fact),
            "category": fact.category,
            "attribute": fact.attribute,
            "value": fact.value,
            "assertion": fact.assertion,
            "relative_time": fact.relative_time,
            "anchor_fact_id": resolved["anchor_fact_id"] if resolved else None,
            "fact_id": fact.fact_id,
            "note_id": fact.note_id,
            "note_label": fact.note_label,
            "quote": fact.quote,
            "char_start": fact.char_start,
            "char_end": fact.char_end,
            "flags": flags,
        })

    def sort_key(event: dict[str, Any]):
        parsed = parse_partial(event.get("date"))
        return (
            parsed[0] if parsed else date.max,
            0 if event["kind"] == "note" else 1,
            event.get("note_label") or "",
        )

    events.sort(key=sort_key)
    return events
