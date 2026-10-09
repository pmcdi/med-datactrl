"""Deterministic conflict detection across grounded facts.

Groups facts by canonical attribute, compares normalized values (numeric
tolerance or categorical equality), ranks sources by reliability, and emits
conflict records carrying both source spans. Measurement variance
(3.87 vs 3.9 cm) is reported separately from real discordance (segment 4a vs 8).
"""

from __future__ import annotations

import hashlib
import re
from datetime import date
from typing import Any

from src.agents.notes.schema import SOURCE_RELIABILITY, Fact
from src.agents.notes.timeline import parse_partial

# Attributes that should not change over a patient's course.
STATIC_PREFIXES = (
    "demographics.", "tumor.", "pathology.", "diagnosis.", "dx.", "staging.",
    "allergy.", "family_history.",
)
# Attributes that change over time: compare only facts from the same date.
TIME_VARYING_PREFIXES = ("lab.", "vital.", "rt.", "performance.")
# Attributes where present-vs-negated across notes is worth flagging.
ASSERTION_PREFIXES = STATIC_PREFIXES + ("comorbidity.", "imaging.", "social_history.")

REL_TOL = 0.05
ABS_TOL = {"tumor.size_cm": 0.15}
# Only continuous measurements get a relative tolerance; counts/scores must match.
CONTINUOUS_PREFIXES = ("tumor.size", "lab.", "vital.", "rt.dose")
TIME_GAP_DAYS = 365
_CATEGORY_TO_STATIC = {"allergy", "family_history"}


def _cmp_date(fact: Fact) -> date | None:
    parsed = parse_partial(fact.event_date) or parse_partial(fact.note_date)
    return parsed[0] if parsed else None


def _value_entry(fact: Fact) -> dict[str, Any]:
    unit = fact.unit or ("cm" if fact.attribute.endswith("_cm") and re.fullmatch(r"[\d.]+", fact.value.strip()) else None)
    return {
        "value": fact.value + (f" {unit}" if unit and unit not in fact.value else ""),
        "normalized": fact.normalized_value,
        "assertion": fact.assertion,
        "fact_id": fact.fact_id,
        "note_id": fact.note_id,
        "note_label": fact.note_label,
        "note_date": fact.note_date,
        "source_type": fact.source_type,
        "derived": fact.derived,
        "quote": fact.quote,
        "char_start": fact.char_start,
        "char_end": fact.char_end,
    }


def _same(a: Fact, b: Fact, attribute: str) -> str:
    """'same' | 'variance' | 'different' for two normalized values."""
    va, vb = a.normalized_value, b.normalized_value
    if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
        if va == vb:
            return "same"
        if not attribute.startswith(CONTINUOUS_PREFIXES):
            return "different"
        diff = abs(va - vb)
        if diff <= ABS_TOL.get(attribute, 0.0) or diff <= REL_TOL * max(abs(va), abs(vb)):
            return "variance"
        return "different"
    sa, sb = str(va or "").strip(), str(vb or "").strip()
    if not sa or not sb or sa == sb:
        return "same"
    if attribute == "tumor.location" and (sa.startswith("segment") != sb.startswith("segment")):
        return "same"  # a non-specific location cannot contradict a specific segment
    if attribute == "tumor.stage" and _stage_system(sa) != _stage_system(sb):
        return "same"  # different staging systems (TNM vs group vs LI-RADS) are not comparable
    ta, tb = _content_tokens(sa), _content_tokens(sb)
    if attribute == "tumor.histology":
        ta, tb = ta - _HISTOLOGY_QUALIFIERS, tb - _HISTOLOGY_QUALIFIERS
        ta = {t for t in ta if not t.isdigit()}
        tb = {t for t in tb if not t.isdigit()}
    if not ta or not tb:
        return "same"
    small, big = sorted((ta, tb), key=len)
    if all(any(_token_match(t, u) for u in big) for t in small):
        return "same"
    return "different"


_FILLER = frozenset(
    "a an the of for with and or is are was were per by in on at to as be identified "
    "noted seen present surgeon surgical".split()
)


# Grade and coding qualifiers that refine rather than contradict a histology.
_HISTOLOGY_QUALIFIERS = frozenset(
    "well moderately poorly un undifferentiated differentiated diff nos type "
    "grade g1 g2 g3 g4 invasive".split()
)


def _stage_system(text: str) -> str:
    t = text.lower()
    if re.search(r"\blr\s*-?\s*\d|li-?rads", t):
        return "lirads"
    if "bclc" in t:
        return "bclc"
    if re.search(r"\b[cpyr]?t[0-4xis][a-d]?\b|\bn[0-3x]\b|\bm[01x]\b|t[0-4][a-d]?n[0-3]", t):
        return "tnm"
    if re.search(r"\bstage\b|\b[iv]+[abc]?\b", t):
        return "group"
    return "other"


def _content_tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _FILLER}


def _token_match(a: str, b: str) -> bool:
    """Equal tokens, or an abbreviation prefix of >= 4 chars (diff -> differentiated)."""
    if a == b:
        return True
    short, long_ = sorted((a, b), key=len)
    return len(short) >= 4 and long_.startswith(short)


_LATER_DISEASE = re.compile(r"recurren|metasta|\bmets?\b|new (hepatic |liver )?lesion|bone|involvement", re.I)


def _describes_later_disease(fact: Fact) -> bool:
    """Recurrence/metastasis mentions describe a different lesion than the primary."""
    return fact.attribute.startswith("tumor.") and bool(
        _LATER_DISEASE.search(f"{fact.value} {fact.quote}")
    )


def _cluster(facts: list[Fact], attribute: str) -> list[list[Fact]]:
    """Group facts whose values agree (equal or within measurement tolerance)."""
    clusters: list[list[Fact]] = []
    for fact in facts:
        for cluster in clusters:
            if all(_same(member, fact, attribute) != "different" for member in cluster):
                cluster.append(fact)
                break
        else:
            clusters.append([fact])
    return clusters


def _record(
    attribute: str, kind: str, reps: list[Fact], outlier_notes: frozenset[str] = frozenset()
) -> dict[str, Any]:
    dates = [d for d in (_cmp_date(f) for f in reps if f.source_note_id not in outlier_notes) if d]
    spread = (max(dates) - min(dates)).days if len(dates) > 1 else 0
    sources = {f.source_type for f in reps}
    preferred = min(reps, key=_rank)
    return {
        "conflict_id": _conflict_id(attribute, [f.fact_id for f in reps]),
        "attribute": attribute,
        "kind": kind,
        "severity": _severity(attribute, kind, reps, spread),
        "values": [_value_entry(f) for f in reps],
        "preferred": _value_entry(preferred),
        "rationale": _rationale(kind, preferred, reps, spread, sources),
    }


def _rank(fact: Fact) -> tuple:
    d = _cmp_date(fact)
    return (
        SOURCE_RELIABILITY.get(fact.source_type, 7),
        1 if fact.derived == "ocr" else 0,
        -(d.toordinal() if d else 0),
    )


def _conflict_id(attribute: str, fact_ids: list[str]) -> str:
    raw = attribute + "|" + "|".join(sorted(fact_ids))
    return "C" + hashlib.sha1(raw.encode()).hexdigest()[:8]


def _is_static(fact: Fact) -> bool:
    return fact.attribute.startswith(STATIC_PREFIXES) or fact.category in _CATEGORY_TO_STATIC


def _value_conflicts(
    facts: list[Fact], outlier_notes: frozenset[str] = frozenset()
) -> list[dict[str, Any]]:
    groups: dict[tuple, list[Fact]] = {}
    for fact in facts:
        if fact.assertion not in {"present", "historical"}:
            continue
        if _describes_later_disease(fact):
            continue
        if _is_static(fact):
            key: tuple = (fact.attribute,)
        elif fact.attribute.startswith(TIME_VARYING_PREFIXES):
            d = _cmp_date(fact)
            if d is None:
                continue
            key = (fact.attribute, d.isoformat())
        else:
            continue
        groups.setdefault(key, []).append(fact)

    out: list[dict[str, Any]] = []
    for key, members in groups.items():
        attribute = key[0]
        clusters = _cluster(members, attribute)
        for cluster in clusters:
            distinct: dict[str, Fact] = {}
            for fact in sorted(cluster, key=_rank):
                if isinstance(fact.normalized_value, (int, float)):
                    distinct.setdefault(str(fact.normalized_value), fact)
            if len(distinct) > 1:
                out.append(_record(attribute, "measurement_variance", list(distinct.values()), outlier_notes))
        if len(clusters) >= 2:
            reps = [min(cluster, key=_rank) for cluster in clusters]
            out.append(_record(attribute, "discordant", reps, outlier_notes))
    return out


def _severity(attribute: str, kind: str, reps: list[Fact], spread: int) -> str:
    if kind == "measurement_variance":
        return "info"
    if attribute == "demographics.age":
        values = [f.normalized_value for f in reps if isinstance(f.normalized_value, (int, float))]
        if values and max(values) - min(values) <= 1:
            return "low"
    if spread > TIME_GAP_DAYS:
        return "medium"
    if attribute.startswith(("tumor.", "diagnosis.", "allergy.", "pathology.")):
        return "high"
    return "medium"


def _rationale(kind: str, preferred: Fact, reps: list[Fact], spread: int, sources: set[str]) -> str:
    if kind == "measurement_variance":
        return "Values agree within measurement tolerance; wording or rounding differs."
    parts = []
    others = [f for f in reps if f is not preferred]
    if any(SOURCE_RELIABILITY.get(f.source_type, 7) > SOURCE_RELIABILITY.get(preferred.source_type, 7) for f in others):
        parts.append(
            f"Preferring {preferred.source_type.replace('_', ' ')} ({preferred.note_label}) "
            "by source reliability."
        )
    else:
        parts.append(f"Sources are of similar reliability; most recent is {preferred.note_label}.")
    if spread > TIME_GAP_DAYS:
        parts.append("Values were documented more than a year apart and may reflect a real change.")
    if {"pathology"} & sources and sources & {"radiology_final", "radiology_prelim"}:
        parts.append("Imaging and pathology measurements can legitimately differ.")
    if any(f.derived == "ocr" for f in reps):
        parts.append("At least one value comes from OCR of a scanned page.")
    return " ".join(parts)


_NEGATIVE_VALUES = frozenset({"no", "none", "absent", "negative", "not identified", "not seen"})


def _polarity(fact: Fact) -> str:
    """'negated: negative for tumor' states a result, not an absence of the attribute."""
    if fact.assertion == "negated" and str(fact.normalized_value or "").strip() in _NEGATIVE_VALUES - {"no", "none", "absent"}:
        return "present"
    return fact.assertion


def _assertion_conflicts(facts: list[Fact]) -> list[dict[str, Any]]:
    groups: dict[str, list[Fact]] = {}
    for fact in facts:
        if fact.attribute.startswith(ASSERTION_PREFIXES) and fact.assertion in {"present", "negated"}:
            groups.setdefault(fact.attribute, []).append(fact)
    out = []
    for attribute, members in groups.items():
        present = [f for f in members if _polarity(f) == "present"]
        negated = [f for f in members if _polarity(f) == "negated"]
        if not present or not negated:
            continue
        # A negation of a specific value does not contradict a different present value.
        pairs = [
            (p, n) for p in present for n in negated
            if _same(p, n, attribute) == "same" or not str(n.normalized_value or "").strip()
            or n.normalized_value in _NEGATIVE_VALUES
        ]
        if not pairs:
            continue
        p, n = pairs[0]
        preferred = min((p, n), key=_rank)
        out.append(
            {
                "conflict_id": _conflict_id(attribute, [p.fact_id, n.fact_id]),
                "attribute": attribute,
                "kind": "assertion_conflict",
                "severity": "medium",
                "values": [_value_entry(p), _value_entry(n)],
                "preferred": _value_entry(preferred),
                "rationale": (
                    f"Documented as present in {p.note_label} but negated in {n.note_label}. "
                    f"Preferring {preferred.note_label} by source reliability/recency."
                ),
            }
        )
    return out


def _date_conflicts(notes: list[dict[str, Any]], timeline: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for event in timeline:
        if event.get("kind") != "note":
            continue
        flags = [f for f in event.get("flags") or [] if f == "date_outlier" or f.startswith("filename_date_disagrees")]
        if not flags:
            continue
        note = next((n for n in notes if n["note_id"] == event["note_id"]), {})
        reason = (
            "Note date is more than 5 years from the rest of the record"
            if "date_outlier" in flags
            else "Header date and filename date disagree"
        )
        out.append(
            {
                "conflict_id": _conflict_id("note.date", [event["note_id"]]),
                "attribute": "note.date",
                "kind": "date_inconsistency",
                "severity": "medium",
                "values": [{
                    "value": str(event.get("date")),
                    "normalized": event.get("date"),
                    "assertion": "present",
                    "fact_id": None,
                    "note_id": event["note_id"],
                    "note_label": event["note_label"],
                    "note_date": event.get("date"),
                    "source_type": note.get("note_type") or "unknown",
                    "derived": "native",
                    "quote": None,
                    "char_start": None,
                    "char_end": None,
                }],
                "preferred": None,
                "rationale": (
                    f"{reason} (date confidence: {note.get('note_date_confidence')}). "
                    "Verify before using it for sequencing."
                ),
            }
        )
    return out


_SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2, "info": 3}


def detect_conflicts(
    facts: list[Fact], notes: list[dict[str, Any]], timeline: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """All value, assertion and date conflicts, most severe first."""
    outlier_notes = frozenset(
        e["note_id"] for e in timeline if e.get("kind") == "note" and "date_outlier" in (e.get("flags") or [])
    )
    conflicts = (
        _value_conflicts(facts, outlier_notes)
        + _assertion_conflicts(facts)
        + _date_conflicts(notes, timeline)
    )
    conflicts.sort(key=lambda c: (_SEVERITY_ORDER.get(c["severity"], 9), c["attribute"]))
    return conflicts


_ADJUDICATE_PROMPT = """You review candidate discrepancies found by string comparison in a patient's clinical notes.
For each item decide whether the values truly CONTRADICT each other (they cannot both be true of the same patient
and the same lesion/event), or are COMPATIBLE (synonyms, abbreviations, different granularity, one refines the other,
different coding systems for the same thing).
Examples: "glottic larynx" vs "both vocal cords" -> compatible; "segment 4a" vs "segment 8" -> contradict;
"HCC" vs "cholangiocarcinoma" -> contradict; "T1bN0M0" vs "stage I" -> compatible.

Items:
{items}

Return ONLY JSON: {{"verdicts": [{{"conflict_id": "<id>", "contradict": true | false, "reason": "<short>"}}]}}"""


def _needs_adjudication(conflict: dict[str, Any]) -> bool:
    return conflict["kind"] == "discordant" and not all(
        isinstance(v.get("normalized"), (int, float)) for v in conflict["values"]
    )


def adjudicate(
    conflicts: list[dict[str, Any]],
    *,
    model: str,
    cached: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Downgrade string discordances an LLM judges compatible (synonyms, granularity).

    Verdicts are cached by conflict id; failures leave conflicts untouched.
    """
    from src.agents.notes import llm

    cached = cached or {}
    pending = [c for c in conflicts if _needs_adjudication(c) and c["conflict_id"] not in cached]
    verdicts = dict(cached)
    if pending:
        items = "\n".join(
            f"- {c['conflict_id']} {c['attribute']}: "
            + " | ".join(f"\"{v['value']}\" ({v['source_type']}, quote: \"{(v.get('quote') or '')[:160]}\")" for v in c["values"])
            for c in pending
        )
        try:
            payload = llm.generate_json(_ADJUDICATE_PROMPT.format(items=items), model=model)
            for item in (payload or {}).get("verdicts", []):
                if isinstance(item, dict) and item.get("conflict_id"):
                    verdicts[str(item["conflict_id"])] = {
                        "contradict": bool(item.get("contradict", True)),
                        "reason": str(item.get("reason") or "")[:200],
                    }
        except Exception as exc:  # noqa: BLE001 - adjudication is best-effort
            print(f"[notes] conflict adjudication failed: {exc}")
    for conflict in conflicts:
        verdict = verdicts.get(conflict["conflict_id"])
        if not verdict or not _needs_adjudication(conflict):
            continue
        conflict["adjudication"] = verdict
        if not verdict["contradict"]:
            conflict["kind"] = "equivalent_wording"
            conflict["severity"] = "info"
            conflict["rationale"] = f"Judged compatible: {verdict['reason']}".strip()
    conflicts.sort(key=lambda c: (_SEVERITY_ORDER.get(c["severity"], 9), c["attribute"]))
    return conflicts
