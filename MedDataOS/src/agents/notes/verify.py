"""Claim verification: every surviving claim cites spans that exist and entail it.

1. Deterministic: each evidence item must resolve to a ledger fact or a quote
   that can be located in the cited note (exact/normalized/fuzzy).
2. One batched LLM entailment check of each claim against its evidence text.
Claims failing either step are dropped; partially entailed claims are kept and
flagged. "Not documented" abstentions are kept.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from src.agents.notes import llm
from src.agents.notes.grounding import locate
from src.agents.notes.ledger import Ledger
from src.agents.notes.schema import Fact

_ENTAIL_PROMPT = """You are verifying claims written about a patient against evidence from the patient's notes.
Each evidence item gives the source note (label, type, date), the cited QUOTE, and surrounding CONTEXT from the same note.
The note's date and type count as evidence (e.g. a radiology note dated 2004-02-24 supports "on imaging dated 2004-02-24").
Judge each claim:
- "supported": every clinical fact in the claim (values, units, locations, dates, negations, certainty) is backed by the evidence. Paraphrase is fine.
- "partial": the core clinical facts are backed, but the claim adds a minor detail or interpretive framing that is not.
- "unsupported": a key clinical fact is missing from or contradicted by the evidence.

{items}

Return JSON: {{"results": [{{"id": <int>, "verdict": "supported" | "partial" | "unsupported", "reason": "<short>"}}]}}"""
ENTAIL_CONTEXT_CHARS = 240

_BROAD_RE = re.compile(
    r"summar|overview|history|review|all |everything|course|timeline|discrepan|conflict|"
    r"inconsisten|contradict|disagree|reconcil|brief|hand ?off|tumou?r board",
    re.I,
)
_LATER_RE = re.compile(r"recur|relapse|metasta|progress", re.I)
_PRIMARY_RE = re.compile(r"primary|original|initial|resect|patholog|diagnos", re.I)
_QUERY_HINTS = {
    "size": ("size",), "big": ("size",), "large": ("size",), "measure": ("size",),
    "where": ("location",), "location": ("location",), "segment": ("location",),
    "site": ("location",), "age": ("age",), "old": ("age",), "date": ("date",),
    "when": ("date",), "grade": ("grade",), "stage": ("stage",), "margin": ("margin",),
    "dose": ("dose",), "fraction": ("fractions",), "ecog": ("ecog",),
    "performance": ("ecog",), "lirads": ("li_rads",), "li": ("li_rads",),
}


@dataclass
class Evidence:
    """One verified citation span."""

    doc_id: str
    label: str
    note_date: str | None
    start: int
    end: int
    quote: str
    derived: str
    fact_id: str | None
    method: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize for the Finding / citation card."""
        return {
            "note_id": self.doc_id,
            "source_note_id": self.doc_id.split("#", 1)[0],
            "label": self.label,
            "note_date": self.note_date,
            "start": self.start,
            "end": self.end,
            "quote": self.quote,
            "derived": self.derived,
            "fact_id": self.fact_id,
            "grounding": self.method,
        }


@dataclass
class Claim:
    """A claim with its verified evidence."""

    text: str
    evidence: list[Evidence]
    entailment: str = "unchecked"  # supported | partial | unchecked


@dataclass
class VerifiedAnswer:
    """Result of verification: surviving claims, abstentions, relevant conflicts."""

    claims: list[Claim] = field(default_factory=list)
    abstentions: list[str] = field(default_factory=list)
    dropped: list[dict[str, Any]] = field(default_factory=list)
    conflicts: list[dict[str, Any]] = field(default_factory=list)


class TextStore:
    """Lazy doc_id -> text lookup across native notes and OCR transcripts."""

    def __init__(self, ledger: Ledger, loader: Callable[[str], str | None]):
        """Bind to a ledger (for OCR transcripts) and a native-note loader."""
        self._ledger = ledger
        self._loader = loader
        self._cache: dict[str, str | None] = {}

    def get(self, doc_id: str) -> str | None:
        """Return the document text, or None if unknown."""
        if doc_id not in self._cache:
            doc = self._ledger.docs.get(doc_id) or {}
            if doc.get("derived") == "ocr":
                self._cache[doc_id] = doc.get("text")
            else:
                self._cache[doc_id] = self._loader(doc_id)
        return self._cache[doc_id]

    def doc_ids(self) -> list[str]:
        """All known document ids (native notes first)."""
        native = [n["note_id"] for n in self._ledger.notes]
        derived = [k for k, v in self._ledger.docs.items() if v.get("derived") == "ocr"]
        return native + derived


def _label_for(ledger: Ledger, doc_id: str) -> tuple[str, str | None, str]:
    doc = ledger.docs.get(doc_id) or {}
    meta = ledger.note_meta(doc_id) or {}
    label = doc.get("label") or meta.get("label") or doc_id
    return label, meta.get("note_date"), doc.get("derived") or "native"


def resolve_evidence(
    item: dict[str, Any],
    ledger: Ledger,
    texts: TextStore,
    extra_facts: dict[str, Fact] | None = None,
) -> Evidence | None:
    """Resolve one evidence item to a verified span, or None."""
    fact_id = (item.get("fact_id") or "").strip() or None
    fact = None
    if fact_id:
        fact = ledger.fact(fact_id) or (extra_facts or {}).get(fact_id)
    quote = (item.get("quote") or "").strip()
    if fact is not None and (not quote or _overlaps(quote, fact.quote)):
        text = texts.get(fact.note_id) if quote and len(quote) > len(fact.quote) else None
        span = locate(quote, text, hint=fact.char_start) if text else None
        if span and span.start <= fact.char_start and span.end >= fact.char_end:
            return Evidence(
                fact.note_id, fact.note_label, fact.note_date, span.start, span.end,
                text[span.start : span.end], fact.derived, fact.fact_id, "fact",
            )
        return Evidence(
            fact.note_id, fact.note_label, fact.note_date, fact.char_start,
            fact.char_end, fact.quote, fact.derived, fact.fact_id, "fact",
        )
    if not quote:
        return None
    doc_id = (item.get("note_id") or "").strip()
    candidates = [doc_id] if doc_id and texts.get(doc_id) is not None else []
    candidates += [d for d in texts.doc_ids() if d not in candidates]
    for candidate in candidates:
        text = texts.get(candidate)
        if not text:
            continue
        span = locate(quote, text)
        if span:
            label, note_date, derived = _label_for(ledger, candidate)
            return Evidence(
                candidate, label, note_date, span.start, span.end,
                text[span.start : span.end], derived, None, span.method,
            )
    return None


def _overlaps(a: str, b: str) -> bool:
    na, nb = re.sub(r"\s+", " ", a.lower()), re.sub(r"\s+", " ", b.lower())
    return na in nb or nb in na or bool(locate(a, b))


def _evidence_block(e: Evidence, ledger: Ledger | None, texts: TextStore | None) -> str:
    meta = (ledger.note_meta(e.doc_id) if ledger else None) or {}
    header = f"{e.label} | type: {meta.get('note_type') or 'unknown'} | date: {e.note_date or 'undated'}"
    if e.derived == "ocr":
        header += " | OCR transcript"
    text = texts.get(e.doc_id) if texts else None
    block = f'      - [{header}] QUOTE: "{e.quote}"'
    if text:
        lo = max(0, e.start - ENTAIL_CONTEXT_CHARS)
        hi = min(len(text), e.end + ENTAIL_CONTEXT_CHARS)
        context = re.sub(r"\s+", " ", text[lo:hi]).strip()
        block += f'\n        CONTEXT: "...{context}..."'
    return block


def entailment_check(
    claims: list[Claim],
    *,
    model: str,
    ledger: Ledger | None = None,
    texts: TextStore | None = None,
) -> dict[int, str] | None:
    """Batched LLM entailment; returns {index: verdict} or None when unavailable."""
    if not claims or os.getenv("NOTES_ENTAILMENT", "1") == "0":
        return None
    items = []
    for i, claim in enumerate(claims):
        evidence = "\n".join(_evidence_block(e, ledger, texts) for e in claim.evidence)
        items.append(f"[{i}] CLAIM: {claim.text}\n    EVIDENCE:\n{evidence}")
    try:
        payload = llm.generate_json(_ENTAIL_PROMPT.format(items="\n".join(items)), model=model)
    except Exception as exc:  # noqa: BLE001 - verification degrades to deterministic only
        print(f"[notes] entailment check failed: {exc}")
        return None
    results = payload.get("results", []) if isinstance(payload, dict) else payload
    out: dict[int, str] = {}
    for row in results or []:
        try:
            verdict = str(row.get("verdict") or "").lower()
            if verdict not in {"supported", "partial", "unsupported"}:
                verdict = "supported" if row.get("supported") else "unsupported"
            out[int(row["id"])] = verdict
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
    return out


def _query_terms(query: str) -> set[str]:
    query = re.sub(r"date of birth|\bdob\b", "birth", query.lower())  # not a note-date question
    words = set(re.findall(r"[a-z0-9_]+", query))
    terms = set(words)
    for word in words:
        terms.update(_QUERY_HINTS.get(word, ()))
    return terms


def relevant_conflicts(
    conflicts: list[dict[str, Any]],
    query: str,
    claims: list[Claim],
    mentioned: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Conflicts that touch the question, the cited facts, or were mentioned."""
    mentioned_set = set(mentioned or [])
    cited_facts = {e.fact_id for c in claims for e in c.evidence if e.fact_id}
    cited_docs = {e.doc_id.split("#", 1)[0] for c in claims for e in c.evidence}
    terms = _query_terms(query)
    broad = bool(_BROAD_RE.search(query))
    later_disease = bool(_LATER_RE.search(query)) and not _PRIMARY_RE.search(query)
    out = []
    for conflict in conflicts:
        attr_tokens = set(re.split(r"[._]", conflict["attribute"])) - {"cm", "gy"}
        fact_ids = {v.get("fact_id") for v in conflict["values"] if v.get("fact_id")}
        doc_ids = {str(v.get("note_id") or "").split("#", 1)[0] for v in conflict["values"]}
        info = conflict["severity"] == "info"
        by_hint = bool(attr_tokens & terms) and (not info or conflict["attribute"].split(".")[-1] in terms)
        if later_disease and conflict["attribute"].startswith("tumor."):
            by_hint = False  # primary-tumor conflicts are off-topic for recurrence questions
        hit = (
            conflict["conflict_id"] in mentioned_set
            or (not info and fact_ids & cited_facts)
            or by_hint
            or (broad and not info)
            or (conflict["kind"] == "date_inconsistency" and doc_ids & cited_docs)
        )
        if hit:
            out.append(conflict)
    return out


def verify_answer(
    payload: dict[str, Any],
    ledger: Ledger,
    texts: TextStore,
    query: str,
    *,
    model: str,
    extra_facts: dict[str, Fact] | None = None,
) -> VerifiedAnswer:
    """Verify the model's claims and attach relevant conflicts."""
    result = VerifiedAnswer()
    pending: list[Claim] = []
    for raw in payload.get("claims") or []:
        text = str(raw.get("text") or "").strip()
        if not text:
            continue
        status = str(raw.get("status") or "supported").lower()
        if status.startswith("not"):
            result.abstentions.append(text)
            continue
        evidence = []
        seen: set[tuple[str, int, int]] = set()
        for item in raw.get("evidence") or []:
            resolved = resolve_evidence(item, ledger, texts, extra_facts)
            if resolved and (resolved.doc_id, resolved.start, resolved.end) not in seen:
                seen.add((resolved.doc_id, resolved.start, resolved.end))
                evidence.append(resolved)
        if not evidence:
            result.dropped.append({"claim": text, "reason": "no_grounded_evidence"})
            continue
        pending.append(Claim(text, evidence))

    verdicts = entailment_check(pending, model=model, ledger=ledger, texts=texts)
    for i, claim in enumerate(pending):
        if verdicts is not None and i in verdicts:
            if verdicts[i] == "unsupported":
                result.dropped.append({"claim": claim.text, "reason": "not_entailed"})
                continue
            claim.entailment = verdicts[i]
        result.claims.append(claim)

    result.conflicts = relevant_conflicts(
        ledger.conflicts, query, result.claims, payload.get("conflicts_mentioned")
    )
    return result
