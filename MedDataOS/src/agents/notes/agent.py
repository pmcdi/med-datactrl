"""Query-time clinical notes agent: size-adaptive direct / navigate modes.

Direct mode (chart fits NOTES_DIRECT_BUDGET): one call over offset-free note
text plus the compiled ledger digest, timeline and conflicts.

Navigate mode (large charts): a bounded tool loop over the ledger, BM25 search,
span/section reads and a depth-1 ``recurse`` sub-call (RLM-style) that extracts
grounded facts from a batch of notes for a sub-question.

Both modes return the same claim contract, which ``verify`` checks.
"""

from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from src.agents.notes import llm
from src.agents.notes.ingest import NoteDoc, doc_text, extract_doc, get_ledger
from src.agents.notes.ledger import Ledger
from src.agents.notes.schema import SOURCE_RELIABILITY, Fact
from src.agents.notes.verify import Claim, Evidence, TextStore, VerifiedAnswer, verify_answer
from src.retrieval.urls import note_span_url

NOTES_DIRECT_BUDGET = int(os.getenv("NOTES_DIRECT_BUDGET", "150000"))
NAV_MAX_STEPS = 8
RECURSE_MAX_CALLS = 3
RECURSE_MAX_NOTES = 12
RECURSE_MAX_CHARS = 80_000
MAX_TOOL_RESULT_ITEMS = 60

_CONTRACT = """Return ONLY JSON:
{"claims": [{"text": "<one sentence>", "status": "supported" | "not_documented",
             "evidence": [{"fact_id": "<F... or null>", "note_id": "<note id>", "quote": "<verbatim text from that note>"}]}],
 "conflicts_mentioned": ["<C... ids you discussed>"]}
Rules:
- Every supported claim needs at least one evidence item whose quote is copied VERBATIM from the cited note (or cite a ledger fact_id).
- Be complete: cover every part of the question and every distinct documented item it asks for (e.g. all risk factors, all sites, all treatments), not just the most prominent one.
- Include specific values, units and dates. Say whether a date is inferred or uncertain.
- For questions about the patient's current state ("today", "now", "current age"), give the last documented value with its date and the documented vital status; never extrapolate past the record.
- Negated findings are findings: state them explicitly ("no portal vein thrombosis").
- When part of the question is not answered by the notes, add a claim with status "not_documented" and no evidence.
- When sources disagree on something the question touches, write a claim for each side with its own evidence and name the more reliable source.
- Never put fact, note or conflict ids in claim text; they belong in evidence / conflicts_mentioned.
- Never state anything that is not in the notes. 3-8 claims."""

_DIRECT_PROMPT = """You are a clinical notes specialist answering a clinician's question from a patient's notes.

Question: {query}

NOTES (each block header gives the note id to cite):
{notes}

LEDGER (pre-extracted, source-grounded facts; cite fact ids when they support a claim):
{ledger}

TIMELINE (inferred dates are approximate):
{timeline}

DETECTED CONFLICTS (deterministic; mention those relevant to the question):
{conflicts}

{contract}"""

_NAV_SYSTEM = """You are a clinical notes specialist working over a patient chart that is too large to read at once.
Use the tools to find evidence, then answer. Start from the ledger (compiled grounded facts), the timeline and the conflicts; use search_notes and read_span/read_section to read source text; use recurse to extract facts for a focused sub-question from a batch of notes you have not read.
You have at most {steps} tool turns. When you have enough evidence, stop calling tools and reply with the final JSON.

"""

_NAV_START = """Question: {query}

Chart: {n_notes} notes, {n_chars} characters, {n_facts} compiled facts, {n_conflicts} detected conflicts.
Notes (id | label | date | type):
{inventory}"""

_TOOLS = [
    {
        "name": "ledger_query",
        "description": "Filter compiled facts by category, attribute prefix (e.g. tumor., lab.afp) and/or free-text terms.",
        "parameters": {"type": "object", "properties": {
            "category": {"type": "string"}, "attribute": {"type": "string"},
            "text": {"type": "string"}, "limit": {"type": "integer"}}},
    },
    {
        "name": "timeline",
        "description": "Ordered clinical events (optionally between from_date and to_date, YYYY or YYYY-MM-DD).",
        "parameters": {"type": "object", "properties": {
            "from_date": {"type": "string"}, "to_date": {"type": "string"},
            "limit": {"type": "integer"}}},
    },
    {
        "name": "conflicts",
        "description": "Deterministically detected discordances, assertion conflicts and date inconsistencies.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "search_notes",
        "description": "BM25 search over note sections; returns note ids, sections, char offsets and snippets.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["query"]},
    },
    {
        "name": "read_span",
        "description": "Read exact note text between character offsets (max 8000 chars).",
        "parameters": {"type": "object", "properties": {
            "note_id": {"type": "string"}, "start": {"type": "integer"}, "end": {"type": "integer"}},
            "required": ["note_id", "start", "end"]},
    },
    {
        "name": "read_section",
        "description": "Read one named section of a note (e.g. impression, assessment_plan).",
        "parameters": {"type": "object", "properties": {
            "note_id": {"type": "string"}, "section": {"type": "string"}},
            "required": ["note_id", "section"]},
    },
    {
        "name": "recurse",
        "description": (
            f"Sub-call: extract grounded facts relevant to sub_question from up to {RECURSE_MAX_NOTES} notes. "
            f"Use for notes you have not read. Limited to {RECURSE_MAX_CALLS} calls."
        ),
        "parameters": {"type": "object", "properties": {
            "sub_question": {"type": "string"},
            "note_ids": {"type": "array", "items": {"type": "string"}}},
            "required": ["sub_question", "note_ids"]},
    },
]


@dataclass
class NotesResult:
    """What the clinical notes agent returns to the orchestrator."""

    summary: str
    evidence: list[dict[str, Any]] = field(default_factory=list)
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    abstentions: list[str] = field(default_factory=list)
    mode: str = "direct"
    stats: dict[str, Any] = field(default_factory=dict)


def _empty_ledger(patient_id: str, case) -> Ledger:
    notes = [
        {
            "note_id": ref.id,
            "label": ref.label,
            "note_date": ref.meta.get("note_date"),
            "note_date_confidence": ref.meta.get("note_date_confidence"),
            "note_date_flags": ref.meta.get("note_date_flags", []),
            "note_type": ref.meta.get("note_type"),
        }
        for ref in case.report.notes
    ]
    return Ledger(patient_id, "", datetime.now(timezone.utc).isoformat(timespec="seconds"), notes=notes)


def _note_blocks(ledger: Ledger, texts: TextStore) -> str:
    blocks = []
    for doc_id in texts.doc_ids():
        text = texts.get(doc_id)
        if text is None:
            continue
        doc = ledger.docs.get(doc_id) or {}
        meta = ledger.note_meta(doc_id) or {}
        label = doc.get("label") or meta.get("label") or doc_id
        ocr = " | OCR transcript of scanned page" if doc.get("derived") == "ocr" else ""
        blocks.append(
            f"=== {doc_id} | {label} | date {meta.get('note_date') or 'unknown'} "
            f"({meta.get('note_date_confidence') or 'none'}) | {meta.get('note_type') or 'unknown'}{ocr} ===\n{text}"
        )
    return "\n\n".join(blocks)


class NavSession:
    """Tool executor for navigate mode (bounded, patient-scoped)."""

    def __init__(self, case, ledger: Ledger, texts: TextStore):
        """Bind tools to one patient's case, ledger and text store."""
        self.case = case
        self.ledger = ledger
        self.texts = texts
        self.extra_facts: dict[str, Fact] = {}
        self.recurse_calls = 0
        self.log: list[str] = []

    def run(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """Execute one tool call; errors are returned, not raised."""
        self.log.append(name)
        try:
            handler = getattr(self, f"_tool_{name}")
        except AttributeError:
            return {"error": f"Unknown tool {name}"}
        try:
            return handler(**args)
        except Exception as exc:  # noqa: BLE001 - surface tool errors to the model
            return {"error": f"{type(exc).__name__}: {exc}"}

    @staticmethod
    def _fact_row(fact: Fact) -> dict[str, Any]:
        return {
            "fact_id": fact.fact_id, "attribute": fact.attribute, "value": fact.value,
            "unit": fact.unit, "assertion": fact.assertion, "date": fact.event_date or fact.note_date,
            "relative_time": fact.relative_time, "source_type": fact.source_type,
            "note_id": fact.note_id, "note_label": fact.note_label, "quote": fact.quote,
        }

    def _tool_ledger_query(self, category=None, attribute=None, text=None, limit=30):
        limit = max(1, min(int(limit or 30), MAX_TOOL_RESULT_ITEMS))
        facts = self.ledger.query(category=category, attribute=attribute, text=text, limit=limit)
        return {"facts": [self._fact_row(f) for f in facts], "count": len(facts)}

    def _tool_timeline(self, from_date=None, to_date=None, limit=40):
        limit = max(1, min(int(limit or 40), MAX_TOOL_RESULT_ITEMS))
        events = [
            {k: e.get(k) for k in ("date", "date_kind", "label", "fact_id", "note_id", "flags")}
            for e in self.ledger.timeline
            if (not from_date or (e.get("date") or "") >= from_date)
            and (not to_date or (e.get("date") or "9999") <= to_date)
        ]
        return {"events": events[:limit], "total": len(events)}

    def _tool_conflicts(self):
        return {"conflicts": [
            {k: c[k] for k in ("conflict_id", "attribute", "kind", "severity", "rationale")}
            | {"values": [{k: v.get(k) for k in ("value", "fact_id", "note_label", "note_date", "source_type")}
                          for v in c["values"]]}
            for c in self.ledger.conflicts
        ]}

    def _tool_search_notes(self, query, limit=8):
        result = self.case.notes_search(query, limit=max(1, min(int(limit or 8), 12)))
        return {"hits": [
            {k: h[k] for k in ("note_id", "label", "note_date", "section", "start", "end", "snippet")}
            for h in result["hits"]
        ]}

    def _tool_read_span(self, note_id, start, end):
        start, end = int(start), int(end)
        end = min(end, start + 8_000)
        text = self.texts.get(note_id)
        if text is None:
            return {"error": f"Unknown note {note_id}"}
        return {"note_id": note_id, "start": start, "end": min(end, len(text)), "text": text[start:end]}

    def _tool_read_section(self, note_id, section):
        if "#" in note_id:
            return self._tool_read_span(note_id, 0, 8_000)
        sections = self.case.fetch(note_id, view="sections")["data"]["sections"]
        wanted = section.strip().lower().replace(" ", "_")
        match = next((s for s in sections if s["name"] == wanted), None) or next(
            (s for s in sections if wanted in s["name"]), None
        )
        if match is None:
            return {"error": f"No section {section!r}", "sections": [s["name"] for s in sections]}
        return self._tool_read_span(note_id, match["start"], match["end"])

    def _tool_recurse(self, sub_question, note_ids):
        if self.recurse_calls >= RECURSE_MAX_CALLS:
            return {"error": f"recurse limit ({RECURSE_MAX_CALLS}) reached; answer with what you have"}
        self.recurse_calls += 1
        docs: list[NoteDoc] = []
        budget = RECURSE_MAX_CHARS
        for doc_id in list(dict.fromkeys(note_ids))[:RECURSE_MAX_NOTES]:
            text = self.texts.get(doc_id)
            if text is None or budget <= 0:
                continue
            text = text[:budget]
            budget -= len(text)
            meta = self.ledger.note_meta(doc_id) or {}
            sections = [] if "#" in doc_id else self.case.fetch(doc_id, view="sections")["data"]["sections"]
            docs.append(NoteDoc(
                doc_id=doc_id, note_id=doc_id.split("#", 1)[0],
                label=(self.ledger.docs.get(doc_id) or {}).get("label") or meta.get("label", doc_id),
                note_date=meta.get("note_date"), date_confidence=meta.get("note_date_confidence") or "none",
                note_type=meta.get("note_type") or "unknown", text=text, sections=sections,
                derived=(self.ledger.docs.get(doc_id) or {}).get("derived", "native"),
            ))
        if not docs:
            return {"error": "No readable notes among note_ids"}
        with ThreadPoolExecutor(max_workers=min(8, len(docs))) as pool:
            results = list(pool.map(lambda d: extract_doc(d, focus=sub_question), docs))
        rows = []
        for facts, _ in results:
            for fact in facts:
                self.extra_facts[fact.fact_id] = fact
                rows.append(self._fact_row(fact))
        return {"sub_question": sub_question, "notes_read": len(docs), "facts": rows[:MAX_TOOL_RESULT_ITEMS]}


def _navigate(query: str, case, ledger: Ledger, texts: TextStore, model: str) -> tuple[dict, NavSession]:
    from google.genai import types

    session = NavSession(case, ledger, texts)
    inventory = "\n".join(
        f"{n['note_id']} | {n['label']} | {n.get('note_date') or 'undated'} | {n.get('note_type') or 'unknown'}"
        for n in ledger.notes[:400]
    )
    start = _NAV_START.format(
        query=query, n_notes=len(ledger.notes), n_chars=case.notes_total_chars(),
        n_facts=len(ledger.facts), n_conflicts=len(ledger.conflicts), inventory=inventory,
    )
    tool = types.Tool(function_declarations=_TOOLS)
    system = _NAV_SYSTEM.format(steps=NAV_MAX_STEPS) + _CONTRACT
    contents = [types.Content(role="user", parts=[types.Part(text=start)])]
    for _ in range(NAV_MAX_STEPS):
        response = llm.generate_with_tools(model, contents, tool, system)
        parts = response.candidates[-1].content.parts or []
        calls = [p.function_call for p in parts if p.function_call and p.function_call.name]
        if not calls:
            text = "".join(p.text or "" for p in parts if getattr(p, "text", None))
            try:
                return llm.parse_json(text), session
            except ValueError:
                break
        contents.append(response.candidates[-1].content)
        contents.append(types.Content(role="user", parts=[
            types.Part.from_function_response(
                name=call.name,
                response=json.loads(json.dumps(session.run(call.name, dict(call.args or {})), default=str)),
            )
            for call in calls
        ]))
    contents.append(types.Content(role="user", parts=[types.Part(
        text="Stop using tools. Return the final JSON answer now using the evidence gathered."
    )]))
    return llm.generate_json(contents, model=model, system=_CONTRACT), session


def _readable(attribute: str) -> str:
    return attribute.split(".", 1)[-1].replace("_cm", "").replace("_", " ")


def compose_summary(verified: VerifiedAnswer) -> str:
    """Plain-prose summary built only from verified claims, abstentions and conflicts."""
    parts = [
        claim.text.rstrip(".")
        + (" (partially verified)" if claim.entailment == "partial" else "")
        + f" [{n}]."
        for n, claim in enumerate(verified.claims, start=1)
    ]
    if verified.abstentions:
        parts.append("Not documented in the notes: " + "; ".join(a.rstrip(".") for a in verified.abstentions) + ".")
    discrepancies = []
    for conflict in verified.conflicts:
        if conflict["severity"] == "info":
            continue
        if conflict["kind"] == "date_inconsistency":
            v = conflict["values"][0]
            discrepancies.append(
                f"{v['value']} recorded on {v['note_label']} is inconsistent with the rest of the record"
            )
            continue
        values = " vs ".join(f"{v['value']} ({v['note_label']})" for v in conflict["values"])
        discrepancies.append(f"{_readable(conflict['attribute'])}: {values}")
    if discrepancies:
        parts.append("Documented discrepancies: " + "; ".join(discrepancies) + ".")
    if not parts:
        return "The notes did not yield verifiable evidence for this question."
    return " ".join(parts)


def _evidence_rows(patient_id: str, verified: VerifiedAnswer) -> list[dict[str, Any]]:
    rows = []
    for n, claim in enumerate(verified.claims, start=1):
        for ev in claim.evidence:
            row = ev.to_dict()
            row.update({
                "n": n,
                "claim": claim.text,
                "entailment": claim.entailment,
                "span_url": note_span_url(patient_id, ev.doc_id, ev.start, ev.end),
            })
            rows.append(row)
    return rows


_DATE_LINE_RE = re.compile(r"^[^\n]*\b(?:study date|date of service|dos|date)\b[^\n]*", re.I | re.M)


def _date_line_span(text: str | None) -> tuple[int, int] | None:
    """Offsets of the note's date header line (fallback: first non-empty line)."""
    if not text:
        return None
    match = _DATE_LINE_RE.search(text[:4000]) or re.search(r"\S[^\n]*", text)
    return (match.start(), match.end()) if match else None


def _conflict_rows(
    patient_id: str, conflicts: list[dict[str, Any]], texts: TextStore | None = None
) -> list[dict[str, Any]]:
    rows = []
    for conflict in conflicts:
        values = []
        for v in conflict["values"]:
            entry = {k: v.get(k) for k in (
                "value", "note_id", "note_label", "note_date", "source_type", "quote",
                "char_start", "char_end", "fact_id", "derived")}
            if v.get("char_start") is None and conflict["kind"] == "date_inconsistency" and texts and v.get("note_id"):
                text = texts.get(v["note_id"])
                span = _date_line_span(text)
                if span:
                    entry["char_start"], entry["char_end"] = span
                    entry["quote"] = text[span[0]:span[1]]
            if entry.get("char_start") is not None:
                entry["span_url"] = note_span_url(patient_id, v["note_id"], entry["char_start"], entry["char_end"])
            values.append(entry)
        rows.append({
            "conflict_id": conflict["conflict_id"],
            "attribute": conflict["attribute"],
            "kind": conflict["kind"],
            "severity": conflict["severity"],
            "rationale": conflict["rationale"],
            "preferred_fact_id": (conflict.get("preferred") or {}).get("fact_id"),
            "values": values,
        })
    return rows


_DECEASED_RE = re.compile(r"deceased|dead|died|expired|death", re.I)


def _ensure_vital_status(verified: VerifiedAnswer, ledger: Ledger) -> None:
    """A documented death is always surfaced, so no answer implies the patient is alive."""
    deaths = [
        f for f in ledger.facts
        if f.attribute in {"outcome.vital_status", "outcome.death", "outcome.date_of_death"}
        and f.assertion == "present" and _DECEASED_RE.search(f"{f.value} {f.quote}")
    ]
    if not deaths or any(_DECEASED_RE.search(c.text) for c in verified.claims):
        return
    fact = min(deaths, key=lambda f: SOURCE_RELIABILITY.get(f.source_type, 7))
    when = fact.event_date or fact.note_date
    verified.claims.append(Claim(
        text=f"The patient is documented as deceased{f' on {when}' if when else ''}",
        evidence=[Evidence(
            fact.note_id, fact.note_label, fact.note_date, fact.char_start, fact.char_end,
            fact.quote, fact.derived, fact.fact_id, "fact",
        )],
        entailment="supported",
    ))


def answer_question(patient_id: str, query: str, *, case=None, ledger: Ledger | None = None) -> NotesResult:
    """Answer a clinician's question from one patient's notes."""
    from src.core.gemini_client import NOTES_REASON_MODEL
    from src.retrieval.case_store import get_case

    case = case or get_case(patient_id)
    stats: dict[str, Any] = {}
    if ledger is None:
        try:
            ledger = get_ledger(patient_id, case)
        except Exception as exc:  # noqa: BLE001 - answer from raw notes if compile fails
            print(f"[notes] ledger build failed for {patient_id}: {exc}")
            ledger = _empty_ledger(patient_id, case)
            stats["ledger_error"] = str(exc)[:200]
    texts = TextStore(ledger, lambda doc_id: doc_text(ledger, doc_id, case))

    ocr_chars = sum(d.get("chars", 0) for d in ledger.docs.values() if d.get("derived") == "ocr")
    total = case.notes_total_chars() + ocr_chars
    mode = "direct" if total <= NOTES_DIRECT_BUDGET else "navigate"
    stats.update({"total_chars": total, "facts": len(ledger.facts), "conflicts": len(ledger.conflicts)})

    session = None
    if mode == "direct":
        prompt = _DIRECT_PROMPT.format(
            query=query,
            notes=_note_blocks(ledger, texts),
            ledger=ledger.digest() or "(no compiled facts)",
            timeline=ledger.timeline_digest() or "(empty)",
            conflicts=ledger.conflicts_digest(),
            contract=_CONTRACT,
        )
        payload = llm.generate_json(prompt, model=NOTES_REASON_MODEL)
    else:
        payload, session = _navigate(query, case, ledger, texts, NOTES_REASON_MODEL)
        stats["tool_calls"] = session.log
    if not isinstance(payload, dict):
        payload = {"claims": payload if isinstance(payload, list) else []}

    verified = verify_answer(
        payload, ledger, texts, query, model=NOTES_REASON_MODEL,
        extra_facts=session.extra_facts if session else None,
    )
    stats["dropped_claims"] = verified.dropped
    _ensure_vital_status(verified, ledger)
    return NotesResult(
        summary=compose_summary(verified),
        evidence=_evidence_rows(patient_id, verified),
        conflicts=_conflict_rows(patient_id, verified.conflicts, texts),
        abstentions=verified.abstentions,
        mode=mode,
        stats=stats,
    )
