"""Clinical notes agent: grounding, conflicts, timeline, verification and routing.

Offline tests mock the LLM seam (``src.agents.notes.llm``). The TCGA ledger test
replays a hand-written extraction fixture over the real notes (skipped without
``data/``). The gold QA test is live and opt-in (GEMINI_API_KEY + NOTES_LIVE_EVAL=1).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest
from google.genai import types

import src.core.gemini_client  # noqa: F401  (loads .env so the live skip sees GEMINI_API_KEY)
import src.retrieval.case_store  # noqa: F401  (puts datactrl on sys.path)
from datactrl import Case

from src.agents.notes import agent as notes_agent
from src.agents.notes import llm
from src.agents.notes.conflicts import detect_conflicts
from src.agents.notes.grounding import locate, value_supported
from src.agents.notes.ingest import NoteDoc, build_ledger, facts_from_payload
from src.agents.notes.ledger import Ledger
from src.agents.notes.schema import Fact, normalize_attribute, normalize_value
from src.agents.notes.timeline import build_timeline
from src.agents.notes.verify import TextStore, verify_answer
from src.patients.catalog import has_case

FIXTURES = Path(__file__).parent / "fixtures"
TCGA = "TCGA-DD-A4NF"
needs_tcga = pytest.mark.skipif(not has_case(TCGA), reason="data/TCGA-DD-A4NF not available")

_COUNTER = iter(range(10_000))


def _fact(attribute, value, *, note="n1", label="note1.txt", source="physician",
          note_date="2020-01-01", category=None, assertion="present", unit=None,
          event_date=None, relative_time=None, quote=None, derived="native") -> Fact:
    attribute = normalize_attribute(attribute)
    return Fact(
        fact_id=f"F{next(_COUNTER):04d}", category=category or attribute.split(".")[0],
        attribute=attribute, value=value, unit=unit,
        normalized_value=normalize_value(attribute, value, unit),
        event_date=event_date, date_kind="explicit" if event_date else "note_date",
        relative_time=relative_time, assertion=assertion, source_type=source,
        note_id=note, note_label=label, note_date=note_date, section=None,
        char_start=0, char_end=len(quote or value), quote=quote or value, derived=derived,
    )


def _note(note_id, label, note_date, note_type="clinic"):
    return {"note_id": note_id, "label": label, "note_date": note_date,
            "note_date_confidence": "high" if note_date else "none",
            "note_date_flags": [], "note_type": note_type}


def _conflicts(facts, notes=None):
    notes = notes or []
    return detect_conflicts(facts, notes, build_timeline(facts, notes))


# ── Grounding ────────────────────────────────────────────────────────────────

def test_locate_exact_normalized_fuzzy_and_missing():
    text = "Findings:\nNo  portal\n vein thrombosis.\nHepatocellular carcinoma, moderately differentiated."
    exact = locate("Findings:", text)
    assert (exact.start, exact.end, exact.method) == (0, 9, "exact")
    norm = locate("No portal vein thrombosis.", text)
    assert norm.method == "normalized" and text[norm.start:norm.end].startswith("No  portal")
    fuzzy = locate("Hepatocelular carcinorna, moderately differentiated", text)
    assert fuzzy.method == "fuzzy" and "carcinoma" in text[fuzzy.start:fuzzy.end]
    assert locate("HCV genotype 1a", text) is None


def test_value_supported_requires_numbers_in_quote():
    assert value_supported("3.87 cm", "mass measuring 3.87 cm")
    assert not value_supported("genotype 1a", "HCV positive")


def test_facts_from_payload_drops_ungrounded_items():
    text = "IMPRESSION: LR-5 observation segment IVA, 3.9 cm.\nNo ascites."
    doc = NoteDoc("n1", "n1", "rad.txt", "2020-01-01", "high", "radiology", text)
    items = [
        {"category": "tumor", "attribute": "tumor.size", "value": "3.9", "unit": "cm",
         "assertion": "present", "source_type": "radiology_final", "quote": "segment IVA, 3.9 cm"},
        {"category": "imaging_finding", "attribute": "imaging.ascites", "value": "ascites",
         "assertion": "negated", "source_type": "radiology_final", "quote": "No ascites."},
        {"category": "lab", "attribute": "lab.hcv_genotype", "value": "1a",
         "assertion": "present", "source_type": "radiology_final", "quote": "HCV genotype 1a"},
        {"category": "tumor", "attribute": "tumor.size", "value": "5.1", "unit": "cm",
         "assertion": "present", "source_type": "radiology_final", "quote": "segment IVA, 3.9 cm"},
    ]
    facts, dropped = facts_from_payload(items, doc, text)
    assert [f.attribute for f in facts] == ["tumor.size_cm", "imaging.ascites"]
    for fact in facts:
        assert text[fact.char_start:fact.char_end] == fact.quote
    assert facts[0].normalized_value == 3.9
    assert sorted(d["reason"] for d in dropped) == ["quote_not_found", "value_not_in_quote"]


def test_value_normalization_handles_messy_nomenclature():
    assert normalize_value("tumor.size_cm", "38.7", "mm") == 3.87
    assert normalize_value("tumor.location", "Segment IVA", None) == "segment 4a"
    assert normalize_value("tumor.location", "segment for a", None) == "segment 4a"
    assert normalize_value("tumor.margin_status", "R0 per surgeon", None) == "negative"
    assert normalize_value("tumor.grade", "moderately differentiated", None) == "G2"
    assert normalize_value("rt.dose_gy", "5100", "cGy") == 51.0
    assert normalize_attribute("tumour.location") == "tumor.location"


# ── Conflicts ────────────────────────────────────────────────────────────────

def test_segment_discordance_prefers_pathology():
    facts = [
        _fact("tumor.location", "Segment IVA", source="radiology_final", label="mri.txt", note="r"),
        _fact("tumor.location", "Liver, segment 8", source="pathology", label="path.pdf", note="p"),
    ]
    (conflict,) = _conflicts(facts)
    assert conflict["kind"] == "discordant" and conflict["severity"] == "high"
    assert conflict["preferred"]["note_label"] == "path.pdf"


def test_measurement_variance_is_not_discordance():
    facts = [
        _fact("tumor.size_cm", "3.87", unit="cm", source="radiology_final"),
        _fact("tumor.size_cm", "3.9 cm", source="radiology_final"),
        _fact("tumor.size_cm", "4", unit="centimeter", source="radiology_prelim"),
    ]
    kinds = [c["kind"] for c in _conflicts(facts)]
    assert kinds == ["measurement_variance"]


def test_wording_and_staging_systems_do_not_conflict():
    facts = [
        _fact("tumor.histology", "HCC", source="operative"),
        _fact("tumor.histology", "Hepatocellular carcinoma NOS (8170/3)", source="operative"),
        _fact("tumor.histology", "moderately differentiated hepatocellular carcinoma", source="pathology"),
        _fact("tumor.stage", "Stage I"),
        _fact("tumor.stage", "T1bN0M0"),
        _fact("tumor.stage", "LR-5", source="radiology_final"),
        _fact("tumor.margin_status", "negative for tumor", assertion="negated", source="pathology"),
        _fact("tumor.margin_status", "R0 per surgeon", source="operative"),
        _fact("tumor.location", "new hepatic lesion, local recurrence"),
        _fact("tumor.location", "segment 8", source="pathology"),
    ]
    assert _conflicts(facts) == []


def test_assertion_conflict_and_age_severity():
    facts = [
        _fact("imaging.ascites", "ascites", assertion="present", label="a.txt"),
        _fact("imaging.ascites", "ascites", assertion="negated", label="b.txt", source="radiology_final"),
        _fact("demographics.age", "72"),
        _fact("demographics.age", "71"),
    ]
    by_attr = {c["attribute"]: c for c in _conflicts(facts)}
    assert by_attr["imaging.ascites"]["kind"] == "assertion_conflict"
    assert by_attr["demographics.age"]["severity"] == "low"


# ── Timeline ─────────────────────────────────────────────────────────────────

def test_relative_date_anchored_to_surgery():
    surgery = _fact("procedure.surgery", "surgical resection", category="procedure", event_date="2011",
                    quote="s/p surgical resection 2011")
    recurrence = _fact("outcome.recurrence", "recurrence", category="outcome",
                       relative_time="~22 mo post-op", note_date=None)
    events = build_timeline([surgery, recurrence], [])
    event = next(e for e in events if e.get("fact_id") == recurrence.fact_id)
    assert event["date"] == "2012-11" and event["date_kind"] == "inferred"
    assert event["anchor_fact_id"] == surgery.fact_id and "approximate" in event["flags"]


def test_unanchored_relative_date_stays_relative():
    recurrence = _fact("outcome.recurrence", "recurrence", category="outcome",
                       relative_time="~22 mo post-op", note_date="2013-05-01")
    (event,) = build_timeline([recurrence], [])
    assert event["date_kind"] == "relative" and event["date"] == "2013-05-01"


def test_outlier_note_date_is_flagged_as_conflict():
    notes = [_note("a", "a.txt", "2011-03-14"), _note("b", "b.txt", "2012-06-01"),
             _note("c", "c.txt", "2013-04-02"), _note("r", "mri.txt", "2004-02-24", "radiology")]
    conflicts = _conflicts([], notes)
    assert [(c["kind"], c["values"][0]["note_label"]) for c in conflicts] == [("date_inconsistency", "mri.txt")]


# ── Verification ─────────────────────────────────────────────────────────────

def _small_ledger():
    text = "Assessment: Stage I (T1bN0M0) squamous cell carcinoma.\nTotal dose delivered: 51 Gy."
    fact = _fact("rt.dose_gy", "51 Gy", note="n1", label="rt.txt", quote="Total dose delivered: 51 Gy.")
    fact.char_start = text.index(fact.quote)
    fact.char_end = fact.char_start + len(fact.quote)
    ledger = Ledger("P1", "test", "now", notes=[_note("n1", "rt.txt", "2006-02-13")], facts=[fact])
    return ledger, TextStore(ledger, {"n1": text}.get), text, fact


def test_verify_keeps_grounded_drops_fabricated_and_records_abstentions(monkeypatch):
    ledger, texts, text, fact = _small_ledger()
    monkeypatch.setattr(llm, "generate_json", lambda prompt, **_: {"results": [
        {"id": 0, "verdict": "supported"}, {"id": 1, "verdict": "unsupported"}]})
    payload = {"claims": [
        {"text": "51 Gy was delivered.", "evidence": [{"fact_id": fact.fact_id}]},
        {"text": "Stage I disease.", "evidence": [{"note_id": "n1", "quote": "Stage I (T1bN0M0)"}]},
        {"text": "HPV positive.", "evidence": [{"note_id": "n1", "quote": "p16 positive by IHC"}]},
        {"text": "HPV status is not documented.", "status": "not_documented", "evidence": []},
    ]}
    result = verify_answer(payload, ledger, texts, "dose?", model="m")
    assert [c.text for c in result.claims] == ["51 Gy was delivered."]
    assert result.claims[0].evidence[0].quote == text[fact.char_start:fact.char_end]
    assert {d["reason"] for d in result.dropped} == {"not_entailed", "no_grounded_evidence"}
    assert result.abstentions == ["HPV status is not documented."]


def test_verify_without_entailment_is_deterministic(monkeypatch):
    monkeypatch.setenv("NOTES_ENTAILMENT", "0")
    ledger, texts, _, _ = _small_ledger()
    payload = {"claims": [{"text": "Stage I.", "evidence": [{"note_id": "n1", "quote": "stage i (t1bn0m0)"}]}]}
    result = verify_answer(payload, ledger, texts, "stage?", model="m")
    assert result.claims[0].entailment == "unchecked"
    assert result.claims[0].evidence[0].method == "normalized"


# ── Routing: direct and navigate (synthetic charts, mocked LLM) ─────────────

def _write_chart(root: Path, n_notes: int, needle_at: int | None = None) -> None:
    notes = root / "notes"
    notes.mkdir(parents=True)
    for i in range(n_notes):
        day = f"2018-{1 + i % 12:02d}-{1 + i % 28:02d}"
        body = (
            f"PROGRESS NOTE\nDate: {day}\n\nHPI:\nFollow-up visit {i}. Patient reports stable fatigue, "
            "no new pain. Tolerating therapy.\n\nPlan:\nContinue current management and repeat imaging "
            "in three months. " + "Routine counselling provided. " * 12 + "\n"
        )
        if i == needle_at:
            body = (
                f"PATHOLOGY ADDENDUM\nDate: {day}\n\nMolecular:\nEGFR exon 19 deletion detected by NGS.\n"
                "PD-L1 TPS 5%.\n"
            )
        (notes / f"{i:03d}-note.txt").write_text(body)


def test_direct_mode_answer_is_verified_and_cited(tmp_path, monkeypatch):
    _write_chart(tmp_path, 3, needle_at=1)
    case = Case.open(tmp_path)
    ledger = notes_agent._empty_ledger("SYN", case)
    needle = next(n for n in ledger.notes if n["label"].startswith("001"))
    monkeypatch.setenv("NOTES_ENTAILMENT", "0")

    def fake_json(prompt, **_):
        assert "Question: EGFR status?" in prompt and "EGFR exon 19" in prompt
        return {"claims": [{"text": "EGFR exon 19 deletion was detected.", "status": "supported",
                            "evidence": [{"note_id": needle["note_id"], "quote": "EGFR exon 19 deletion detected by NGS."}]}]}

    monkeypatch.setattr(llm, "generate_json", fake_json)
    result = notes_agent.answer_question("SYN", "EGFR status?", case=case, ledger=ledger)
    assert result.mode == "direct"
    assert result.summary.startswith("EGFR exon 19 deletion was detected [1].")
    (row,) = result.evidence
    assert row["label"] == needle["label"] and row["n"] == 1
    assert row["span_url"].startswith("/api/cases/SYN/notes/span?id=")


def _model_turn(*, call=None, text=None):
    part = types.Part(function_call=types.FunctionCall(name=call[0], args=call[1])) if call else types.Part(text=text)
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=[part]))]
    )


def test_navigate_mode_scales_to_300_notes_with_recursive_extraction(tmp_path, monkeypatch):
    _write_chart(tmp_path, 300, needle_at=237)
    case = Case.open(tmp_path)
    assert len(case.report.notes) == 300 and case.notes_total_chars() > 150_000
    ledger = notes_agent._empty_ledger("SYN", case)
    monkeypatch.setenv("NOTES_ENTAILMENT", "0")
    turns = []

    def fake_tools(model, contents, tool, system):
        turns.append(contents[-1])
        step = len(turns)
        if step == 1:
            return _model_turn(call=("search_notes", {"query": "EGFR mutation molecular"}))
        response = contents[-1].parts[0].function_response.response
        if step == 2:
            top = response["hits"][0]
            assert "EGFR" in top["snippet"]
            return _model_turn(call=("recurse", {"sub_question": "EGFR status", "note_ids": [top["note_id"]]}))
        fact = response["facts"][0]
        return _model_turn(text=json.dumps({"claims": [{
            "text": "EGFR exon 19 deletion was detected by NGS.", "status": "supported",
            "evidence": [{"fact_id": fact["fact_id"], "note_id": fact["note_id"], "quote": fact["quote"]}]}]}))

    def fake_extract(prompt, **kwargs):
        assert "Focus: only extract facts relevant to: EGFR status" in prompt
        return {"facts": [{"category": "pathology", "attribute": "pathology.egfr", "value": "exon 19 deletion",
                           "assertion": "present", "source_type": "pathology",
                           "quote": "EGFR exon 19 deletion detected by NGS."}]}

    monkeypatch.setattr(llm, "generate_with_tools", fake_tools)
    monkeypatch.setattr(llm, "generate_json", fake_extract)
    result = notes_agent.answer_question("SYN", "What is the EGFR status?", case=case, ledger=ledger)

    assert result.mode == "navigate"
    assert result.stats["tool_calls"] == ["search_notes", "recurse"]
    (row,) = result.evidence
    assert row["label"].startswith("237") and row["fact_id"]
    text = case.fetch(row["note_id"], view="text")["data"]["text"]
    assert text[row["start"]:row["end"]] == "EGFR exon 19 deletion detected by NGS."


# ── TCGA ledger replay (real notes, fixture extraction) ─────────────────────

@needs_tcga
def test_tcga_ledger_replay_grounds_facts_and_finds_conflicts(monkeypatch):
    from src.retrieval.case_store import get_case

    extraction = json.loads((FIXTURES / "tcga_extraction.json").read_text())
    monkeypatch.setenv("NOTES_ADJUDICATE", "0")

    def fake_json(prompt, **_):
        label = re.search(r"Document: (.+?) \(id ", prompt).group(1)
        return {"facts": extraction.get(label, [])}

    monkeypatch.setattr(llm, "generate_json", fake_json)
    case = get_case(TCGA)
    ledger = build_ledger(TCGA, case, ocr=False)

    texts = TextStore(ledger, lambda doc_id: case.fetch(doc_id, view="text")["data"]["text"])
    for fact in ledger.facts:
        assert texts.get(fact.note_id)[fact.char_start:fact.char_end] == fact.quote
    assert any(d.get("reason") == "quote_not_found" and "genotype" in json.dumps(d) for d in ledger.dropped)

    kinds = {(c["attribute"], c["kind"]) for c in ledger.conflicts}
    assert ("tumor.location", "discordant") in kinds
    assert ("tumor.size_cm", "discordant") in kinds
    assert ("note.date", "date_inconsistency") in kinds
    assert not any(a == "tumor.histology" and k == "discordant" for a, k in kinds)

    recurrence = [e for e in ledger.timeline if e.get("attribute", "").startswith(("outcome.recurrence", "recurrence"))
                  and e["date_kind"] == "inferred"]
    assert recurrence and all(e["date"].startswith("2012") for e in recurrence)


# ── Live gold QA (opt-in) ────────────────────────────────────────────────────

GOLD = json.loads((FIXTURES / "notes_gold.json").read_text())
live = pytest.mark.skipif(
    not (os.getenv("GEMINI_API_KEY") and os.getenv("NOTES_LIVE_EVAL") == "1"),
    reason="live eval: set GEMINI_API_KEY and NOTES_LIVE_EVAL=1",
)


def _matches(expected: str, summary: str) -> bool:
    """Substring match, or a regex when prefixed with 're:'."""
    if expected.startswith("re:"):
        return re.search(expected[3:], summary) is not None
    return expected in summary


def test_gold_fixture_is_well_formed():
    assert len(GOLD) >= 30 and len({g["id"] for g in GOLD}) == len(GOLD)
    assert {g["type"] for g in GOLD} >= {"fact", "temporal", "negation", "conflict", "abstention"}
    for item in GOLD:
        for alternatives in item["expect"]:
            for expected in alternatives:
                if expected.startswith("re:"):
                    re.compile(expected[3:])


@live
@pytest.mark.parametrize("item", GOLD, ids=[g["id"] for g in GOLD])
def test_gold_live(item):
    if not has_case(item["patient_id"]):
        pytest.skip(f"{item['patient_id']} not available")
    result = notes_agent.answer_question(item["patient_id"], item["question"])
    summary = result.summary.lower()
    for alternatives in item["expect"]:
        assert any(_matches(a, summary) for a in alternatives), f"none of {alternatives} in: {result.summary}"
    if item.get("expect_abstain"):
        assert result.abstentions, result.summary
    if item.get("expect_conflict"):
        assert item["expect_conflict"] in {c["attribute"] for c in result.conflicts}, result.summary
    for row in result.evidence:
        assert row["quote"] and row["end"] > row["start"]
