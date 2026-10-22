from pathlib import Path

import pytest

from datactrl import Case
from datactrl.errors import LimitExceeded, UnknownView
from datactrl.limits import MAX_TABLE_LIMIT
from datactrl.tools import dispatch, reset_sessions

FIXTURE = Path(__file__).parent / "fixtures" / "messy"


def test_discover_messy_tree():
    case = Case.open(FIXTURE)
    report = case.report
    assert case.case_id.startswith("case_")
    assert len(report.structured) >= 2
    assert len(report.notes) >= 2  # txt + prose json
    assert any(item["path"].endswith("readme.bin") for item in report.unknown)
    assert report.dicom == []
    # Namespaced ids
    assert all(ref.id.startswith("note:") for ref in report.notes)
    assert all(ref.id.startswith("structured:") for ref in report.structured)


def test_unified_describe_fetch_and_bounds():
    case = Case.open(FIXTURE)
    table = case.report.structured[0]
    described = case.describe(table.id)
    assert "rows" in described["views"]
    assert described["summary"]["columns"]

    rows = case.fetch(table.id, view="rows", limit=1)
    assert rows["view"] == "rows"
    assert rows["bounds"]["limit"] == 1
    assert len(rows["data"]["rows"]) == 1
    assert rows["bounds"]["truncated"] is True

    note = next(n for n in case.report.notes if n.path.endswith("clinic_note.txt"))
    preview = case.fetch(note.id, view="text", max_chars=500)
    assert "fatigue" in preview["data"]["text"].lower()


def test_limit_exceeded_rejects():
    case = Case.open(FIXTURE)
    table = case.report.structured[0]
    with pytest.raises(LimitExceeded):
        case.fetch(table.id, view="rows", limit=MAX_TABLE_LIMIT + 1)


def test_unknown_view():
    case = Case.open(FIXTURE)
    note = case.report.notes[0]
    with pytest.raises(UnknownView):
        case.fetch(note.id, view="rows")


def test_asserted_link_session_only():
    case = Case.open(FIXTURE)
    note = case.report.notes[0]
    table = case.report.structured[0]
    link = case.assert_link(note.id, table.id, relation="mentions")
    assert link["source"] == "asserted"
    described = case.describe(note.id)
    assert any(
        item["to_id"] == table.id and item["source"] == "asserted"
        for item in described["links"]
    )


def test_tool_dispatch_discover_and_fetch():
    reset_sessions()
    payload = dispatch("discover", {"root": str(FIXTURE)})
    assert "structured" in payload
    assert "case_id" in payload
    assert "note:" in payload or "structured:" in payload

    import json

    report = json.loads(payload)
    table_id = report["structured"][0]["id"]
    fetched = json.loads(
        dispatch(
            "fetch",
            {"root": str(FIXTURE), "resource_id": table_id, "view": "rows", "limit": 1},
        )
    )
    assert fetched["view"] == "rows"
    assert len(fetched["data"]["rows"]) == 1


def test_inventory_is_compact():
    case = Case.open(FIXTURE)
    inventory = case.report.to_inventory()
    assert "resources" not in inventory
    assert inventory["counts"]["structured"] >= 2
    assert inventory["counts"]["notes"] >= 2
    assert all("id" in item and "label" in item for item in inventory["notes"])


def _hep_note(case):
    return next(n for n in case.report.notes if n.label == "2011-03-14-hep-clinic.txt")


def test_note_meta_date_type_quality():
    case = Case.open(FIXTURE)
    note = _hep_note(case)
    assert note.meta["note_date"] == "2011-03-14"
    assert note.meta["note_date_confidence"] == "high"  # header "(?)" but filename agrees
    assert "header_date_flagged_uncertain" in note.meta["note_date_flags"]
    assert note.meta["note_type"] == "clinic"
    assert note.meta["text_quality"] == "good"
    inventory = case.report.to_inventory()
    row = next(item for item in inventory["notes"] if item["id"] == note.id)
    assert row["note_date"] == "2011-03-14"


def test_note_sections_and_span_offsets():
    case = Case.open(FIXTURE)
    note = _hep_note(case)
    sections = case.fetch(note.id, view="sections")["data"]["sections"]
    names = [s["name"] for s in sections]
    assert names[0] == "header"
    assert {"chief_complaint", "hpi", "labs", "impression", "assessment_plan"} <= set(names)
    assert sections[-1]["end"] == case.fetch(note.id, view="sections")["data"]["char_count"]

    full = case.fetch(note.id, view="text")["data"]["text"]
    start = full.index("No PVT")
    span = case.fetch(note.id, view="span", start=start, end=start + 6, context_chars=10)
    assert span["data"]["text"] == "No PVT"
    assert span["data"]["section"] == "impression"
    assert len(span["data"]["context_before"]) == 10


def test_note_span_bounds():
    from datactrl.errors import InvalidFetchOptions
    from datactrl.limits import MAX_SPAN_CHARS, MAX_SPAN_CONTEXT_CHARS

    case = Case.open(FIXTURE)
    note = _hep_note(case)
    with pytest.raises(InvalidFetchOptions):
        case.fetch(note.id, view="span", start=20, end=10)
    with pytest.raises(LimitExceeded):
        case.fetch(note.id, view="span", start=0, end=MAX_SPAN_CHARS + 1)
    with pytest.raises(LimitExceeded):
        case.fetch(
            note.id, view="span", start=0, end=5, context_chars=MAX_SPAN_CONTEXT_CHARS + 1
        )
    with pytest.raises(UnknownView):
        case.fetch(note.id, view="page")


def test_notes_search_returns_cited_snippets():
    case = Case.open(FIXTURE)
    note = _hep_note(case)
    result = case.notes_search("portal vein thrombosis", limit=3)
    top = result["hits"][0]
    assert top["note_id"] == note.id
    assert top["section"] == "impression"
    full = case.fetch(note.id, view="text")["data"]["text"]
    assert full[top["snippet_start"] : top["snippet_end"]] == top["snippet"]
    with pytest.raises(LimitExceeded):
        case.notes_search("x", limit=10_000)

    import json

    reset_sessions()
    payload = json.loads(dispatch("notes_search", {"root": str(FIXTURE), "query": "AFP"}))
    assert payload["hits"] and payload["hits"][0]["section"] == "labs"


_TCGA = Path(__file__).resolve().parents[1] / "data" / "TCGA-DD-A4NF"


@pytest.mark.skipif(not _TCGA.is_dir(), reason="local data/ case not present")
def test_pdf_note_quality_and_page_view():
    case = Case.open(_TCGA)
    pdf = next(n for n in case.report.notes if n.meta["format"] == "pdf")
    assert pdf.meta["note_type"] == "pathology"
    assert pdf.meta["text_quality"] in {"mixed", "poor"}
    page = case.fetch(pdf.id, view="page", page_index=0)["data"]
    assert page["mime_type"] == "application/pdf" and page["pdf_base64"]
    assert "page" in case.describe(pdf.id)["views"]


def test_gemini_function_declarations_shape():
    from datactrl.tools import gemini_function_declarations

    decls = gemini_function_declarations()
    names = {item["name"] for item in decls}
    assert {"discover", "describe", "fetch", "dicom_groups"} <= names
    for item in decls:
        assert "parameters" in item
        assert "type" not in item or item.get("type") != "function"
