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

    note = next(n for n in case.report.notes if n.path.endswith(".txt"))
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
