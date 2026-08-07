"""Layer 0: smoke Case.open/discover against real local data/rawdata cases.

Skipped automatically when data/rawdata is absent (gitignored).
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from datactrl import Case
from datactrl.errors import InvalidFetchOptions

RAWDATA = Path(__file__).resolve().parents[1] / "data" / "rawdata"

# Expected modality presence for each known case under data/rawdata.
CASE_EXPECTATIONS = {
    "KiTS-00008": {"dicom": True, "notes": False},
    "STS_002": {"dicom": True, "notes": False},
    "Adrenal_Ki67_Seg_002": {"dicom": True, "notes": False},
    "TCGA-DD-A4NF": {"dicom": True, "notes": True},
}


def _available_cases() -> list[str]:
    if not RAWDATA.is_dir():
        return []
    return sorted(
        name
        for name in CASE_EXPECTATIONS
        if (RAWDATA / name).is_dir()
    )


AVAILABLE = _available_cases()


@pytest.mark.skipif(not AVAILABLE, reason="data/rawdata cases not present")
@pytest.mark.parametrize("case_name", AVAILABLE)
def test_layer0_discover_rawdata_case(case_name: str):
    root = RAWDATA / case_name
    expect = CASE_EXPECTATIONS[case_name]

    t0 = time.perf_counter()
    case = Case.open(root)
    report = case.report
    elapsed = time.perf_counter() - t0

    # Inventory should finish in a reasonable wall time for local disk.
    assert elapsed < 180, f"discover() took {elapsed:.1f}s for {case_name}"
    assert case.case_id.startswith("case_")

    if expect["dicom"]:
        assert report.dicom, f"{case_name}: expected DICOM series, found none"
        series = report.dicom[0]
        assert series.id.startswith("dicom:")
        described = case.describe(series.id)
        assert described["id"] == series.id
        assert "metadata" in described["views"]
        meta = case.fetch(series.id, view="metadata")
        assert meta["id"] == series.id
        assert meta["data"]["metadata"]

    if expect["notes"]:
        assert report.notes, f"{case_name}: expected clinical notes, found none"
        note = report.notes[0]
        assert note.id.startswith("note:")
        preview = case.fetch(note.id, view="text", max_chars=500)
        assert preview["id"] == note.id
        assert preview["data"]["text"]

    # Hidden caches must not surface as resources.
    all_paths = (
        [ref.path for ref in report.resources]
        + [item["path"] for item in report.unknown]
    )
    assert not any(part.startswith(".") for path in all_paths for part in Path(path).parts)


@pytest.mark.skipif("TCGA-DD-A4NF" not in AVAILABLE, reason="TCGA-DD-A4NF not present")
def test_layer0_tcga_notes_and_unknown_svs():
    case = Case.open(RAWDATA / "TCGA-DD-A4NF")
    report = case.report

    note_paths = {ref.path for ref in report.notes}
    assert any(path.endswith(".txt") for path in note_paths)
    assert any(".pdf" in path for path in note_paths)

    # WSI is out of v1 scope; .svs should be unknown, not misclassified.
    unknown_paths = [item["path"] for item in report.unknown]
    assert any(path.endswith(".svs") for path in unknown_paths)
    assert not any(ref.path.endswith(".svs") for ref in report.resources)


@pytest.mark.skipif("KiTS-00008" not in AVAILABLE, reason="KiTS-00008 not present")
def test_layer0_kits_dicom_references():
    """KiTS has CT↔SEG; harness should surface that link."""

    case = Case.open(RAWDATA / "KiTS-00008")
    tree = case.dicom_tree()
    assert tree["series_count"] >= 2
    assert "CT,SEG" in tree["valid_queries"]

    groups = case.dicom_groups("CT,SEG")
    assert groups["group_count"] >= 1
    group = groups["groups"][0]
    modalities = group["modalities"]
    assert "CT" in modalities and "SEG" in modalities

    seg = next(ref for ref in case.report.dicom if ref.meta.get("Modality") == "SEG")
    assert seg.id.startswith("dicom:")
    assert seg.meta.get("ReferencedSeriesUID")
    related = case.dicom_related(seg.id)
    assert related["references"]
    assert related["references"][0]["Modality"] == "CT"
    assert related["references"][0]["series_id"] == seg.meta["ReferencedSeriesUID"]
    assert related["references"][0]["id"].startswith("dicom:")

    ct_id = related["references"][0]["id"]
    ct_related = case.dicom_related(ct_id)
    assert any(child["id"] == seg.id for child in ct_related["referenced_by"])

    described = case.describe(seg.id)
    assert described["links"]
    assert any(
        link["to_id"] == ct_id and link["relation"] == "references"
        for link in described["links"]
    )


@pytest.mark.skipif("KiTS-00008" not in AVAILABLE, reason="KiTS-00008 not present")
def test_layer0_kits_preview_slice_index():
    case = Case.open(RAWDATA / "KiTS-00008")
    ct = next(ref for ref in case.report.dicom if ref.meta.get("Modality") == "CT")

    middle = case.fetch(ct.id, view="preview")
    assert middle["view"] == "preview"
    assert "image_base64" in middle["data"]
    n_slices = middle["bounds"]["n_slices"]
    assert n_slices >= 1
    mid_idx = middle["bounds"]["slice_index"]

    zero = case.fetch(ct.id, view="preview", slice_index=0)
    assert zero["bounds"]["slice_index"] == 0
    # Different slice should usually differ; at minimum indices are honored.
    assert mid_idx == n_slices // 2

    with pytest.raises(InvalidFetchOptions):
        case.fetch(ct.id, view="preview", slice_index=n_slices + 5)


@pytest.mark.skipif(not AVAILABLE, reason="data/rawdata cases not present")
def test_layer0_multi_patient_root_ok():
    """Opening the whole rawdata root (multi-case dump) must not crash."""

    case = Case.open(RAWDATA)
    report = case.report
    assert report.dicom, "expected DICOM across multi-patient dump"
    # Multiple patient folders → at least one patient_id present
    patient_ids = {ref.patient_id for ref in report.dicom if ref.patient_id}
    assert patient_ids
