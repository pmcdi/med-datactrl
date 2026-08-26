"""DICOM adapter smoke test using a tiny synthetic series."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from datactrl import Case
from datactrl.dicom_index import (
    ABDOMEN_FROM_SUPERIOR_MM,
    classify_series_kind,
    parse_query_intent,
    rank_series,
    suggested_slice_index,
)
from datactrl.types import Modality, ResourceRef

pydicom = pytest.importorskip("pydicom")
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid


def _ref(key: str, **meta) -> ResourceRef:
    return ResourceRef(
        id=f"dicom:{key}",
        modality=Modality.DICOM,
        label=str(meta.get("SeriesDescription") or key),
        meta=meta,
    )


def test_classify_kind_from_image_type_and_tokens():
    assert classify_series_kind(["ORIGINAL", "PRIMARY", "LOCALIZER"], "CT") == "localizer"
    assert classify_series_kind(["ORIGINAL", "PRIMARY", "AXIAL"], "Topogram 0.6") == "localizer"
    assert classify_series_kind(["DERIVED", "SECONDARY"], "screen save") == "derived"
    assert classify_series_kind(["ORIGINAL", "PRIMARY", "AXIAL"], "AX LAVA") == "acquisition"
    assert classify_series_kind(None, "", modality="RTSTRUCT") == "rtstruct"
    assert classify_series_kind(None, "", modality="RTPLAN") == "rtplan"


def test_parse_query_intent_ct_and_hep():
    intent = parse_query_intent(
        "What did the hep clinic note say, and what does a representative CT slice show?"
    )
    assert "CT" in intent.modalities
    assert "abdomen" in intent.regions


def test_rank_prefers_liver_mr_over_runoff_ct_for_organ_query():
    refs = _runoff_fixture_refs()
    result = rank_series(refs, "liver lesion on MRI")
    assert result["confident"]
    assert result["selected"]["id"] == "dicom:lava"
    assert result["selected"]["Modality"] == "MR"


def test_rank_ct_query_skips_localizer_and_uses_superior_slice():
    refs = _runoff_fixture_refs()
    query = (
        "What did the hep clinic note say, and what does a representative CT slice show?"
    )
    result = rank_series(refs, query)
    assert result["confident"]
    selected = result["selected"]
    assert selected["id"] == "dicom:runoff5"
    assert selected["series_kind"] == "acquisition"
    assert selected["id"] != "dicom:topogram"
    assert selected["id"] != "dicom:delay"
    # Long-FOV abdomen: ~80 mm below the superior end, not 15% and not mid-thigh.
    assert selected["suggested_slice_index"] == _abdomen_index(-174.0, -1510.0, 268)
    assert selected["suggested_slice_index"] != 268 // 2
    assert "runoff" in " ".join(selected["warnings"]).lower() or any(
        "extremity" in reason or "runoff" in reason for reason in selected["reasons"]
    )
    ids = [item["id"] for item in result["candidates"]]
    assert "dicom:topogram" not in ids


def test_rank_body_part_liver_does_not_override_runoff_series_text():
    refs = _runoff_fixture_refs()
    result = rank_series(refs, "hepatic mass")
    assert result["selected"]["id"] == "dicom:lava"


def test_choose_preview_window_by_region():
    from datactrl.dicom_index import choose_preview_window

    assert choose_preview_window("MR", parse_query_intent("liver")) == "minmax"
    assert choose_preview_window("CT", parse_query_intent("liver CT")) == "liver"
    assert choose_preview_window("CT", parse_query_intent("chest CT")) == "lung"
    assert choose_preview_window("CT", parse_query_intent("representative CT")) == "abdomen"
    assert choose_preview_window("CT", parse_query_intent("larynx CT")) == "abdomen"
    assert choose_preview_window("CT", parse_query_intent("brain CT")) == "brain"


def test_parse_query_intent_neck_and_contours():
    neck = parse_query_intent("representative CT slice of the neck")
    assert "CT" in neck.modalities
    assert "neck" in neck.regions
    assert not neck.contours

    contours = parse_query_intent(
        "Which structures are contoured on the planning CT, and is there a GTV?"
    )
    assert contours.contours
    assert "CT" in contours.modalities


def test_rank_contour_query_prefers_rtstruct():
    refs = _hnc_fixture_refs()
    result = rank_series(refs, "which structures are contoured on the planning CT")
    assert result["confident"]
    assert result["selected"]["id"] == "dicom:rt"
    assert result["selected"]["series_kind"] == "rtstruct"
    assert "GTVp" in (result["selected"].get("roi_names") or [])


def test_rank_ct_slice_skips_rtstruct():
    refs = _hnc_fixture_refs()
    result = rank_series(refs, "representative CT slice of the neck")
    assert result["confident"]
    assert result["selected"]["id"] == "dicom:neck"
    assert result["selected"]["series_kind"] == "acquisition"


def test_patient_xyz_to_pixel_identity():
    from datactrl.rtstruct import patient_xyz_to_pixel

    col, row = patient_xyz_to_pixel(
        (10.0, 20.0, 5.0),
        (0.0, 0.0, 5.0),
        (1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        (1.0, 1.0),
    )
    assert col == 10.0
    assert row == 20.0


def test_suggested_slice_middle_for_short_fov():
    meta = {
        "instance_count": 80,
        "series_kind": "acquisition",
        "z_first_mm": -50.0,
        "z_last_mm": -220.0,
        "z_span_mm": 170.0,
    }
    assert suggested_slice_index(meta) == 40


def test_suggested_slice_is_mm_from_superior_on_long_fov():
    cranio_caudal = {
        "instance_count": 268,
        "series_kind": "acquisition",
        "z_first_mm": -174.0,
        "z_last_mm": -1510.0,
        "z_span_mm": 1336.0,
    }
    intent = parse_query_intent("representative CT slice of the liver")
    idx = suggested_slice_index(cranio_caudal, intent)
    assert idx == _abdomen_index(-174.0, -1510.0, 268)
    assert idx < 40  # much closer to the superior end than the old 15% pick

    # GDCM often orders feet→head; superior is then near the end of the stack.
    caudo_cranial = {
        "instance_count": 268,
        "series_kind": "acquisition",
        "z_first_mm": -1510.0,
        "z_last_mm": -174.0,
        "z_span_mm": 1336.0,
    }
    idx = suggested_slice_index(caudo_cranial, intent)
    assert idx == _abdomen_index(-1510.0, -174.0, 268)
    assert idx > 240


def _abdomen_index(z_first: float, z_last: float, n: int) -> int:
    z_sup = max(z_first, z_last)
    target = z_sup - ABDOMEN_FROM_SUPERIOR_MM
    last_i = n - 1
    return int(round((target - z_first) / (z_last - z_first) * last_i))


def _runoff_fixture_refs() -> list[ResourceRef]:
    return [
        _ref(
            "topogram",
            Modality="CT",
            SeriesDescription="Topogram 0.6 T20s",
            series_kind="localizer",
            instance_count=1,
            BodyPartExamined="LIVER",
        ),
        _ref(
            "runoff5",
            Modality="CT",
            SeriesDescription="AP Runoff 5.0 B41f",
            ProtocolName="AP_RUNOFF",
            StudyDescription="CT ANGIO ABD/PELVIS W RUNOFFS",
            BodyPartExamined="LIVER",
            series_kind="acquisition",
            instance_count=268,
            z_first_mm=-174.0,
            z_last_mm=-1510.0,
            z_span_mm=1336.0,
            z_superior_mm=-174.0,
        ),
        _ref(
            "runoff1",
            Modality="CT",
            SeriesDescription="AP Runoff 1.0 B31f",
            ProtocolName="AP_RUNOFF",
            StudyDescription="CT ANGIO ABD/PELVIS W RUNOFFS",
            BodyPartExamined="LIVER",
            series_kind="acquisition",
            instance_count=1338,
            z_first_mm=-174.0,
            z_last_mm=-1510.0,
            z_span_mm=1336.0,
            z_superior_mm=-174.0,
        ),
        _ref(
            "delay",
            Modality="CT",
            SeriesDescription="Delay 5.0 B31f",
            ProtocolName="AP_RUNOFF",
            StudyDescription="CT ANGIO ABD/PELVIS W RUNOFFS",
            BodyPartExamined="LIVER",
            series_kind="acquisition",
            instance_count=125,
            z_first_mm=-889.0,
            z_last_mm=-1510.0,
            z_span_mm=621.0,
            z_superior_mm=-889.0,
        ),
        _ref(
            "lava",
            Modality="MR",
            SeriesDescription="AX LAVA",
            ProtocolName="liver",
            StudyDescription="MR ABDOMEN W/WO CONTRAST",
            BodyPartExamined="LIVER",
            series_kind="acquisition",
            instance_count=88,
            z_first_mm=-80.0,
            z_last_mm=-280.0,
            z_span_mm=200.0,
            z_superior_mm=-80.0,
        ),
    ]


def _hnc_fixture_refs() -> list[ResourceRef]:
    return [
        _ref(
            "neck",
            Modality="CT",
            SeriesDescription="Neck 2.0 CE",
            ProtocolName="Head and Neck",
            StudyDescription="Research HNC Planning CT",
            series_kind="acquisition",
            instance_count=181,
            z_first_mm=-792.0,
            z_last_mm=-1152.0,
            z_span_mm=360.0,
            z_superior_mm=-792.0,
        ),
        _ref(
            "rt",
            Modality="RTSTRUCT",
            SeriesDescription="",
            series_kind="rtstruct",
            instance_count=1,
            roi_names=["SpinalCord", "Esophagus", "GTVp", "Larynx", "EXTERNAL"],
            preferred_roi="GTVp",
            ReferencedSeriesUID="neck-uid",
        ),
    ]


def _write_tiny_ct(
    path: Path,
    *,
    series_uid: str,
    instance_number: int,
    series_description: str = "TinyTest",
    image_type: list[str] | None = None,
    z_mm: float | None = None,
    protocol: str | None = None,
    study: str | None = None,
    body_part: str | None = None,
    study_uid: str | None = None,
    frame_of_reference_uid: str | None = None,
    sop_uid: str | None = None,
) -> str:
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    file_meta.MediaStorageSOPInstanceUID = sop_uid or generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.PatientID = "TEST"
    ds.StudyInstanceUID = study_uid or generate_uid()
    ds.SeriesInstanceUID = series_uid
    ds.Modality = "CT"
    ds.SeriesDescription = series_description
    ds.InstanceNumber = instance_number
    ds.ImageType = image_type or ["ORIGINAL", "PRIMARY", "AXIAL"]
    ds.ImagePositionPatient = [0.0, 0.0, float(z_mm if z_mm is not None else -instance_number)]
    ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    ds.PixelSpacing = [1.0, 1.0]
    ds.SliceThickness = 2.0
    if frame_of_reference_uid:
        ds.FrameOfReferenceUID = frame_of_reference_uid
    if protocol:
        ds.ProtocolName = protocol
    if study:
        ds.StudyDescription = study
    if body_part:
        ds.BodyPartExamined = body_part
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows = 8
    ds.Columns = 8
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelData = np.arange(64, dtype=np.uint16).tobytes()
    ds.save_as(path, enforce_file_format=True)
    return str(ds.SOPInstanceUID)


def _write_tiny_rtstruct(
    path: Path,
    *,
    series_uid: str,
    study_uid: str,
    frame_of_reference_uid: str,
    referenced_series_uid: str,
    referenced_sop_uid: str,
    roi_name: str = "GTVp",
    z_mm: float = -2.0,
) -> None:
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.481.3"
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.PatientID = "TEST"
    ds.StudyInstanceUID = study_uid
    ds.SeriesInstanceUID = series_uid
    ds.Modality = "RTSTRUCT"
    ds.StructureSetLabel = "Unapproved"
    ds.StructureSetDate = "20200101"
    ds.StructureSetTime = "120000"
    ds.FrameOfReferenceUID = frame_of_reference_uid

    roi = Dataset()
    roi.ROINumber = 1
    roi.ROIName = roi_name
    roi.ReferencedFrameOfReferenceUID = frame_of_reference_uid
    ds.StructureSetROISequence = [roi]

    contour_image = Dataset()
    contour_image.ReferencedSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    contour_image.ReferencedSOPInstanceUID = referenced_sop_uid

    contour = Dataset()
    contour.ContourGeometricType = "CLOSED_PLANAR"
    contour.NumberOfContourPoints = 4
    contour.ContourImageSequence = [contour_image]
    contour.ContourData = [
        1.0, 1.0, z_mm,
        6.0, 1.0, z_mm,
        6.0, 6.0, z_mm,
        1.0, 6.0, z_mm,
    ]

    roi_contour = Dataset()
    roi_contour.ReferencedROINumber = 1
    roi_contour.ROIDisplayColor = [255, 0, 0]
    roi_contour.ContourSequence = [contour]
    ds.ROIContourSequence = [roi_contour]

    rt_series = Dataset()
    rt_series.SeriesInstanceUID = referenced_series_uid
    rt_series.ContourImageSequence = [contour_image]

    rt_study = Dataset()
    rt_study.ReferencedSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    rt_study.ReferencedSOPInstanceUID = study_uid
    rt_study.RTReferencedSeriesSequence = [rt_series]

    frame = Dataset()
    frame.FrameOfReferenceUID = frame_of_reference_uid
    frame.RTReferencedStudySequence = [rt_study]
    ds.ReferencedFrameOfReferenceSequence = [frame]
    ds.save_as(path, enforce_file_format=True)


def test_dicom_discover_and_metadata(tmp_path: Path):
    series_uid = generate_uid()
    series_dir = tmp_path / "ct"
    series_dir.mkdir()
    for i in range(2):
        _write_tiny_ct(series_dir / f"img{i}.dcm", series_uid=series_uid, instance_number=i + 1)

    # Mix in a note so discovery is multimodal.
    (tmp_path / "note.txt").write_text("synthetic case")

    case = Case.open(tmp_path)
    assert len(case.report.dicom) >= 1
    assert len(case.report.notes) == 1
    assert case.report.notes[0].id.startswith("note:")

    series = case.report.dicom[0]
    assert series.id.startswith("dicom:")
    assert series.label == "TinyTest"
    assert series.meta.get("series_kind") == "acquisition"
    assert int(series.meta["instance_count"]) == 2

    inventory = case.report.to_inventory()
    row = inventory["dicom"][0]
    assert row["label"] == "TinyTest"
    assert row["series_kind"] == "acquisition"
    assert row["instance_count"] == 2
    assert isinstance(row["instance_count"], int)

    described = case.describe(series.id)
    assert described["summary"]["Modality"] == "CT"
    assert described["summary"]["SeriesDescription"] == "TinyTest"
    assert described["summary"]["instance_count"] == 2
    assert isinstance(described["summary"]["instance_count"], int)
    assert "instances" not in described["summary"]
    assert "metadata" in described["views"]
    assert "preview" in described["views"]

    meta = case.fetch(series.id, view="metadata")
    assert meta["view"] == "metadata"
    assert "metadata" in meta["data"]
    assert "instances" not in meta["data"]["metadata"]
    assert meta["data"]["metadata"]["instance_count"] == 2
    assert "links" in meta["data"]

    tree = case.dicom_tree()
    assert tree["series_count"] >= 1
    assert "patients" in tree

    related = case.dicom_related(series.id)
    assert related["id"] == series.id
    assert related["Modality"] == "CT"

    preview = case.fetch(series.id, view="preview")
    assert preview["view"] == "preview"
    assert "image_base64" in preview["data"]
    assert preview["bounds"]["n_slices"] == 2


def test_dicom_select_skips_localizer(tmp_path: Path):
    axial_uid = generate_uid()
    loc_uid = generate_uid()
    axial_dir = tmp_path / "axial"
    loc_dir = tmp_path / "scout"
    axial_dir.mkdir()
    loc_dir.mkdir()
    for i in range(3):
        _write_tiny_ct(
            axial_dir / f"img{i}.dcm",
            series_uid=axial_uid,
            instance_number=i + 1,
            series_description="AXIAL ABDOMEN",
            z_mm=-10.0 * i,
        )
    _write_tiny_ct(
        loc_dir / "scout.dcm",
        series_uid=loc_uid,
        instance_number=1,
        series_description="Topogram 0.6 T20s",
        image_type=["ORIGINAL", "PRIMARY", "LOCALIZER"],
        z_mm=0.0,
    )

    case = Case.open(tmp_path)
    result = case.dicom_select("representative CT slice of the abdomen")
    assert result["confident"]
    assert result["selected"]["label"] == "AXIAL ABDOMEN"
    assert result["selected"]["series_kind"] == "acquisition"
    kinds = {item["series_kind"] for item in result["candidates"]}
    assert "localizer" not in kinds


def test_summarize_rtstruct_lists_rois_and_prefers_gtv(tmp_path: Path):
    from datactrl.rtstruct import overlay_contours, summarize_rt_file

    study_uid = generate_uid()
    ct_uid = generate_uid()
    rt_uid = generate_uid()
    frame_uid = generate_uid()
    sop = _write_tiny_ct(
        tmp_path / "ct.dcm",
        series_uid=ct_uid,
        instance_number=1,
        study_uid=study_uid,
        frame_of_reference_uid=frame_uid,
        z_mm=-2.0,
    )
    _write_tiny_rtstruct(
        tmp_path / "rt.dcm",
        series_uid=rt_uid,
        study_uid=study_uid,
        frame_of_reference_uid=frame_uid,
        referenced_series_uid=ct_uid,
        referenced_sop_uid=sop,
        roi_name="GTVp",
        z_mm=-2.0,
    )
    summary = summarize_rt_file(tmp_path / "rt.dcm")
    assert summary["series_kind"] == "rtstruct"
    assert summary["roi_names"] == ["GTVp"]
    assert summary["preferred_roi"] == "GTVp"
    assert summary["referenced_series_uid"] == ct_uid
    assert summary["structure_set_label"] == "Unapproved"

    from PIL import Image

    blank = Image.new("RGB", (8, 8), (0, 0, 0))
    drawn, meta = overlay_contours(blank, tmp_path / "ct.dcm", tmp_path / "rt.dcm")
    assert meta["overlay"] is True
    assert meta["overlay_rois"][0]["name"] == "GTVp"
    arr = np.asarray(drawn)
    assert arr.any()


def test_rtstruct_discover_and_preview(tmp_path: Path):
    study_uid = generate_uid()
    ct_uid = generate_uid()
    rt_uid = generate_uid()
    frame_uid = generate_uid()
    ct_dir = tmp_path / "ct"
    rt_dir = tmp_path / "rt"
    ct_dir.mkdir()
    rt_dir.mkdir()
    sop = None
    for i in range(3):
        written = _write_tiny_ct(
            ct_dir / f"img{i}.dcm",
            series_uid=ct_uid,
            instance_number=i + 1,
            series_description="Neck 2.0 CE",
            protocol="Head and Neck",
            study_uid=study_uid,
            frame_of_reference_uid=frame_uid,
            z_mm=-1.0 * (i + 1),
        )
        if i == 1:
            sop = written
    _write_tiny_rtstruct(
        rt_dir / "struct.dcm",
        series_uid=rt_uid,
        study_uid=study_uid,
        frame_of_reference_uid=frame_uid,
        referenced_series_uid=ct_uid,
        referenced_sop_uid=sop or written,
        roi_name="GTVp",
        z_mm=-2.0,
    )
    (tmp_path / "note.txt").write_text("synthetic rt case")

    case = Case.open(tmp_path)
    kinds = {ref.meta.get("series_kind") for ref in case.report.dicom}
    assert "rtstruct" in kinds
    rt_ref = next(ref for ref in case.report.dicom if ref.meta.get("series_kind") == "rtstruct")
    assert "GTVp" in (rt_ref.meta.get("roi_names") or [])
    assert rt_ref.label.startswith("RTSTRUCT")
    inventory = case.report.to_inventory()
    rt_row = next(row for row in inventory["dicom"] if row.get("series_kind") == "rtstruct")
    assert "GTVp" in (rt_row.get("roi_names") or [])

    selected = case.dicom_select("which structures are contoured")
    assert selected["confident"]
    assert selected["selected"]["series_kind"] == "rtstruct"

    preview = case.fetch(rt_ref.id, view="preview")
    assert preview["view"] == "preview"
    assert "image_base64" in preview["data"]
    assert preview["data"]["meta"].get("overlay") is True
    assert preview["data"]["meta"].get("roi_names") == ["GTVp"]

