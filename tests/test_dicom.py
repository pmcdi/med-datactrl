"""DICOM adapter smoke test using a tiny synthetic series."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from datactrl import Case
from datactrl.utils.dicom_meta import (
    ABDOMEN_FROM_SUPERIOR_MM,
    classify_series_kind,
    infer_regions,
    suggested_slice_index,
)
from datactrl.utils.dicom_views import preview_supported, views_for_series

pydicom = pytest.importorskip("pydicom")
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid


# ---------------------------------------------------------------------------
# Classification and slice heuristics
# ---------------------------------------------------------------------------


def test_classify_kind_from_image_type_and_tokens():
    assert (
        classify_series_kind(["ORIGINAL", "PRIMARY", "LOCALIZER"], "CT") == "localizer"
    )
    assert (
        classify_series_kind(["ORIGINAL", "PRIMARY", "AXIAL"], "Topogram 0.6")
        == "localizer"
    )
    assert classify_series_kind(["DERIVED", "SECONDARY"], "screen save") == "derived"
    assert (
        classify_series_kind(["ORIGINAL", "PRIMARY", "AXIAL"], "AX LAVA")
        == "acquisition"
    )
    assert classify_series_kind(None, "", modality="RTSTRUCT") == "rtstruct"
    assert classify_series_kind(None, "", modality="RTPLAN") == "rtplan"


def test_preview_views_exclude_rt_objects():
    assert not preview_supported({"Modality": "RTPLAN", "series_kind": "rtplan"})
    assert not preview_supported({"Modality": "RTDOSE", "series_kind": "rtdose"})
    rtplan_views = [
        s.name
        for s in views_for_series({"Modality": "RTPLAN", "series_kind": "rtplan"})
    ]
    assert rtplan_views == ["metadata"]
    ct_views = [
        s.name
        for s in views_for_series({"Modality": "CT", "series_kind": "acquisition"})
    ]
    assert "preview" in ct_views
    rt_views = [
        s.name
        for s in views_for_series({"Modality": "RTSTRUCT", "series_kind": "rtstruct"})
    ]
    assert "preview" in rt_views
    pt_views = views_for_series({"Modality": "PT", "series_kind": "acquisition"})
    pt_spec = next(spec for spec in pt_views if spec.name == "preview")
    assert pt_spec.defaults["window"] == "pet"
    assert "overlay=true" in pt_spec.description


def test_infer_regions_from_text():
    assert "abdomen" in infer_regions("representative CT slice of the liver")
    assert "neck" in infer_regions("representative CT slice of the neck")


def test_preferred_roi_name_priority():
    from datactrl.utils.rtstruct import preferred_roi_name

    assert preferred_roi_name(["SpinalCord", "Larynx", "PTV70", "GTVp"]) == "GTVp"
    assert preferred_roi_name(["Body", "Larynx"]) == "Larynx"
    assert preferred_roi_name(["Kidney", "Mass"]) == "Mass"
    assert preferred_roi_name(["Brainstem"]) == "Brainstem"
    assert preferred_roi_name([]) is None


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
    idx = suggested_slice_index(cranio_caudal, regions=frozenset({"abdomen"}))
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
    idx = suggested_slice_index(caudo_cranial, regions=frozenset({"abdomen"}))
    assert idx == _abdomen_index(-1510.0, -174.0, 268)
    assert idx > 240


def _abdomen_index(z_first: float, z_last: float, n: int) -> int:
    z_sup = max(z_first, z_last)
    target = z_sup - ABDOMEN_FROM_SUPERIOR_MM
    last_i = n - 1
    return int(round((target - z_first) / (z_last - z_first) * last_i))


# ---------------------------------------------------------------------------
# Synthetic DICOM fixtures
# ---------------------------------------------------------------------------


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
    modality: str = "CT",
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
    ds.Modality = modality
    ds.SeriesDescription = series_description
    ds.InstanceNumber = instance_number
    ds.ImageType = image_type or ["ORIGINAL", "PRIMARY", "AXIAL"]
    ds.ImagePositionPatient = [
        0.0,
        0.0,
        float(z_mm if z_mm is not None else -instance_number),
    ]
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
    rois: list[tuple[str, float]] | None = None,
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

    entries = list(rois) if rois else [(roi_name, z_mm)]
    structures: list[Dataset] = []
    contours: list[Dataset] = []
    contour_image = Dataset()
    contour_image.ReferencedSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    contour_image.ReferencedSOPInstanceUID = referenced_sop_uid
    for number, (name, z) in enumerate(entries, start=1):
        roi = Dataset()
        roi.ROINumber = number
        roi.ROIName = name
        roi.ReferencedFrameOfReferenceUID = frame_of_reference_uid
        structures.append(roi)

        contour = Dataset()
        contour.ContourGeometricType = "CLOSED_PLANAR"
        contour.NumberOfContourPoints = 4
        contour.ContourImageSequence = [contour_image]
        contour.ContourData = [
            1.0,
            1.0,
            z,
            6.0,
            1.0,
            z,
            6.0,
            6.0,
            z,
            1.0,
            6.0,
            z,
        ]
        roi_contour = Dataset()
        roi_contour.ReferencedROINumber = number
        roi_contour.ROIDisplayColor = [255, 0, 0]
        roi_contour.ContourSequence = [contour]
        contours.append(roi_contour)
    ds.StructureSetROISequence = structures
    ds.ROIContourSequence = contours

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


# ---------------------------------------------------------------------------
# Case previews
# ---------------------------------------------------------------------------


def test_dicom_discover_and_metadata(tmp_path: Path):
    series_uid = generate_uid()
    series_dir = tmp_path / "ct"
    series_dir.mkdir()
    for i in range(2):
        _write_tiny_ct(
            series_dir / f"img{i}.dcm", series_uid=series_uid, instance_number=i + 1
        )

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


def test_series_kind_uses_header_image_type(tmp_path: Path):
    series_uid = generate_uid()
    for i in range(2):
        _write_tiny_ct(
            tmp_path / f"loc{i}.dcm",
            series_uid=series_uid,
            instance_number=i + 1,
            series_description="Scout",
            image_type=["ORIGINAL", "PRIMARY", "LOCALIZER"],
            study="CT ABDOMEN",
        )

    case = Case.open(tmp_path)
    series = case.report.dicom[0]
    assert series.label == "Scout"
    assert series.meta["ImageType"] == ["ORIGINAL", "PRIMARY", "LOCALIZER"]
    assert series.meta["StudyDescription"] == "CT ABDOMEN"
    assert series.meta["series_kind"] == "localizer"


def test_summarize_rtstruct_lists_rois_and_prefers_gtv(tmp_path: Path):
    from imgtools.coretypes import Scan

    from datactrl.utils.rtstruct import (
        load_structure_set,
        overlay_rtstruct,
        rtstruct_slice_index,
        summarize_rt_file,
    )

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
    assert summary["preferred_roi_z_mm"] == -2.0

    from PIL import Image

    scan = Scan.from_dicom(str(tmp_path), series_id=ct_uid)
    rt = load_structure_set(tmp_path / "rt.dcm")
    assert rtstruct_slice_index(rt, scan, "GTVp") == 0

    blank = Image.new("RGB", (8, 8), (0, 0, 0))
    drawn, meta = overlay_rtstruct(blank, scan, 0, rt)
    assert meta["overlay"] is True
    assert meta["overlay_rois"][0]["name"] == "GTVp"
    arr = np.asarray(drawn)
    assert arr.any()
    # Contour spans pixels 1..6; the outline must not touch the far corner.
    assert not arr[7, 7].any()


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
    rt_ref = next(
        ref for ref in case.report.dicom if ref.meta.get("series_kind") == "rtstruct"
    )
    assert "GTVp" in (rt_ref.meta.get("roi_names") or [])
    assert rt_ref.label.startswith("RTSTRUCT")
    inventory = case.report.to_inventory()
    rt_row = next(
        row for row in inventory["dicom"] if row.get("series_kind") == "rtstruct"
    )
    assert "GTVp" in (rt_row.get("roi_names") or [])

    preview = case.fetch(rt_ref.id, view="preview")
    assert preview["view"] == "preview"
    assert "image_base64" in preview["data"]
    assert preview["data"]["meta"].get("overlay") is True
    assert preview["data"]["meta"].get("roi_names") == ["GTVp"]
    assert preview["data"]["meta"]["slice_z_mm"] == -2.0
    assert preview["data"]["meta"]["window"] == "abdomen"

    ct_ref = next(ref for ref in case.report.dicom if ref.meta.get("Modality") == "CT")
    assert ct_ref.label == "Neck 2.0 CE"
    overlaid = case.fetch(ct_ref.id, view="preview", overlay=True, slice_index=0)
    assert overlaid["data"]["meta"]["overlay_source"] == "rtstruct"
    assert overlaid["data"]["meta"]["overlay"] is False
    assert overlaid["data"]["meta"]["overlay_resource_id"] == rt_ref.id


def test_pet_window_keeps_uptake_visible_beside_a_hot_voxel():
    from datactrl.utils.image import to_uint8_rgb

    data = np.full((20, 20), 10.0)
    data[0, 0] = 10_000.0
    pet = np.asarray(to_uint8_rgb(data, window="pet"))
    stretched = np.asarray(to_uint8_rgb(data, window="minmax"))
    assert int(pet[1, 1, 0]) > 200
    assert int(stretched[1, 1, 0]) < 5


def test_pet_preview_defaults_to_roi_slice_and_overlays(tmp_path: Path):
    study_uid = generate_uid()
    pt_uid = generate_uid()
    rt_uid = generate_uid()
    frame_uid = generate_uid()
    pt_dir = tmp_path / "pt"
    rt_dir = tmp_path / "rt"
    pt_dir.mkdir()
    rt_dir.mkdir()
    sop_at_roi = None
    for i in range(8):
        written = _write_tiny_ct(
            pt_dir / f"img{i}.dcm",
            series_uid=pt_uid,
            instance_number=i + 1,
            series_description="WB_PET",
            modality="PT",
            study_uid=study_uid,
            frame_of_reference_uid=frame_uid,
            z_mm=-100.0 * i,
        )
        if i == 2:
            sop_at_roi = written
    _write_tiny_rtstruct(
        rt_dir / "struct.dcm",
        series_uid=rt_uid,
        study_uid=study_uid,
        frame_of_reference_uid=frame_uid,
        referenced_series_uid=pt_uid,
        referenced_sop_uid=sop_at_roi or written,
        roi_name="GTVp",
        z_mm=-200.0,
    )

    case = Case.open(tmp_path)
    pt_ref = next(ref for ref in case.report.dicom if ref.meta.get("Modality") == "PT")
    assert "preview" in [spec.name for spec in views_for_series(pt_ref.meta)]

    preview = case.fetch(pt_ref.id, view="preview", overlay=True)
    meta = preview["data"]["meta"]
    assert meta["window"] == "pet"
    assert meta["overlay"] is True
    assert meta["overlay_source"] == "rtstruct"
    assert meta["overlay_rois"][0]["name"] == "GTVp"
    assert meta["slice_z_mm"] == -200.0


def test_preview_roi_selects_one_structure_and_its_slice(tmp_path: Path):
    from datactrl.errors import InvalidFetchOptions

    study_uid = generate_uid()
    ct_uid = generate_uid()
    rt_uid = generate_uid()
    frame_uid = generate_uid()
    ct_dir = tmp_path / "ct"
    rt_dir = tmp_path / "rt"
    ct_dir.mkdir()
    rt_dir.mkdir()
    sop = None
    for i, z in enumerate((-1.0, -5.0)):
        written = _write_tiny_ct(
            ct_dir / f"img{i}.dcm",
            series_uid=ct_uid,
            instance_number=i + 1,
            study_uid=study_uid,
            frame_of_reference_uid=frame_uid,
            z_mm=z,
        )
        if i == 0:
            sop = written
    _write_tiny_rtstruct(
        rt_dir / "struct.dcm",
        series_uid=rt_uid,
        study_uid=study_uid,
        frame_of_reference_uid=frame_uid,
        referenced_series_uid=ct_uid,
        referenced_sop_uid=sop or written,
        rois=[("GTVp", -1.0), ("SpinalCord", -5.0)],
    )

    case = Case.open(tmp_path)
    rt_ref = next(
        ref for ref in case.report.dicom if ref.meta.get("series_kind") == "rtstruct"
    )
    ct_ref = next(ref for ref in case.report.dicom if ref.meta.get("Modality") == "CT")

    selected = case.fetch(rt_ref.id, view="preview", roi="spinalcord")
    meta = selected["data"]["meta"]
    assert meta["roi"] == "SpinalCord"
    assert meta["slice_z_mm"] == -5.0
    assert [row["name"] for row in meta["overlay_rois"]] == ["SpinalCord"]

    default = case.fetch(rt_ref.id, view="preview")
    assert default["data"]["meta"].get("roi") is None
    assert default["data"]["meta"]["slice_z_mm"] == -1.0
    assert [row["name"] for row in default["data"]["meta"]["overlay_rois"]] == ["GTVp"]

    on_ct = case.fetch(
        ct_ref.id,
        view="preview",
        overlay=True,
        slice_index=selected["bounds"]["slice_index"],
        roi="SpinalCord",
    )
    assert [row["name"] for row in on_ct["data"]["meta"]["overlay_rois"]] == [
        "SpinalCord"
    ]

    with pytest.raises(InvalidFetchOptions, match="Unknown ROI"):
        case.fetch(rt_ref.id, view="preview", roi="Brainstem")
    with pytest.raises(InvalidFetchOptions, match="overlay=true"):
        case.fetch(ct_ref.id, view="preview", roi="GTVp")
