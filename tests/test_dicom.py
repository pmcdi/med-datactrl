"""DICOM adapter smoke test using a tiny synthetic series."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from datactrl import Case

pydicom = pytest.importorskip("pydicom")
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid


def _write_tiny_ct(path: Path, *, series_uid: str, instance_number: int) -> None:
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    ds = FileDataset(str(path), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.SOPClassUID = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.PatientID = "TEST"
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = series_uid
    ds.Modality = "CT"
    ds.SeriesDescription = "TinyTest"
    ds.InstanceNumber = instance_number
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
    described = case.describe(series.id)
    assert described["summary"]["Modality"] == "CT"
    assert "metadata" in described["views"]
    assert "preview" in described["views"]

    meta = case.fetch(series.id, view="metadata")
    assert meta["view"] == "metadata"
    assert "metadata" in meta["data"]
    assert "links" in meta["data"]

    tree = case.dicom_tree()
    assert tree["series_count"] >= 1
    assert "patients" in tree

    related = case.dicom_related(series.id)
    assert related["id"] == series.id
    assert related["Modality"] == "CT"
