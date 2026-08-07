"""NIfTI adapter tests using tiny synthetic volumes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from datactrl import Case, NiftiCrawlOptions
from datactrl.errors import InvalidFetchOptions
from datactrl.tools import dispatch, reset_sessions

sitk = pytest.importorskip("SimpleITK")


def _write_volume(path: Path, *, shape: tuple[int, int, int] = (4, 6, 8)) -> None:
    """Write a tiny NIfTI volume (array shape is z, y, x)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape)
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((1.0, 1.5, 2.0))
    image.SetOrigin((0.0, 0.0, 0.0))
    sitk.WriteImage(image, str(path))


def test_nifti_discover_metadata_and_preview(tmp_path: Path):
    _write_volume(tmp_path / "scan.nii.gz")
    (tmp_path / "note.txt").write_text("synthetic nifti case")

    case = Case.open(tmp_path)
    assert len(case.report.nifti) == 1
    assert len(case.report.notes) == 1
    resource = case.report.nifti[0]
    assert resource.id.startswith("nifti:")
    assert resource.path == "scan.nii.gz"

    described = case.describe(resource.id)
    assert described["views"] == ["metadata", "preview"]
    assert described["summary"]["shape"] == [4, 6, 8]
    assert described["summary"]["spacing"] == [1.0, 1.5, 2.0]
    assert described["summary"]["compressed"] is True

    meta = case.fetch_nifti_metadata(resource.id)
    assert meta["view"] == "metadata"
    assert meta["data"]["metadata"]["shape"] == [4, 6, 8]
    assert meta["bounds"]["truncated"] is False

    preview = case.fetch_nifti_preview(resource.id)
    assert preview["view"] == "preview"
    assert preview["data"]["image_base64"]
    assert preview["bounds"]["slice_index"] == 2
    assert preview["bounds"]["n_slices"] == 4

    edge = case.fetch(resource.id, view="preview", slice_index=0)
    assert edge["bounds"]["slice_index"] == 0

    with pytest.raises(InvalidFetchOptions):
        case.fetch(resource.id, view="preview", slice_index=99)


def test_nifti_classify_compound_extension(tmp_path: Path):
    from datactrl.discovery.classify import classify_file
    from datactrl.types import Modality

    assert classify_file(tmp_path / "a.nii") == Modality.NIFTI
    assert classify_file(tmp_path / "a.NII.GZ") == Modality.NIFTI
    assert classify_file(tmp_path / "archive.tar.gz") is None


def test_nifti_crawl_patterns_and_links(tmp_path: Path):
    pytest.importorskip("imgtools.nifti.crawl")

    _write_volume(tmp_path / "images" / "P001_CT.nii.gz")
    _write_volume(tmp_path / "masks" / "P001_CT_tumor.nii.gz")
    _write_volume(tmp_path / "orphan.nii.gz")

    cache_dir = tmp_path / "crawl_out"
    case = Case.open(
        tmp_path,
        nifti=NiftiCrawlOptions(
            scan_name_pattern="images/{PatientID}_{Modality}.nii.gz",
            mask_name_pattern="masks/{PatientID}_{Modality}_{ROI}.nii.gz",
            output_dir=cache_dir,
            dataset_name="demo",
            force=True,
            n_jobs=1,
        ),
    )

    assert len(case.report.nifti) == 3
    by_path = {ref.path: ref for ref in case.report.nifti}
    assert set(by_path) == {
        "images/P001_CT.nii.gz",
        "masks/P001_CT_tumor.nii.gz",
        "orphan.nii.gz",
    }

    scan = by_path["images/P001_CT.nii.gz"]
    mask = by_path["masks/P001_CT_tumor.nii.gz"]
    orphan = by_path["orphan.nii.gz"]

    assert scan.meta["file_type"] == "scan"
    assert scan.meta["placeholders"]["PatientID"] == "P001"
    assert mask.meta["file_type"] == "mask"
    assert "file_type" not in orphan.meta

    mask_desc = case.describe(mask.id)
    assert any(
        link["to_id"] == scan.id
        and link["relation"] == "references"
        and link["source"] == "structural"
        for link in mask_desc["links"]
    )
    scan_links = case.links(scan.id)
    assert any(link["to_id"] == mask.id for link in scan_links)

    assert (cache_dir / "demo" / "index.csv").is_file()
    assert (cache_dir / "demo" / "crawl_cache.json").is_file()


def test_nifti_tool_dispatch(tmp_path: Path):
    _write_volume(tmp_path / "vol.nii")
    reset_sessions()

    import json

    report = json.loads(dispatch("discover", {"root": str(tmp_path)}))
    assert report["counts"]["nifti"] == 1
    resource_id = report["nifti"][0]["id"]
    fetched = json.loads(
        dispatch(
            "fetch",
            {
                "root": str(tmp_path),
                "resource_id": resource_id,
                "view": "metadata",
            },
        )
    )
    assert fetched["view"] == "metadata"
    assert fetched["data"]["metadata"]["extension"] == ".nii"
