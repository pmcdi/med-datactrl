"""Load an imagetools 3D volume and encode it as gzip NIfTI."""

from __future__ import annotations

import base64
import os
import re
import tempfile
from pathlib import Path
from typing import Any, cast

import SimpleITK as sitk
from imgtools.coretypes import (
    PET,
    SEG,
    Dose,
    MedImage,
    ROIMatcher,
    ROIMatchFailurePolicy,
    ROIMatchStrategy,
    RTStructureSet,
    Scan,
)
from imgtools.transforms import Resample

from datactrl.errors import InvalidFetchOptions, LimitExceeded
from datactrl.limits import MAX_VOLUME_BYTES, MAX_VOLUME_VOXELS

MaskSource = RTStructureSet | SEG


# ---------------------------------------------------------------------------
# Spacing
# ---------------------------------------------------------------------------


def volume_spacing(raw: Any) -> tuple[float, float, float] | None:
    """Parse ``spacing`` in mm. All zeros means keep the native grid."""
    parsed = _parse_spacing(raw)
    if parsed is None or all(value == 0 for value in parsed):
        return None
    return parsed


def _parse_spacing(raw: Any) -> tuple[float, float, float] | None:
    """Coerce one number or three numbers, in mm."""
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        parsed = (float(raw), float(raw), float(raw))
    else:
        try:
            values = tuple(float(part) for part in raw)
        except (TypeError, ValueError) as exc:
            raise InvalidFetchOptions(
                "spacing must be three non-negative numbers (mm)"
            ) from exc
        if len(values) != 3:
            raise InvalidFetchOptions("spacing must be three non-negative numbers (mm)")
        parsed = (values[0], values[1], values[2])
    if any(value < 0 for value in parsed):
        raise InvalidFetchOptions("spacing must be three non-negative numbers (mm)")
    return parsed


def resample_labels(
    image: sitk.Image, spacing: tuple[float, float, float]
) -> sitk.Image:
    """Nearest-neighbor resample so label ids stay whole numbers."""
    return Resample(spacing, interpolation="nearest", anti_alias=False)(image)


def resample_volume(image: MedImage, spacing: tuple[float, float, float]) -> MedImage:
    """Resample with imagetools. A 0 component keeps that axis.

    Anti-alias is off: the Gaussian needs four voxels on every axis, and short
    series fail that check. The result stays the same imagetools class so mask
    voxelization can read its geometry.
    """
    resampled = Resample(spacing, interpolation="linear", anti_alias=False)(image)
    metadata = dict(getattr(image, "metadata", {}) or {})
    return cast(MedImage, type(image)(resampled, metadata))


def load_scalar_volume(
    modality: str,
    directory: Path,
    series_uid: str,
    files: list[str],
) -> MedImage:
    """Load CT, MR, PT, or RTDOSE through that class's ``from_dicom``."""
    folder = str(directory)
    if modality in {"CT", "MR"}:
        return Scan.from_dicom(folder, series_id=series_uid, file_names=files)
    if modality == "PT":
        return PET.from_dicom(folder, series_id=series_uid, file_names=files)
    if modality == "RTDOSE":
        return _dose_from_dicom(folder, series_uid, files)
    raise InvalidFetchOptions(f"{modality or 'series'} has no imagetools volume")


def _dose_from_dicom(directory: str, series_uid: str, files: list[str]) -> Dose:
    """Call ``Dose.from_dicom``.

    The series is read from a directory, then ``dcmread`` is called on that same
    path for ``DoseGridScaling``. A directory is not a DICOM file, so that header
    read is pointed at the first instance for this call.
    """
    import imgtools.coretypes.imagetypes.dose as dose_mod

    real_dcmread = dose_mod.dcmread
    header = files[0]

    def dcmread(path: str, *args: Any, **kwargs: Any) -> Any:
        if Path(path).is_dir():
            return real_dcmread(header, *args, **kwargs)
        return real_dcmread(path, *args, **kwargs)

    dose_mod.dcmread = dcmread  # type: ignore[assignment]
    try:
        return Dose.from_dicom(directory, series_id=series_uid, file_names=files)
    finally:
        dose_mod.dcmread = real_dcmread


def mask_to_label(
    mask: MaskSource,
    reference: MedImage,
    roi_names: list[str],
) -> tuple[sitk.Image, dict[str, str]]:
    """Voxelize every ROI onto ``reference`` and collapse to one label image."""
    match_map = {name: [f"^{re.escape(name)}$"] for name in roi_names} or {
        "ROI": [".*"]
    }
    matcher = ROIMatcher(
        match_map=match_map,
        handling_strategy=ROIMatchStrategy.SEPARATE,
        ignore_case=True,
        on_missing_regex=ROIMatchFailurePolicy.ERROR,
    )
    vector = mask.get_vector_mask(reference, matcher)
    if vector is None:
        raise InvalidFetchOptions(
            "No ROIs could be voxelized onto the referenced series"
        )
    try:
        label = vector.to_label_image()
    except ValueError:
        label = vector.to_sparse_mask()
    labels = {
        str(index): (mapping.roi_names[0] if mapping.roi_names else mapping.roi_key)
        for index, mapping in vector.roi_mapping.items()
        if index != 0 and mapping.roi_key != "Background"
    }
    return label, labels


# ---------------------------------------------------------------------------
# Encode
# ---------------------------------------------------------------------------


def volume_payload(image: sitk.Image) -> dict[str, Any]:
    """Encode ``image`` as base64 ``.nii.gz`` or reject when it is too large."""
    n_voxels = 1
    for dim in image.GetSize():
        n_voxels *= int(dim)
    if n_voxels > MAX_VOLUME_VOXELS:
        raise LimitExceeded(
            f"volume has {n_voxels} voxels (max {MAX_VOLUME_VOXELS}); "
            "retry with coarser spacing"
        )
    blob = _nifti_gz(image)
    if len(blob) > MAX_VOLUME_BYTES:
        raise LimitExceeded(
            f"volume is {len(blob)} bytes (max {MAX_VOLUME_BYTES}); "
            "retry with coarser spacing"
        )
    return {
        "volume_base64": base64.b64encode(blob).decode("ascii"),
        "encoding": "nii.gz",
        "meta": {
            "size": [int(value) for value in image.GetSize()],
            "spacing": [round(float(value), 4) for value in image.GetSpacing()],
            "origin": [round(float(value), 4) for value in image.GetOrigin()],
            "n_voxels": n_voxels,
            "nbytes": len(blob),
        },
    }


def _nifti_gz(image: sitk.Image) -> bytes:
    fd, name = tempfile.mkstemp(suffix=".nii.gz")
    os.close(fd)
    path = Path(name)
    try:
        sitk.WriteImage(image, str(path), useCompression=True, compressionLevel=9)
        return path.read_bytes()
    finally:
        path.unlink(missing_ok=True)
