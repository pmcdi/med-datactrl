"""RTSTRUCT / RT object retrieve: ROI inventory and CT overlay.

RTSTRUCT files are not images. SimpleITK cannot preview them. ROI names and
contour geometry are header facts — list them and draw them on the referenced
CT/MR slice instead of asking a vision model to invent GTVs.
"""

from __future__ import annotations

import math
import re
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw

_RT_KINDS = {
    "RTSTRUCT": "rtstruct",
    "RTPLAN": "rtplan",
    "RTDOSE": "rtdose",
}

_MAX_ROI_NAMES = 40
_MAX_DRAW_ROIS = 16
_SKIP_WHEN_CROWDED = re.compile(r"external|body|skin|outline|patient", re.I)
_PREFERRED_ROI = (
    re.compile(r"gtv", re.I),
    re.compile(r"ptv", re.I),
    re.compile(r"ctv", re.I),
    re.compile(r"tumor|gross", re.I),
    re.compile(r"larynx|glottic|primary", re.I),
)
_PALETTE = (
    (255, 72, 72),
    (64, 196, 255),
    (255, 200, 48),
    (80, 220, 120),
    (200, 120, 255),
    (255, 140, 64),
    (64, 255, 220),
    (255, 96, 180),
)


def rt_series_kind(modality: str | None) -> str | None:
    """Map a DICOM Modality tag to rtstruct/rtplan/rtdose, else None."""
    if not modality:
        return None
    return _RT_KINDS.get(str(modality).strip().upper())


def summarize_rt_file(path: str | Path) -> dict[str, Any]:
    """Header-only RT summary: ROI names and a preferred contour z, no point lists."""
    ds = _read_dataset(path)
    if ds is None:
        return {}

    modality = _scalar(getattr(ds, "Modality", None)) or ""
    kind = rt_series_kind(modality) or "acquisition"
    meta: dict[str, Any] = {
        "Modality": modality or None,
        "series_kind": kind,
        "SeriesDescription": _scalar(getattr(ds, "SeriesDescription", None)),
        "ProtocolName": _scalar(getattr(ds, "ProtocolName", None)),
        "StudyDescription": _scalar(getattr(ds, "StudyDescription", None)),
        "FrameOfReferenceUID": _scalar(getattr(ds, "FrameOfReferenceUID", None)),
        "structure_set_label": _scalar(getattr(ds, "StructureSetLabel", None)),
        "referenced_series_uid": _referenced_series_uid(ds),
    }

    if kind == "rtplan":
        meta["structure_set_label"] = _scalar(getattr(ds, "RTPlanLabel", None))
        return _drop_empty(meta)
    if kind == "rtdose":
        meta["structure_set_label"] = _scalar(getattr(ds, "DoseSummationType", None))
        return _drop_empty(meta)
    if kind != "rtstruct":
        return _drop_empty(meta)

    rois: list[dict[str, Any]] = []
    for item in getattr(ds, "StructureSetROISequence", None) or []:
        number = _as_int(getattr(item, "ROINumber", None))
        name = _scalar(getattr(item, "ROIName", None))
        if number is None or not name:
            continue
        rois.append({"number": number, "name": name})

    z_by_number: dict[int, list[float]] = {}
    for item in getattr(ds, "ROIContourSequence", None) or []:
        number = _as_int(getattr(item, "ReferencedROINumber", None))
        if number is None:
            continue
        zs: list[float] = []
        for contour in getattr(item, "ContourSequence", None) or []:
            zs.extend(_contour_z_values(contour))
        if zs:
            z_by_number[number] = zs

    names = [roi["name"] for roi in rois][:_MAX_ROI_NAMES]
    preferred = _pick_preferred_roi(rois, z_by_number)
    preferred_z = None
    if preferred is not None:
        zs = z_by_number.get(int(preferred["number"])) or []
        if zs:
            preferred_z = round(float(statistics.median(zs)), 3)

    meta["roi_names"] = names
    meta["preferred_roi"] = None if preferred is None else preferred["name"]
    meta["preferred_roi_z_mm"] = preferred_z
    if rois:
        z_lo = [min(z_by_number[roi["number"]]) for roi in rois if roi["number"] in z_by_number]
        z_hi = [max(z_by_number[roi["number"]]) for roi in rois if roi["number"] in z_by_number]
        if z_lo and z_hi:
            meta["roi_z_min_mm"] = round(min(z_lo), 3)
            meta["roi_z_max_mm"] = round(max(z_hi), 3)
    return _drop_empty(meta)


def overlay_contours(
    image: Image.Image,
    ct_path: str | Path,
    rt_path: str | Path,
    *,
    roi_filter: Sequence[str] | None = None,
) -> tuple[Image.Image, dict[str, Any]]:
    """Draw RTSTRUCT contours that fall on this CT slice. Outline only."""
    ct = _read_dataset(ct_path)
    rt = _read_dataset(rt_path)
    if ct is None or rt is None:
        return image, {"overlay": False, "overlay_error": "could not read CT or RTSTRUCT header"}

    geometry = _slice_geometry(ct)
    if geometry is None:
        return image, {"overlay": False, "overlay_error": "CT slice missing IPP/IOP/PixelSpacing"}

    wanted = None
    if roi_filter:
        wanted = {name.lower() for name in roi_filter if name}

    name_by_number: dict[int, str] = {}
    for item in getattr(rt, "StructureSetROISequence", None) or []:
        number = _as_int(getattr(item, "ROINumber", None))
        name = _scalar(getattr(item, "ROIName", None))
        if number is not None and name:
            name_by_number[number] = name

    candidates: list[tuple[int, str, Any]] = []
    for item in getattr(rt, "ROIContourSequence", None) or []:
        number = _as_int(getattr(item, "ReferencedROINumber", None))
        if number is None:
            continue
        name = name_by_number.get(number) or f"ROI {number}"
        if wanted is not None and name.lower() not in wanted:
            continue
        candidates.append((number, name, item))

    if len(candidates) > _MAX_DRAW_ROIS:
        crowded = [
            row
            for row in candidates
            if not _SKIP_WHEN_CROWDED.search(row[1])
        ]
        candidates = (crowded or candidates)[:_MAX_DRAW_ROIS]

    drawn: list[dict[str, Any]] = []
    canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size
    half_thick = _match_tolerance_mm(ct)

    for index, (_number, name, item) in enumerate(candidates):
        color = _PALETTE[index % len(_PALETTE)]
        n_on_slice = 0
        for contour in getattr(item, "ContourSequence", None) or []:
            points = _contour_patient_points(contour)
            if len(points) < 2:
                continue
            z_values = [pt[2] for pt in points]
            if min(abs(z - geometry["z_mm"]) for z in z_values) > half_thick:
                continue
            pixels = [
                patient_xyz_to_pixel(pt, geometry["ipp"], geometry["iop"], geometry["spacing"])
                for pt in points
            ]
            xy = [
                (max(-1.0, min(width + 1.0, col)), max(-1.0, min(height + 1.0, row)))
                for col, row in pixels
            ]
            if len(xy) < 2:
                continue
            geom = str(getattr(contour, "ContourGeometricType", "CLOSED_PLANAR") or "")
            if geom.upper() == "POINT":
                for col, row in xy:
                    draw.ellipse((col - 2, row - 2, col + 2, row + 2), fill=color)
            else:
                if xy[0] != xy[-1] and geom.upper() != "OPEN_PLANAR":
                    xy = [*xy, xy[0]]
                draw.line(xy, fill=color, width=2)
            n_on_slice += 1
        if n_on_slice:
            drawn.append(
                {
                    "name": name,
                    "color": "#{:02x}{:02x}{:02x}".format(*color),
                    "n_contours": n_on_slice,
                }
            )

    return canvas, {
        "overlay": bool(drawn),
        "overlay_rois": drawn,
        "overlay_source": "rtstruct",
    }


def patient_xyz_to_pixel(
    xyz: Sequence[float],
    ipp: Sequence[float],
    iop: Sequence[float],
    spacing: Sequence[float],
) -> tuple[float, float]:
    """Project a patient-space point onto the slice (column, row)."""
    origin = (float(ipp[0]), float(ipp[1]), float(ipp[2]))
    # IOP: direction of the first row (increasing column), then first column (increasing row).
    row_dir = (float(iop[0]), float(iop[1]), float(iop[2]))
    col_dir = (float(iop[3]), float(iop[4]), float(iop[5]))
    row_sp = float(spacing[0])
    col_sp = float(spacing[1] if len(spacing) > 1 else spacing[0])
    delta = (
        float(xyz[0]) - origin[0],
        float(xyz[1]) - origin[1],
        float(xyz[2]) - origin[2],
    )
    col = _dot(delta, row_dir) / col_sp if col_sp else 0.0
    row = _dot(delta, col_dir) / row_sp if row_sp else 0.0
    return col, row


def _pick_preferred_roi(
    rois: Sequence[Mapping[str, Any]],
    z_by_number: Mapping[int, Sequence[float]],
) -> dict[str, Any] | None:
    with_z = [roi for roi in rois if roi["number"] in z_by_number]
    if not with_z:
        return None
    for pattern in _PREFERRED_ROI:
        for roi in with_z:
            if pattern.search(str(roi["name"])):
                return dict(roi)
    return dict(with_z[0])


def _referenced_series_uid(dataset: Any) -> str | None:
    for frame in getattr(dataset, "ReferencedFrameOfReferenceSequence", None) or []:
        for study in getattr(frame, "RTReferencedStudySequence", None) or []:
            for series in getattr(study, "RTReferencedSeriesSequence", None) or []:
                uid = _scalar(getattr(series, "SeriesInstanceUID", None))
                if uid:
                    return uid
    return None


def _slice_geometry(dataset: Any) -> dict[str, Any] | None:
    ipp = getattr(dataset, "ImagePositionPatient", None)
    iop = getattr(dataset, "ImageOrientationPatient", None)
    spacing = getattr(dataset, "PixelSpacing", None)
    try:
        ipp_t = (float(ipp[0]), float(ipp[1]), float(ipp[2]))
        iop_t = tuple(float(v) for v in iop)
        spacing_t = (float(spacing[0]), float(spacing[1] if len(spacing) > 1 else spacing[0]))
    except (TypeError, ValueError, IndexError):
        return None
    if len(iop_t) < 6:
        return None
    return {"ipp": ipp_t, "iop": iop_t, "spacing": spacing_t, "z_mm": ipp_t[2]}


def _match_tolerance_mm(dataset: Any) -> float:
    thickness = getattr(dataset, "SliceThickness", None)
    try:
        half = abs(float(thickness)) * 0.51
    except (TypeError, ValueError):
        half = 1.0
    return max(half, 1.0)


def _contour_patient_points(contour: Any) -> list[tuple[float, float, float]]:
    data = getattr(contour, "ContourData", None)
    if data is None:
        return []
    try:
        values = [float(v) for v in data]
    except (TypeError, ValueError):
        return []
    points: list[tuple[float, float, float]] = []
    for i in range(0, len(values) - 2, 3):
        points.append((values[i], values[i + 1], values[i + 2]))
    return points


def _contour_z_values(contour: Any) -> list[float]:
    return [pt[2] for pt in _contour_patient_points(contour)]


def _read_dataset(path: str | Path) -> Any | None:
    try:
        import pydicom

        return pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
    except Exception:
        return None


def _scalar(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None
    return text


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return float(a[0]) * float(b[0]) + float(a[1]) * float(b[1]) + float(a[2]) * float(b[2])


def _drop_empty(meta: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in meta.items():
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        if isinstance(value, list) and not value:
            continue
        out[key] = value
    return out
