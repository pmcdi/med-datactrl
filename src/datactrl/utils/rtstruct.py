"""RTSTRUCT / RT object retrieve: ROI inventory and CT overlay.

RTSTRUCT files are not images. SimpleITK cannot preview them. ROI names and
contour geometry are header facts — list them and draw them on the referenced
CT/MR slice instead of asking a vision model to invent GTVs.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from imgtools.coretypes import MedImage
from imgtools.coretypes.masktypes.roi_matching import (
    ROIMatchFailurePolicy,
    ROIMatchStrategy,
    create_roi_matcher,
)
from imgtools.coretypes.masktypes.structureset import RTStructureSet
from imgtools.dicom.dicom_metadata import extract_metadata
from imgtools.utils.imageutils import physical_points_to_idxs
from PIL import Image, ImageDraw

_RT_KINDS = {
    "RTSTRUCT": "rtstruct",
    "RTPLAN": "rtplan",
    "RTDOSE": "rtdose",
}

_MAX_ROI_NAMES = 40
_MAX_DRAW_ROIS = 16
_PREFERRED_ROI_MATCHER = create_roi_matcher(
    {
        "gtv": [".*gtv.*"],
        "ptv": [".*ptv.*"],
        "ctv": [".*ctv.*"],
        "tumor": [".*(tumor|tumour|gross|mass|lesion).*"],
        "primary": [".*(larynx|glottic|primary).*"],
    },
    handling_strategy=ROIMatchStrategy.KEEP_FIRST,
    on_missing_regex=ROIMatchFailurePolicy.IGNORE,
)
_OUTLINE_ROI_MATCHER = create_roi_matcher(
    {"outline": [".*(external|body|skin|outline|patient).*"]},
    handling_strategy=ROIMatchStrategy.SEPARATE,
    on_missing_regex=ROIMatchFailurePolicy.IGNORE,
)
OVERLAY_PALETTE = (
    (255, 72, 72),
    (64, 196, 255),
    (255, 200, 48),
    (80, 220, 120),
    (200, 120, 255),
    (255, 140, 64),
    (64, 255, 220),
    (255, 96, 180),
)


# ---------------------------------------------------------------------------
# Kinds and ROI names
# ---------------------------------------------------------------------------


def rt_series_kind(modality: str | None) -> str | None:
    """Map a DICOM Modality tag to rtstruct/rtplan/rtdose, else None."""
    if not modality:
        return None
    return _RT_KINDS.get(str(modality).strip().upper())


def preferred_roi_name(names: Sequence[str]) -> str | None:
    """First GTV/PTV/CTV/tumor/primary ROI by priority, else the first name."""
    names = [str(name) for name in names if name]
    if not names:
        return None
    for _key, hits in _PREFERRED_ROI_MATCHER.match_rois(names):
        if hits:
            return hits[0]
    return names[0]


def select_overlay_rois(
    names: Sequence[str],
    roi_filter: Sequence[str] | None = None,
) -> list[str]:
    """Apply an optional name filter, then drop body outlines when crowded."""
    selected = [str(name) for name in names if name]
    if roi_filter:
        wanted = {name.lower() for name in roi_filter if name}
        selected = [name for name in selected if name.lower() in wanted]
    if len(selected) > _MAX_DRAW_ROIS:
        outlines = {
            hit
            for _key, hits in _OUTLINE_ROI_MATCHER.match_rois(selected)
            for hit in hits
        }
        crowded = [name for name in selected if name not in outlines]
        selected = (crowded or selected)[:_MAX_DRAW_ROIS]
    return selected


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------


def summarize_rt_file(
    path: str | Path,
    crawl_meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """RT summary from crawl metadata; RTSTRUCT contours give a preferred ROI z."""
    source = dict(crawl_meta) if crawl_meta else _extract_rt_metadata(path)
    if not source:
        return {}

    modality = _scalar(source.get("Modality")) or ""
    kind = rt_series_kind(modality) or "acquisition"
    meta: dict[str, Any] = {
        "Modality": modality or None,
        "series_kind": kind,
        "FrameOfReferenceUID": _scalar(source.get("FrameOfReferenceUID")),
        "referenced_series_uid": _scalar(source.get("ReferencedSeriesUID")),
    }

    if kind == "rtplan":
        meta["RTPlanLabel"] = _scalar(source.get("RTPlanLabel"))
        meta["RTPlanName"] = _scalar(source.get("RTPlanName"))
        meta["structure_set_label"] = meta["RTPlanLabel"]
        return _drop_empty(meta)
    if kind == "rtdose":
        for key in ("DoseType", "DoseUnits", "DoseSummationType"):
            meta[key] = _scalar(source.get(key))
        return _drop_empty(meta)
    if kind != "rtstruct":
        return _drop_empty(meta)

    meta["structure_set_label"] = _scalar(source.get("StructureSetLabel"))
    names = [str(name) for name in source.get("ROINames") or [] if name]
    meta["roi_names"] = names[:_MAX_ROI_NAMES]
    meta["NumROIs"] = len(names)

    try:
        rt = load_structure_set(path, source)
    except Exception:
        return _drop_empty(meta)

    z_by_name = {name: _roi_z_values(rt, name) for name in rt.roi_names}
    z_by_name = {name: zs for name, zs in z_by_name.items() if zs}
    preferred = preferred_roi_name(list(z_by_name))
    meta["preferred_roi"] = preferred
    if preferred is not None:
        meta["preferred_roi_z_mm"] = round(
            float(statistics.median(z_by_name[preferred])), 3
        )
    if z_by_name:
        meta["roi_z_min_mm"] = round(min(min(zs) for zs in z_by_name.values()), 3)
        meta["roi_z_max_mm"] = round(max(max(zs) for zs in z_by_name.values()), 3)
    if rt.roi_map_errors:
        meta["roi_extraction_errors"] = sorted(rt.roi_map_errors)
    return _drop_empty(meta)


def load_structure_set(
    path: str | Path,
    crawl_meta: Mapping[str, Any] | None = None,
) -> RTStructureSet:
    """Lazily parse an RTSTRUCT; crawl metadata skips re-extracting ROI names."""
    metadata = dict(crawl_meta) if crawl_meta and crawl_meta.get("ROINames") else None
    return RTStructureSet.from_dicom(str(path), metadata=metadata)


def rtstruct_slice_index(
    rt: RTStructureSet,
    scan: MedImage,
    roi_name: str | None,
) -> int | None:
    """Median contour slice of ``roi_name`` on the ``scan`` grid."""
    if not roi_name or roi_name not in rt.roi_map:
        return None
    first_points = [
        points[:1] for points in _roi_contour_points(rt, roi_name) if len(points)
    ]
    if not first_points:
        return None
    z_idx = [
        float(zyx[0][0])
        for zyx in physical_points_to_idxs(scan, first_points, continuous=True)
    ]
    depth = int(scan.GetSize()[2])
    return max(0, min(depth - 1, int(round(statistics.median(z_idx)))))


# ---------------------------------------------------------------------------
# Overlay
# ---------------------------------------------------------------------------


def overlay_rtstruct(
    image: Image.Image,
    scan: MedImage,
    slice_index: int,
    rt: RTStructureSet,
    *,
    roi_filter: Sequence[str] | None = None,
) -> tuple[Image.Image, dict[str, Any]]:
    """Draw RTSTRUCT contours that fall on ``slice_index`` of ``scan``. Outline only."""
    canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size

    drawn: list[dict[str, Any]] = []
    for index, name in enumerate(select_overlay_rois(rt.roi_names, roi_filter)):
        color = OVERLAY_PALETTE[index % len(OVERLAY_PALETTE)]
        on_slice = [
            points
            for points in _roi_contour_points(rt, name)
            if len(points) >= 2 and _near_slice(scan, points[0], slice_index)
        ]
        if not on_slice:
            continue
        n_on_slice = 0
        for zyx in physical_points_to_idxs(scan, on_slice, continuous=True):
            if abs(float(np.mean(zyx[:, 0])) - slice_index) > 0.5:
                continue
            xy = [
                (
                    max(-1.0, min(width + 1.0, float(x))),
                    max(-1.0, min(height + 1.0, float(y))),
                )
                for _z, y, x in zyx
            ]
            draw.line([*xy, xy[0]], fill=color, width=2)
            n_on_slice += 1
        if n_on_slice:
            drawn.append(
                {
                    "name": name,
                    "color": "#{:02x}{:02x}{:02x}".format(*color),
                    "n_contours": n_on_slice,
                }
            )

    meta: dict[str, Any] = {
        "overlay": bool(drawn),
        "overlay_rois": drawn,
        "overlay_source": "rtstruct",
    }
    if rt.roi_map_errors:
        meta["overlay_skipped_rois"] = sorted(rt.roi_map_errors)
    return canvas, meta


# ---------------------------------------------------------------------------
# Contour geometry
# ---------------------------------------------------------------------------


def _near_slice(scan: MedImage, point: np.ndarray, slice_index: int) -> bool:
    z_idx = float(
        scan.TransformPhysicalPointToContinuousIndex(tuple(float(v) for v in point))[2]
    )
    return abs(z_idx - slice_index) <= 1.0


def _roi_contour_points(rt: RTStructureSet, roi_name: str) -> list[np.ndarray]:
    points: list[np.ndarray] = []
    for contour in rt.roi_map.get(roi_name) or []:
        data = getattr(contour, "ContourData", None)
        if data is None:
            continue
        try:
            values = np.asarray([float(v) for v in data], dtype=float)
        except (TypeError, ValueError):
            continue
        usable = len(values) - len(values) % 3
        if usable:
            points.append(values[:usable].reshape(-1, 3))
    return points


def _roi_z_values(rt: RTStructureSet, roi_name: str) -> list[float]:
    return [
        float(z) for points in _roi_contour_points(rt, roi_name) for z in points[:, 2]
    ]


def _extract_rt_metadata(path: str | Path) -> dict[str, Any]:
    try:
        return dict(extract_metadata(str(path)))
    except Exception:
        return {}


def _scalar(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None
    return text


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
