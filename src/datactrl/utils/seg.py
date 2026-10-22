"""DICOM-SEG retrieve: segment outlines on the referenced CT/MR slice.

SEG frames are binary masks, not anatomy. Previewing them alone shows a
silhouette; outline them on the image series they reference instead.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from imgtools.coretypes import SEG, MedImage
from PIL import Image

from datactrl.utils.rtstruct import OVERLAY_PALETTE, select_overlay_rois


def load_seg(path: str | Path, crawl_meta: Mapping[str, Any] | None = None) -> SEG:
    """Decode a DICOM-SEG (binary or fractional) into per-segment arrays."""
    return SEG.from_dicom(str(path), metadata=dict(crawl_meta) if crawl_meta else None)


def seg_slice_index(seg: SEG, scan: MedImage, label: str | None) -> int | None:
    """Slice of ``scan`` where segment ``label`` covers the most pixels."""
    segment = _segment_by_label(seg, label)
    if segment is None or segment.data_array is None:
        return None
    best_z: int | None = None
    best_area = 0
    for z_idx, frame in zip(
        _frame_slice_indices(seg, scan), segment.data_array, strict=True
    ):
        if z_idx is None:
            continue
        area = int(np.count_nonzero(frame))
        if area > best_area:
            best_z, best_area = z_idx, area
    return best_z


def overlay_seg(
    image: Image.Image,
    scan: MedImage,
    slice_index: int,
    seg: SEG,
    *,
    roi_filter: Sequence[str] | None = None,
) -> tuple[Image.Image, dict[str, Any]]:
    """Outline SEG segments that fall on ``slice_index`` of ``scan``."""
    canvas = np.asarray(image.convert("RGB")).copy()
    frame_z = _frame_slice_indices(seg, scan)
    shape = canvas.shape[:2]

    drawn: list[dict[str, Any]] = []
    skipped: list[str] = []
    for index, label in enumerate(select_overlay_rois(seg.labels, roi_filter)):
        segment = _segment_by_label(seg, label)
        if segment is None or segment.data_array is None:
            continue
        if tuple(segment.data_array.shape[1:]) != shape:
            skipped.append(label)
            continue
        mask = np.zeros(shape, dtype=bool)
        for z_idx, frame in zip(frame_z, segment.data_array, strict=True):
            if z_idx == slice_index:
                mask |= frame > 0
        if not mask.any():
            continue
        color = OVERLAY_PALETTE[index % len(OVERLAY_PALETTE)]
        canvas[mask_outline(mask)] = color
        drawn.append(
            {
                "name": label,
                "color": "#{:02x}{:02x}{:02x}".format(*color),
                "n_pixels": int(mask.sum()),
            }
        )

    meta: dict[str, Any] = {
        "overlay": bool(drawn),
        "overlay_rois": drawn,
        "overlay_source": "seg",
    }
    if skipped:
        meta["overlay_skipped_rois"] = skipped
    return Image.fromarray(canvas, mode="RGB"), meta


def mask_outline(mask: np.ndarray) -> np.ndarray:
    """Boundary pixels of a 2-D boolean mask (4-neighbour)."""
    padded = np.pad(mask, 1, constant_values=False)
    interior = (
        padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:]
    )
    outline: np.ndarray = mask & ~interior
    return outline


def _frame_slice_indices(seg: SEG, scan: MedImage) -> list[int | None]:
    depth = int(scan.GetSize()[2])
    out: list[int | None] = []
    for position in seg.ref_indices:
        z_idx = scan.TransformPhysicalPointToIndex(tuple(float(v) for v in position))[2]
        out.append(z_idx if 0 <= z_idx < depth else None)
    return out


def _segment_by_label(seg: SEG, label: str | None) -> Any | None:
    if not label:
        return None
    for segment in seg.segments.values():
        if segment.label == label:
            return segment
    return None
