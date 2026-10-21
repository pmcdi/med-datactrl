"""Per-series DICOM fetch view specs (modality / series_kind aware)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from datactrl.types import ViewSpec

_METADATA_VIEW = ViewSpec(
    name="metadata",
    description="DICOM series metadata plus structural references",
    defaults={},
    max={},
)

_IMAGE_MODALITIES = frozenset({"CT", "MR", "PT", "NM", "US"})
_NO_PREVIEW_KINDS = frozenset({"rtplan", "rtdose"})
_NO_PREVIEW_MODALITIES = frozenset({"RTPLAN", "RTDOSE", "SR", "PR", "KO"})


def adapter_view_catalog() -> Sequence[ViewSpec]:
    """Union of views the adapter may expose (per-resource list is narrower)."""
    return (
        _METADATA_VIEW,
        ViewSpec(
            name="preview",
            description=(
                "Single-slice PNG when this series supports it; "
                "see describe().view_specs"
            ),
            defaults={},
            max={},
        ),
    )


def preview_supported(meta: Mapping[str, Any]) -> bool:
    """Whether this series can render a pixel preview."""
    kind = str(meta.get("series_kind") or "")
    if kind in _NO_PREVIEW_KINDS:
        return False
    modality = str(meta.get("Modality") or "").upper()
    if modality in _NO_PREVIEW_MODALITIES:
        return False
    if kind in {"rtstruct", "seg"} or modality == "SEG":
        return True
    return modality in _IMAGE_MODALITIES or kind in {
        "acquisition",
        "localizer",
        "derived",
    }


def preview_view_spec(meta: Mapping[str, Any]) -> ViewSpec:
    """Build the preview ViewSpec for one series."""
    kind = str(meta.get("series_kind") or "")
    modality = str(meta.get("Modality") or "").upper()
    # Mask series render their referenced image; the adapter picks its window.
    defaults: dict[str, Any] = (
        {}
        if kind in {"rtstruct", "seg"} or modality == "SEG"
        else {"window": _default_window(modality)}
    )

    if kind == "rtstruct":
        description = (
            "Referenced CT/MR/PT slice with this RTSTRUCT's contours overlaid. "
            "Optional slice_index; default is the preferred ROI's median contour "
            "slice, or the named roi's slice when roi is set. "
            "roi limits the drawing to one structure (case-insensitive)."
        )
    elif kind == "seg" or modality == "SEG":
        description = (
            "Referenced CT/MR/PT slice with this SEG's segment outlines overlaid. "
            "Optional slice_index; default is the slice where the preferred "
            "segment is largest, or the named roi when roi is set. "
            "roi limits the drawing to one segment (case-insensitive)."
        )
    elif modality == "CT":
        description = (
            "Single-slice PNG; slice_index defaults from series geometry "
            "(long-FOV: mm-from-superior, else middle). "
            "CT window presets: abdomen, liver, lung, bone, brain, or minmax. "
            "Set overlay=true to draw a linked RTSTRUCT's ROIs or SEG's segments. "
            "Optional roi draws only that structure."
        )
    elif modality == "MR":
        description = (
            "Single-slice PNG; slice_index defaults from suggested_slice_index. "
            "Uses minmax scaling unless window is set. "
            "Set overlay=true to draw a linked RTSTRUCT's ROIs or SEG's segments. "
            "Optional roi draws only that structure."
        )
    elif modality == "PT":
        description = (
            "Single-slice PNG. When an RTSTRUCT references this series, "
            "slice_index defaults to that structure's preferred ROI; otherwise "
            "the series-geometry suggestion. window='pet' clips hot voxels "
            "(99.5th percentile) so uptake stays visible. "
            "Set overlay=true to draw the linked RTSTRUCT's ROIs. "
            "Optional roi draws only that structure."
        )
    else:
        description = (
            "Single-slice PNG preview when pixel data is available; "
            "optional slice_index and window (minmax for non-CT)."
        )

    return ViewSpec(
        name="preview",
        description=description,
        defaults=defaults,
        max={},
    )


def _default_window(modality: str) -> str:
    if modality == "CT":
        return "abdomen"
    if modality == "PT":
        return "pet"
    return "minmax"


def views_for_series(meta: Mapping[str, Any]) -> Sequence[ViewSpec]:
    """Metadata always; preview only when supported."""
    specs: list[ViewSpec] = [_METADATA_VIEW]
    if preview_supported(meta):
        specs.append(preview_view_spec(meta))
    return specs
