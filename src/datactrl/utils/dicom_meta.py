"""DICOM series classification, slice-index heuristics, and LLM-safe metadata."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from datactrl.utils.rtstruct import rt_series_kind

LONG_FOV_MM = 500.0
ABDOMEN_FROM_SUPERIOR_MM = 80.0
CHEST_FROM_SUPERIOR_MM = 100.0
HEAD_FROM_SUPERIOR_MM = 50.0
PELVIS_FROM_SUPERIOR_MM = 400.0
EXTREMITY_FROM_INFERIOR_MM = 200.0
_WHOLE_BODY_FROM_SUPERIOR_MM = {
    "head": 50.0,
    "chest": 180.0,
    "abdomen": 400.0,
    "pelvis": 550.0,
}
_REGION_PRIORITY = ("abdomen", "chest", "neck", "head", "pelvis", "extremity")

_PUBLIC_META_KEYS = (
    "Modality",
    "SeriesDescription",
    "ProtocolName",
    "StudyDescription",
    "BodyPartExamined",
    "ImageType",
    "series_kind",
    "instance_count",
    "z_first_mm",
    "z_last_mm",
    "z_span_mm",
    "z_superior_mm",
    "suggested_slice_index",
    "roi_names",
    "NumROIs",
    "structure_set_label",
    "preferred_roi",
    "preferred_roi_z_mm",
    "roi_z_min_mm",
    "roi_z_max_mm",
    "roi_extraction_errors",
    "RTPlanLabel",
    "RTPlanName",
    "DoseType",
    "DoseUnits",
    "DoseSummationType",
    "FrameOfReferenceUID",
    "PatientID",
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "ReferencedSeriesUID",
    "child_series_ids",
    "child_modalities",
)

_LOCALIZER_TOKENS = (
    "localizer",
    "localiser",
    "scout",
    "topogram",
    "survey",
    "locator",
    "3-plane",
    "3 plane",
    "loc ",
)
_CALIBRATION_TOKENS = (
    "calibration",
    "calibrat",
    "coil survey",
    "sense ref",
    "refscan",
    "reference scan",
    "b1 map",
    "b1map",
)
_DERIVED_TOKENS = (
    "screen save",
    "screensave",
    "screen-save",
    "secondary capture",
    "captured",
    "volume rendered",
    "3d recon",
    "testbolus",
    "test bolus",
    "bolus track",
    "smartprep",
    "smart prep",
)

_REGION_TOKENS: dict[str, tuple[str, ...]] = {
    "abdomen": (
        "abdomen",
        "abdominal",
        "abd",
        "liver",
        "hepatic",
        "hepatocellular",
        "hepato",
        "hepat",
        "gallbladder",
        "pancreas",
        "spleen",
        "kidney",
        "renal",
        "hcc",
        "cirrhosis",
        "hep",
    ),
    "chest": (
        "chest",
        "thorax",
        "thoracic",
        "lung",
        "cardiac",
        "heart",
        "mediastinum",
        "pulmonary",
    ),
    "neck": (
        "neck",
        "larynx",
        "laryngeal",
        "glottic",
        "glottis",
        "hnc",
        "oropharynx",
        "oropharyngeal",
        "hypopharynx",
        "hypopharyngeal",
        "nasopharynx",
        "vocal cord",
        "vocalcord",
        "head and neck",
        "hn&n",
    ),
    "head": ("brain", "head", "skull", "neuro", "cranial"),
    "pelvis": ("pelvis", "pelvic"),
    "extremity": (
        "runoff",
        "extremity",
        "thigh",
        "calf",
        "leg",
        "foot",
        "ankle",
        "lower ext",
        "upper ext",
        "femur",
        "popliteal",
        "tibial",
    ),
    "whole_body": ("whole body", "wholebody", "wb pet", "petct wb"),
}

_NON_IMAGE_KINDS = frozenset({"rtstruct", "rtplan", "rtdose", "seg"})


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def infer_regions(text: str) -> frozenset[str]:
    """Return anatomic regions mentioned in ``text``."""
    blob = (text or "").lower()
    found = {
        region
        for region, tokens in _REGION_TOKENS.items()
        if any(_has_token(blob, token) for token in tokens)
    }
    return frozenset(found)


def classify_series_kind(
    image_type: Sequence[str] | str | None,
    blob: str,
    modality: str | None = None,
) -> str:
    """Classify a series as localizer, derived, calibration, RT, SEG, or acquisition."""
    rt_kind = rt_series_kind(modality)
    if rt_kind:
        return rt_kind
    if str(modality or "").strip().upper() == "SEG":
        return "seg"
    tokens = _image_type_tokens(image_type)
    lowered = (blob or "").lower()
    if any(token in {"LOCALIZER", "LOCALISER"} for token in tokens):
        return "localizer"
    if any(_has_token(lowered, token) for token in _LOCALIZER_TOKENS):
        return "localizer"
    if any(token == "DERIVED" for token in tokens) or any(
        _has_token(lowered, token) for token in _DERIVED_TOKENS
    ):
        return "derived"
    if any(_has_token(lowered, token) for token in _CALIBRATION_TOKENS):
        return "calibration"
    return "acquisition"


# ---------------------------------------------------------------------------
# Slice index
# ---------------------------------------------------------------------------


def suggested_slice_index(
    meta: Mapping[str, Any],
    *,
    regions: frozenset[str] | None = None,
) -> int:
    """Pick a preview index from geometry, not a fraction of slice count."""
    n = max(_as_int(meta.get("instance_count"), 1), 1)
    kind = str(meta.get("series_kind") or "")
    if kind in _NON_IMAGE_KINDS:
        if meta.get("suggested_slice_index") is not None:
            return max(0, _as_int(meta.get("suggested_slice_index"), 0))
        return 0
    if n <= 2:
        return 0

    if kind == "localizer":
        return 0

    z_first = _as_float(meta.get("z_first_mm"))
    z_last = _as_float(meta.get("z_last_mm"))
    z_span = _as_float(meta.get("z_span_mm"))
    if z_span is None and z_first is not None and z_last is not None:
        z_span = abs(z_last - z_first)

    long_fov = z_span is not None and z_span >= LONG_FOV_MM
    if not long_fov or z_first is None or z_last is None:
        return n // 2

    target_z = _target_z_mm(meta, regions, z_first, z_last)
    return _index_for_z(z_first, z_last, n, target_z)


def slice_index_for_z(meta: Mapping[str, Any], z_mm: float) -> int | None:
    """Map a patient-z coordinate onto this series' instance index."""
    n = max(_as_int(meta.get("instance_count"), 1), 1)
    z_first = _as_float(meta.get("z_first_mm"))
    z_last = _as_float(meta.get("z_last_mm"))
    if z_first is None or z_last is None:
        return None
    return _index_for_z(z_first, z_last, n, z_mm)


# ---------------------------------------------------------------------------
# Public metadata
# ---------------------------------------------------------------------------


def merged_series_metadata(
    ref_meta: Mapping[str, Any],
    crawler_meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """LLM-safe metadata: crawl row then harness ref (ref wins on overlap)."""
    return public_dicom_meta(ref_meta, crawler_meta)


def public_dicom_meta(
    ref_meta: Mapping[str, Any],
    crawler_meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """LLM-safe series metadata: tags and counts, never SOP→file maps."""
    out: dict[str, Any] = {}
    for source in (crawler_meta or {}, ref_meta):
        for key in _PUBLIC_META_KEYS:
            if key not in source or source[key] is None:
                continue
            value = source[key]
            if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                continue
            if isinstance(value, str) and value.strip().lower() in {"", "nan", "none"}:
                continue
            out[key] = _jsonable(value)
    if "instance_count" in out:
        out["instance_count"] = _as_int(out["instance_count"], 0)
    return out


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _primary_region(regions: frozenset[str] | None) -> str | None:
    if not regions:
        return "abdomen"
    for region in _REGION_PRIORITY:
        if region in regions:
            return region
    return None


def _target_z_mm(
    meta: Mapping[str, Any],
    regions: frozenset[str] | None,
    z_first: float,
    z_last: float,
) -> float:
    z_sup = max(z_first, z_last)
    z_inf = min(z_first, z_last)
    region = _primary_region(regions) or "abdomen"
    blob = " ".join(
        part
        for part in (
            _clean_text(meta.get("SeriesDescription")),
            _clean_text(meta.get("ProtocolName")),
            _clean_text(meta.get("StudyDescription")),
        )
        if part
    )
    series_regions = infer_regions(blob)
    if "whole_body" in series_regions and region in _WHOLE_BODY_FROM_SUPERIOR_MM:
        offset = _WHOLE_BODY_FROM_SUPERIOR_MM[region]
        return _clamp(z_sup - offset, z_inf, z_sup)
    if region == "extremity":
        return _clamp(z_inf + EXTREMITY_FROM_INFERIOR_MM, z_inf, z_sup)
    if region == "pelvis":
        return _clamp(z_sup - PELVIS_FROM_SUPERIOR_MM, z_inf, z_sup)
    if region == "chest":
        return _clamp(z_sup - CHEST_FROM_SUPERIOR_MM, z_inf, z_sup)
    if region == "head":
        return _clamp(z_sup - HEAD_FROM_SUPERIOR_MM, z_inf, z_sup)
    return _clamp(z_sup - ABDOMEN_FROM_SUPERIOR_MM, z_inf, z_sup)


def _index_for_z(z_first: float, z_last: float, n: int, target_z: float) -> int:
    last_i = n - 1
    span = z_last - z_first
    if abs(span) < 1e-6:
        return last_i // 2
    t = (target_z - z_first) / span
    return max(0, min(last_i, int(round(t * last_i))))


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _image_type_tokens(image_type: Sequence[str] | str | None) -> list[str]:
    if not image_type:
        return []
    if isinstance(image_type, str):
        return [part.strip().upper() for part in image_type.split("\\") if part.strip()]
    return [str(part).strip().upper() for part in image_type if str(part).strip()]


def _has_token(text: str, token: str) -> bool:
    needle = token.lower().strip()
    if not needle or not text:
        return False
    if " " in needle or any(ch in needle for ch in "/-_"):
        return needle in text
    return re.search(rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])", text) is not None


def _clean_text(value: Any) -> str:
    cleaned = _clean_scalar(value)
    return cleaned.lower() if cleaned else ""


def _clean_scalar(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None
    return text


def _as_int(value: Any, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        return [_jsonable(item) for item in value]
    return str(value)
