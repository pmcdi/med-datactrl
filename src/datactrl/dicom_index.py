"""DICOM series summaries and deterministic ranking for retrieval.

Reads a small set of headers per series (not pixels, not SOP maps) and ranks
series from tags + geometry. Ranking is lexicon-based, not case-specific.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from datactrl.rtstruct import rt_series_kind, summarize_rt_file
from datactrl.types import ResourceRef

# Long cranio-caudal coverage (mm). Dedicated organ series stay well below this;
# runoff / whole-body stacks typically exceed it. Organ height is roughly fixed
# in mm, so long-FOV slice picks use a physical offset, not a fraction of N.
LONG_FOV_MM = 500.0
# mm from the superior (max z, LPS) end toward the feet for mid-organ cuts.
ABDOMEN_FROM_SUPERIOR_MM = 80.0
CHEST_FROM_SUPERIOR_MM = 100.0
HEAD_FROM_SUPERIOR_MM = 50.0
PELVIS_FROM_SUPERIOR_MM = 400.0
EXTREMITY_FROM_INFERIOR_MM = 200.0
# Extra mm from the head when the series itself includes the whole body.
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
    "structure_set_label",
    "preferred_roi",
    "preferred_roi_z_mm",
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

_ORGAN_REGIONS = frozenset({"abdomen", "chest", "neck", "head"})

_MODALITY_PATTERNS: tuple[tuple[str, str], ...] = (
    ("CT", r"\b(?:ct|cta|ctp)\b"),
    ("MR", r"\b(?:mr|mri)\b"),
    ("PT", r"\b(?:pt|pet)\b"),
    ("US", r"\b(?:us|ultrasound)\b"),
    ("NM", r"\bnm\b"),
    ("RTSTRUCT", r"\brt[\s-]?struct\b"),
    ("RTPLAN", r"\brtplan\b"),
    ("RTDOSE", r"\brtdose\b"),
)
_CONTOUR_RE = re.compile(
    r"\b(?:contours?|contoured|rt[\s-]?struct|structure set|"
    r"rois?|gtv\w*|ptv\w*|ctv\w*|overlay|delineat\w*|outlined|"
    r"oars?|organs at risk|target volumes?)\b",
    re.I,
)
_RT_KIND_SET = frozenset({"rtstruct", "rtplan", "rtdose"})


@dataclass(frozen=True)
class QueryIntent:
    """Parsed anatomy/modality intent from a free-text query."""

    modalities: frozenset[str]
    regions: frozenset[str]
    raw: str
    contours: bool = False


def parse_query_intent(query: str) -> QueryIntent:
    """Extract requested modalities and anatomic regions from ``query``."""
    text = (query or "").lower()
    modalities = {
        modality
        for modality, pattern in _MODALITY_PATTERNS
        if re.search(pattern, text)
    }
    return QueryIntent(
        modalities=frozenset(modalities),
        regions=infer_regions(text),
        raw=query or "",
        contours=bool(_CONTOUR_RE.search(text)),
    )


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
    """Classify a series as localizer, derived, calibration, RT, or acquisition."""
    rt_kind = rt_series_kind(modality)
    if rt_kind:
        return rt_kind
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


def summarize_series(
    series_path: Path | None,
    series_uid: str,
) -> dict[str, Any]:
    """Read first/last headers for tags, kind, counts, and geometry."""
    files = _ordered_files(series_path, series_uid)
    if not files:
        return {}

    first = _read_header(files[0])
    last = _read_header(files[-1]) if len(files) > 1 else first
    if first is None:
        return {"instance_count": len(files)}

    description = _header_str(first, "SeriesDescription")
    protocol = _header_str(first, "ProtocolName")
    study = _header_str(first, "StudyDescription")
    body = _header_str(first, "BodyPartExamined")
    image_type = _header_image_type(first)
    modality = _header_str(first, "Modality")
    blob = " ".join(part for part in (description, protocol, study) if part)
    if rt_series_kind(modality):
        rt_meta = summarize_rt_file(files[0])
        rt_meta["instance_count"] = len(files)
        if description:
            rt_meta.setdefault("SeriesDescription", description)
        if protocol:
            rt_meta.setdefault("ProtocolName", protocol)
        if study:
            rt_meta.setdefault("StudyDescription", study)
        return {key: value for key, value in rt_meta.items() if value is not None}
    z_first = _header_z_mm(first)
    z_last = _header_z_mm(last)
    z_span = None
    z_superior = None
    if z_first is not None and z_last is not None:
        z_span = abs(z_last - z_first)
        z_superior = max(z_first, z_last)
    elif z_first is not None:
        z_superior = z_first

    meta: dict[str, Any] = {
        "SeriesDescription": description,
        "ProtocolName": protocol,
        "StudyDescription": study,
        "BodyPartExamined": body,
        "ImageType": image_type,
        "series_kind": classify_series_kind(image_type, blob, modality),
        "instance_count": len(files),
        "z_first_mm": z_first,
        "z_last_mm": z_last,
        "z_span_mm": z_span,
        "z_superior_mm": z_superior,
        "FrameOfReferenceUID": _header_str(first, "FrameOfReferenceUID"),
    }
    meta["suggested_slice_index"] = suggested_slice_index(meta)
    return {key: value for key, value in meta.items() if value is not None}


def suggested_slice_index(
    meta: Mapping[str, Any],
    intent: QueryIntent | None = None,
) -> int:
    """Pick a preview index from geometry, not a fraction of slice count.

    Short-FOV series (dedicated organ): middle slice.
    Long-FOV series: interpolate ImagePositionPatient and aim a fixed mm
    offset from the superior or inferior end for the requested region.
    """
    n = max(_as_int(meta.get("instance_count"), 1), 1)
    kind = str(meta.get("series_kind") or "")
    if kind in _RT_KIND_SET:
        if meta.get("suggested_slice_index") is not None:
            return max(0, _as_int(meta.get("suggested_slice_index"), 0))
        return 0
    if n <= 2:
        return 0

    kind = str(meta.get("series_kind") or "")
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

    target_z = _target_z_mm(meta, intent, z_first, z_last)
    return _index_for_z(z_first, z_last, n, target_z)


def slice_index_for_z(meta: Mapping[str, Any], z_mm: float) -> int | None:
    """Map a patient-z coordinate onto this series' instance index."""
    n = max(_as_int(meta.get("instance_count"), 1), 1)
    z_first = _as_float(meta.get("z_first_mm"))
    z_last = _as_float(meta.get("z_last_mm"))
    if z_first is None or z_last is None:
        return None
    return _index_for_z(z_first, z_last, n, z_mm)


def choose_preview_window(modality: str | None, intent: QueryIntent | None) -> str:
    """CT window name for a query; non-CT stays min-max."""
    if str(modality or "").upper() != "CT":
        return "minmax"
    region = _primary_region(intent)
    if region == "abdomen":
        return "liver"
    if region == "chest":
        return "lung"
    if region == "neck":
        return "abdomen"
    if region == "head":
        return "brain"
    return "abdomen"


def _primary_region(intent: QueryIntent | None) -> str | None:
    if intent is None:
        return "abdomen"
    for region in _REGION_PRIORITY:
        if region in intent.regions:
            return region
    return None


def _target_z_mm(
    meta: Mapping[str, Any],
    intent: QueryIntent | None,
    z_first: float,
    z_last: float,
) -> float:
    z_sup = max(z_first, z_last)
    z_inf = min(z_first, z_last)
    region = _primary_region(intent) or "abdomen"
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


def rank_series(
    refs: Sequence[ResourceRef],
    query: str,
    *,
    limit: int = 5,
    include_localizers: bool = False,
) -> dict[str, Any]:
    """Rank DICOM refs for ``query`` and optionally select one series."""
    intent = parse_query_intent(query)
    pool = list(refs)
    if not include_localizers:
        acquisitions = [
            ref
            for ref in pool
            if str(ref.meta.get("series_kind") or "acquisition") != "localizer"
        ]
        if acquisitions:
            pool = acquisitions
    if intent.modalities:
        matching = [
            ref
            for ref in pool
            if str(ref.meta.get("Modality") or "").upper() in intent.modalities
            or (
                intent.contours
                and str(ref.meta.get("series_kind") or "") == "rtstruct"
            )
        ]
        if matching:
            pool = matching

    ranked = [_score_ref(ref, intent) for ref in pool]
    by_id = {ref.id: ref for ref in pool}
    ranked.sort(key=lambda item: _rank_sort_key(item, by_id[item["id"]], intent))
    candidates = ranked[: max(1, int(limit))]
    selected = candidates[0] if candidates else None
    confident = _is_confident(selected, intent, ranked)
    return {
        "query": query,
        "intent": {
            "modalities": sorted(intent.modalities),
            "regions": sorted(intent.regions),
            "contours": intent.contours,
        },
        "confident": confident,
        "selected": selected if confident else None,
        "candidates": candidates,
    }


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


def _score_ref(ref: ResourceRef, intent: QueryIntent) -> dict[str, Any]:
    meta = ref.meta
    kind = str(meta.get("series_kind") or "acquisition")
    modality = str(meta.get("Modality") or "").upper()
    series_blob = _series_blob(meta)
    study_blob = _clean_text(meta.get("StudyDescription"))
    body_blob = _clean_text(meta.get("BodyPartExamined"))
    series_regions = infer_regions(series_blob)
    study_regions = infer_regions(study_blob)
    body_regions = infer_regions(body_blob)

    score = 0.0
    reasons: list[str] = []
    warnings: list[str] = []

    if kind == "localizer":
        score -= 100
        reasons.append("localizer/scout")
    elif kind == "calibration":
        score -= 80
        reasons.append("calibration")
    elif kind == "derived":
        score -= 50
        reasons.append("derived/non-diagnostic")
    elif kind in _RT_KIND_SET:
        if intent.contours and kind == "rtstruct":
            score += 50
            reasons.append("RT structure set")
        else:
            score -= 25
            reasons.append("RT object (no pixel data)")
    else:
        score += 5
        reasons.append("acquisition")
        if intent.contours:
            score -= 10
            reasons.append("image series (contours live on RTSTRUCT)")

    if intent.modalities:
        if modality in intent.modalities:
            score += 25
            reasons.append(f"modality {modality}")
        elif intent.contours and kind == "rtstruct":
            score += 20
            reasons.append("RTSTRUCT linked to image query")
        else:
            score -= 15
            reasons.append(f"modality mismatch ({modality})")

    if intent.regions:
        if intent.regions & series_regions:
            score += 30
            reasons.append("series anatomy match")
        elif "extremity" in series_regions and intent.regions & _ORGAN_REGIONS:
            score -= 40
            reasons.append("series is runoff/extremity")
            warnings.append(
                "series protocol is runoff/extremity; BodyPartExamined is not trusted"
            )
        elif intent.regions & study_regions:
            score += 10
            reasons.append("study anatomy match")

        if "extremity" not in series_regions and intent.regions & body_regions:
            score += 5
            reasons.append("BodyPartExamined match")
        elif "extremity" in series_regions and intent.regions & body_regions:
            reasons.append("ignored BodyPartExamined (series text is extremity/runoff)")

        if (
            _has_token(series_blob, "delay")
            and "extremity" in (series_regions | study_regions)
            and intent.regions & _ORGAN_REGIONS
        ):
            score -= 15
            reasons.append("delay phase on runoff")

    n = _as_int(meta.get("instance_count"), 0)
    if n >= 10:
        score += 5
    elif n <= 1 and kind == "acquisition":
        score -= 10
        reasons.append("single frame")
    if n >= 800:
        score -= 4
        reasons.append("very large stack")

    z_span = _as_float(meta.get("z_span_mm"))
    if intent.regions & _ORGAN_REGIONS and z_span is not None and z_span >= LONG_FOV_MM:
        score -= 10
        reasons.append("long-FOV series")

    slice_index = suggested_slice_index(meta, intent)
    payload: dict[str, Any] = {
        "id": ref.id,
        "label": ref.label,
        "Modality": modality or None,
        "series_kind": kind,
        "instance_count": n,
        "SeriesDescription": meta.get("SeriesDescription"),
        "ProtocolName": meta.get("ProtocolName"),
        "StudyDescription": meta.get("StudyDescription"),
        "BodyPartExamined": meta.get("BodyPartExamined"),
        "suggested_slice_index": slice_index,
        "score": round(score, 2),
        "reasons": reasons,
        "warnings": warnings,
    }
    roi_names = meta.get("roi_names")
    if roi_names:
        payload["roi_names"] = list(roi_names)[:20]
        payload["preferred_roi"] = meta.get("preferred_roi")
    return payload


def _is_confident(
    selected: Mapping[str, Any] | None,
    intent: QueryIntent,
    _ranked: Sequence[Mapping[str, Any]],
) -> bool:
    if not selected:
        return False
    kind = str(selected.get("series_kind") or "")
    if kind in {"localizer", "calibration", "derived"}:
        return False
    modality = str(selected.get("Modality") or "").upper()
    if intent.modalities and modality not in intent.modalities:
        if not (intent.contours and kind == "rtstruct"):
            return False
    if float(selected.get("score") or 0) < 0:
        # Explicit modality (e.g. "CT slice") still selects the best matching
        # acquisition even when anatomy tags conflict (runoff vs liver).
        return bool(intent.modalities and modality in intent.modalities)
    return True


def _rank_sort_key(
    item: Mapping[str, Any],
    ref: ResourceRef,
    intent: QueryIntent,
) -> tuple[float, float, int, str]:
    """Higher score first; for organ queries, more-superior coverage next."""
    z_sup = _as_float(ref.meta.get("z_superior_mm"))
    organ_tie = (
        z_sup if (intent.regions & _ORGAN_REGIONS and z_sup is not None) else 0.0
    )
    return (
        -float(item["score"]),
        -organ_tie,
        -_as_int(item.get("instance_count"), 0),
        str(item["id"]),
    )


def _series_blob(meta: Mapping[str, Any]) -> str:
    parts = [
        _clean_text(meta.get("SeriesDescription")),
        _clean_text(meta.get("ProtocolName")),
    ]
    names = meta.get("roi_names") or []
    if names:
        parts.append(" ".join(str(name).lower() for name in names[:20] if name))
    return " ".join(part for part in parts if part)


def _ordered_files(series_path: Path | None, series_uid: str) -> list[str]:
    if series_path is None or not series_path.is_dir():
        return []
    try:
        import SimpleITK as sitk

        names = list(
            sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(series_path), series_uid)
        )
        if names:
            return names
    except Exception:
        pass
    files = sorted(
        str(path)
        for path in series_path.iterdir()
        if path.is_file() and path.suffix.lower() in {".dcm", ".dicom", ""}
    )
    return files


def ordered_series_files(series_path: Path | None, series_uid: str) -> list[str]:
    """Return GDCM-ordered instance paths for a series (no pixels)."""
    return _ordered_files(series_path, series_uid)


def _read_header(path: str) -> Any | None:
    try:
        import pydicom

        return pydicom.dcmread(path, stop_before_pixels=True, force=True)
    except Exception:
        return None


def _header_str(dataset: Any, name: str) -> str | None:
    if not hasattr(dataset, name):
        return None
    return _clean_scalar(getattr(dataset, name))


def _header_image_type(dataset: Any) -> list[str] | None:
    if not hasattr(dataset, "ImageType"):
        return None
    value = getattr(dataset, "ImageType")
    if value is None:
        return None
    if isinstance(value, str):
        parts = [part.strip() for part in value.split("\\") if part.strip()]
        return parts or None
    try:
        parts = [str(part).strip() for part in value if str(part).strip()]
    except TypeError:
        return [str(value)]
    return parts or None


def _header_z_mm(dataset: Any) -> float | None:
    if not hasattr(dataset, "ImagePositionPatient"):
        return None
    ipp = getattr(dataset, "ImagePositionPatient")
    try:
        return float(ipp[2])
    except (TypeError, ValueError, IndexError):
        return None


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
