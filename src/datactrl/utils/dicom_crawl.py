"""Imagetools crawl helpers: series paths, metadata enrichment, preview slices."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
from imgtools.coretypes import Scan
from imgtools.dicom.crawl import Crawler
from imgtools.dicom.dicom_reader import load_dicom

from datactrl.utils.dicom_meta import classify_series_kind, suggested_slice_index
from datactrl.utils.rtstruct import (
    preferred_roi_name,
    rt_series_kind,
    summarize_rt_file,
)

# Not in the imgtools crawl extractor tag set; read from the first instance header.
_HEADER_TAGS = (
    "SeriesDescription",
    "StudyDescription",
    "ImageType",
    "ImagePositionPatient",
)
_MAX_ROI_NAMES = 40


def enrich_series_meta(
    crawler: Crawler,
    series_uid: str,
    modality: str | None,
) -> dict[str, Any]:
    """Build harness series meta from crawl results (no folder rescan)."""
    sub = _primary_subseries(crawler, series_uid)
    if not sub:
        return {}

    flat = _flatten_crawl_entry(sub)
    files = ordered_series_files(crawler, series_uid)
    instance_count = len(files) or _instance_count(sub)
    header = _read_header_tags(files[0]) if files else {}

    description = _meta_str(flat, "SeriesDescription") or _meta_str(
        header, "SeriesDescription"
    )
    protocol = _meta_str(flat, "ProtocolName")
    study = _meta_str(flat, "StudyDescription") or _meta_str(header, "StudyDescription")
    body = _meta_str(flat, "BodyPartExamined")
    image_type = _meta_image_type(flat) or _meta_image_type(header)
    mod = _meta_str(flat, "Modality") or (modality or "")
    blob = " ".join(part for part in (description, protocol, study) if part)

    if rt_series_kind(mod) or mod.upper() == "SEG":
        if mod.upper() == "SEG":
            mask_meta = _seg_meta(flat)
        else:
            mask_meta = summarize_rt_file(files[0], crawl_meta=flat) if files else {}
        mask_meta["instance_count"] = instance_count
        if description:
            mask_meta.setdefault("SeriesDescription", description)
        if protocol:
            mask_meta.setdefault("ProtocolName", protocol)
        if study:
            mask_meta.setdefault("StudyDescription", study)
        return {key: value for key, value in mask_meta.items() if value is not None}

    z_first = _z_from_header(header)
    z_last = _z_from_header(_read_header_tags(files[-1])) if len(files) > 1 else z_first
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
        "series_kind": classify_series_kind(image_type, blob, mod),
        "instance_count": instance_count,
        "z_first_mm": z_first,
        "z_last_mm": z_last,
        "z_span_mm": z_span,
        "z_superior_mm": z_superior,
        "FrameOfReferenceUID": _meta_str(flat, "FrameOfReferenceUID"),
    }
    meta["suggested_slice_index"] = suggested_slice_index(meta)
    return {key: value for key, value in meta.items() if value is not None}


# ---------------------------------------------------------------------------
# Series files and slices
# ---------------------------------------------------------------------------


def gdcm_series_file_names(series_dir: Path | str, series_uid: str) -> list[str]:
    """Same ordering ``read_dicom_series`` uses (GDCM series reader)."""
    import SimpleITK as sitk

    return list(
        sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(series_dir), series_uid)
    )


def ordered_series_files(crawler: Crawler, series_uid: str) -> list[str]:
    """Instance paths: GDCM order when resolvable, else crawl ``instances`` map."""
    try:
        series_dir = resolve_series_directory(crawler, series_uid)
    except ValueError:
        return _paths_from_crawl_instances(crawler, series_uid)

    try:
        names = gdcm_series_file_names(series_dir, series_uid)
        if names:
            return names
    except Exception:
        pass
    return _paths_from_crawl_instances(crawler, series_uid)


def load_series_scan(crawler: Crawler, series_uid: str) -> tuple[Scan, list[str]]:
    """Load a crawled series as an imgtools ``Scan`` plus its ordered file paths."""
    files = ordered_series_files(crawler, series_uid)
    if not files:
        raise ValueError(f"No instances for series {series_uid!r}")
    series_dir = resolve_series_directory(crawler, series_uid)
    scan = Scan.from_dicom(str(series_dir), series_id=series_uid, file_names=files)
    return scan, files


def scan_slice_count(scan: Scan) -> int:
    """Number of slices along the scan's third axis."""
    return int(scan.GetSize()[2]) if scan.GetDimension() >= 3 else 1


def scan_slice(scan: Scan, slice_index: int) -> tuple[np.ndarray, int, int]:
    """Return one 2-D slice (clamped index) and the slice count."""
    array, _geometry = scan.to_numpy(view=True)
    frame = np.squeeze(np.asarray(array))
    if frame.ndim == 2:
        return frame, 0, 1
    if frame.ndim >= 3:
        n_slices = int(frame.shape[0])
        idx = max(0, min(n_slices - 1, int(slice_index)))
        slice_2d = np.squeeze(frame[idx])
        if slice_2d.ndim != 2:
            raise ValueError("Could not extract a 2-D slice from this series")
        return slice_2d, idx, n_slices
    raise ValueError("Could not render this series")


def scan_slice_z_mm(scan: Scan, slice_index: int) -> float | None:
    """Patient z (mm) of a slice origin, from the scan geometry."""
    if scan.GetDimension() < 3:
        return None
    return round(
        float(scan.TransformIndexToPhysicalPoint((0, 0, int(slice_index)))[2]), 3
    )


def resolve_series_directory(crawler: Crawler, series_uid: str) -> Path:
    """Resolve the on-disk folder for a crawled series."""
    folder = str(crawler.get_series_info(series_uid).get("folder", "")).strip()
    if not folder:
        sub = _primary_subseries(crawler, series_uid)
        folder = str(sub.get("folder", "")).strip() if sub else ""
    if not folder:
        raise ValueError(f"Missing series folder for {series_uid!r}")

    dicom_dir = Path(crawler.dicom_dir)
    resolved = try_resolve_series_path(dicom_dir, folder)
    if resolved is not None:
        return resolved
    raise ValueError(
        f"Could not resolve series folder {folder!r} for {series_uid!r} "
        f"under {dicom_dir}"
    )


# ---------------------------------------------------------------------------
# Crawl entry helpers
# ---------------------------------------------------------------------------


def _primary_subseries(crawler: Crawler, series_uid: str) -> dict[str, Any]:
    raw = crawler.crawl_db_raw.get(series_uid)
    if not raw:
        return {}
    best: dict[str, Any] = {}
    best_count = -1
    for entry in raw.values():
        if not isinstance(entry, dict):
            continue
        instances = entry.get("instances") or {}
        count = len(instances) if isinstance(instances, dict) else 0
        if count > best_count:
            best_count = count
            best = entry
    if best:
        return best
    first = next(iter(raw.values()), None)
    return first if isinstance(first, dict) else {}


def _paths_from_crawl_instances(crawler: Crawler, series_uid: str) -> list[str]:
    sub = _primary_subseries(crawler, series_uid)
    instances = sub.get("instances")
    folder = str(sub.get("folder", "")).strip()
    if not isinstance(instances, dict) or not folder:
        return []
    dicom_dir = Path(crawler.dicom_dir)
    series_dir = try_resolve_series_path(dicom_dir, folder)
    if series_dir is None:
        return []
    names = sorted(str(name) for name in instances.values() if name)
    return [str(series_dir / name) for name in names]


def _instance_count(sub: Mapping[str, Any]) -> int:
    instances = sub.get("instances")
    if isinstance(instances, dict):
        return len(instances)
    return 0


def _flatten_crawl_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in entry.items():
        if key in ("instances", "folder"):
            continue
        out[key] = _unwrap_meta_value(value)
    return out


def _unwrap_meta_value(value: Any) -> Any:
    if value is None:
        return None
    if hasattr(value, "value"):
        return _unwrap_meta_value(value.value)
    if isinstance(value, (list, tuple)):
        return [_unwrap_meta_value(item) for item in value]
    return value


def _meta_str(flat: Mapping[str, Any], key: str) -> str | None:
    for candidate in (key, key.lower(), key.upper()):
        if candidate not in flat:
            continue
        text = _scalar_str(flat.get(candidate))
        if text:
            return text
    return None


def _meta_image_type(flat: Mapping[str, Any]) -> list[str] | None:
    raw = flat.get("ImageType") or flat.get("image_type")
    raw = _unwrap_meta_value(raw)
    if raw is None:
        return None
    if isinstance(raw, str):
        parts = [part.strip() for part in raw.split("\\") if part.strip()]
        return parts or None
    if isinstance(raw, (list, tuple)):
        parts = [str(part).strip() for part in raw if str(part).strip()]
        return parts or None
    text = str(raw).strip()
    return [text] if text else None


def _scalar_str(value: Any) -> str | None:
    value = _unwrap_meta_value(value)
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None
    return text


def _seg_meta(flat: Mapping[str, Any]) -> dict[str, Any]:
    names = [str(name) for name in flat.get("ROINames") or [] if name]
    return {
        "Modality": "SEG",
        "series_kind": "seg",
        "roi_names": names[:_MAX_ROI_NAMES] or None,
        "NumROIs": len(names),
        "preferred_roi": preferred_roi_name(names),
        "referenced_series_uid": _meta_str(flat, "ReferencedSeriesUID"),
        "FrameOfReferenceUID": _meta_str(flat, "FrameOfReferenceUID"),
    }


def _read_header_tags(path: str) -> dict[str, Any]:
    try:
        ds = load_dicom(path, stop_before_pixels=True)
    except Exception:
        return {}
    out: dict[str, Any] = {}
    for tag in _HEADER_TAGS:
        value = getattr(ds, tag, None)
        if value is None:
            continue
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
            value = list(value)
        out[tag] = value
    return out


def _z_from_header(header: Mapping[str, Any]) -> float | None:
    ipp = header.get("ImagePositionPatient")
    if ipp is None:
        return None
    try:
        return float(ipp[2])
    except (TypeError, ValueError, IndexError):
        return None


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def try_resolve_series_path(dicom_dir: Path, folder: str) -> Path | None:
    """Resolve a series folder, including paths relative to ``dicom_dir``."""
    if not folder:
        return None
    path = Path(folder)
    candidates: list[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.append(dicom_dir / path)
        candidates.append(dicom_dir.parent / path)
        if path.parts and path.parts[0] == dicom_dir.name:
            candidates.append(dicom_dir.joinpath(*path.parts[1:]))
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None
