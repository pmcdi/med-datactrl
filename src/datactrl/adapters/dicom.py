"""DICOM adapter backed by med-imagetools, including series reference graphs."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, TypeVar

import pandas as pd
from imgtools.coretypes import SEG, RTStructureSet, Scan
from imgtools.dicom.crawl import Crawler
from imgtools.dicom.interlacer import Interlacer
from imgtools.utils.interlacer_utils import SeriesNode
from PIL import Image

from datactrl.adapters.base import resolve_fetch_options
from datactrl.errors import InvalidFetchOptions, ResourceNotFound
from datactrl.types import (
    DescribeResult,
    FetchResult,
    Link,
    Modality,
    ResourceRef,
    ViewSpec,
    make_resource_id,
    parse_resource_id,
)
from datactrl.utils.dicom_crawl import (
    enrich_series_meta,
    load_series_scan,
    ordered_series_files,
    scan_slice,
    scan_slice_count,
    scan_slice_z_mm,
    try_resolve_series_path,
)
from datactrl.utils.dicom_meta import merged_series_metadata, slice_index_for_z
from datactrl.utils.dicom_views import adapter_view_catalog, views_for_series
from datactrl.utils.rtstruct import (
    load_structure_set,
    overlay_rtstruct,
    rtstruct_slice_index,
)
from datactrl.utils.seg import load_seg, overlay_seg, seg_slice_index

_T = TypeVar("_T")
_SCAN_CACHE_SIZE = 2
_MASK_CACHE_SIZE = 4
_MASK_KINDS = ("rtstruct", "seg")

MaskSource = RTStructureSet | SEG


class DicomAdapter:
    """DICOM modality adapter backed by med-imagetools."""

    modality = Modality.DICOM
    version = "1"

    def __init__(self) -> None:
        """Create an empty adapter; call ``ingest`` before describe/fetch."""
        self._root: Path | None = None
        self._crawler: Crawler | None = None
        self._interlacer: Interlacer | None = None
        self._refs: dict[str, ResourceRef] = {}
        self._scans: OrderedDict[str, tuple[Scan, list[str]]] = OrderedDict()
        self._masks: OrderedDict[str, MaskSource] = OrderedDict()

    def views(self) -> Sequence[ViewSpec]:
        """Return supported DICOM fetch views."""
        return adapter_view_catalog()

    # -----------------------------------------------------------------------
    # Ingest
    # -----------------------------------------------------------------------

    def ingest(self, root: Path, paths: Sequence[Path]) -> list[ResourceRef]:
        """Index DICOM series via imagetools.

        ``paths`` is a presence hint: if empty, skip crawl. When any DICOM
        files were classified, imagetools crawls ``root`` (series truth source).
        """
        self._refs.clear()
        self._scans.clear()
        self._masks.clear()
        self._crawler = None
        self._interlacer = None
        self._root = root.resolve()

        if not paths:
            return []

        crawler = Crawler(self._root)
        crawler.crawl()
        self._crawler = crawler
        self._interlacer = Interlacer(crawler.index)

        refs: list[ResourceRef] = []
        for series_uid, node in sorted(
            self._interlacer.series_nodes.items(),
            key=lambda item: item[0],
        ):
            resource_id = make_resource_id(Modality.DICOM, series_uid)
            crawler_meta = _safe_series_info(crawler, series_uid)

            child_modalities = sorted({child.Modality for child in node.children})
            child_ids = [
                make_resource_id(Modality.DICOM, child.SeriesInstanceUID)
                for child in node.children
            ]
            patient_id = str(node.PatientID) if node.PatientID is not None else None

            raw_folder = str(node.folder or crawler_meta.get("folder", ""))
            series_path = _try_resolve_series_path(self._root, raw_folder)
            rel_path = (
                _rel(series_path, self._root) if series_path is not None else raw_folder
            )
            overlay = enrich_series_meta(crawler, series_uid, node.Modality)
            parent_uid = _clean_uid(node.ReferencedSeriesUID) or overlay.get(
                "referenced_series_uid"
            )

            label = _series_label(node.Modality, overlay, crawler_meta, series_uid)

            ref = ResourceRef(
                id=resource_id,
                modality=Modality.DICOM,
                path=rel_path,
                label=str(label),
                patient_id=patient_id,
                meta={
                    "Modality": node.Modality,
                    "PatientID": patient_id,
                    "StudyInstanceUID": node.StudyInstanceUID,
                    "SeriesInstanceUID": series_uid,
                    "ReferencedSeriesUID": parent_uid,
                    "child_series_ids": child_ids,
                    "child_modalities": child_modalities,
                    **overlay,
                },
            )
            self._refs[resource_id] = ref
            refs.append(ref)
        for ref in refs:
            self._annotate_rtstruct_slice(ref)
        return refs

    # -----------------------------------------------------------------------
    # Describe and fetch
    # -----------------------------------------------------------------------

    def describe(self, resource_id: str) -> DescribeResult:
        """Return metadata summary and structural links for one series."""
        ref = self._require_ref(resource_id)
        view_specs = list(views_for_series(ref.meta))
        summary = _series_metadata(self, ref)
        links = self._structural_links(resource_id)
        return DescribeResult(
            id=resource_id,
            modality=Modality.DICOM,
            label=ref.label,
            path=ref.path,
            patient_id=ref.patient_id,
            views=[spec.name for spec in view_specs],
            view_specs=view_specs,
            summary=summary,
            links=links,
        )

    def fetch(
        self, resource_id: str, view: str | None = None, **opts: Any
    ) -> FetchResult:
        """Fetch ``metadata`` or a single-slice ``preview`` for a series."""
        ref = self._require_ref(resource_id)
        resolved_view, merged = resolve_fetch_options(
            views_for_series(ref.meta), view, opts, default_view="metadata"
        )
        if resolved_view == "preview":
            return self._fetch_preview(resource_id, merged)
        return self._fetch_metadata(resource_id)

    def _fetch_metadata(self, resource_id: str) -> FetchResult:
        """Build the metadata fetch payload for one series."""
        ref = self._require_ref(resource_id)
        series_uid = _series_uid(resource_id)
        links = self._structural_links(resource_id)
        return FetchResult(
            id=resource_id,
            modality=Modality.DICOM,
            view="metadata",
            data={
                "metadata": _series_metadata(self, ref),
                "links": [link.to_dict() for link in links],
            },
            bounds={"truncated": False},
            provenance={
                "SeriesInstanceUID": series_uid,
                "path": self._require_ref(resource_id).path,
                "adapter": "dicom",
                "adapter_version": self.version,
            },
        )

    # -----------------------------------------------------------------------
    # Preview
    # -----------------------------------------------------------------------

    def _fetch_preview(self, resource_id: str, opts: dict[str, Any]) -> FetchResult:
        """Render one slice of a series as a base64 PNG preview."""
        from datactrl.utils.image import to_llm_image_payload, to_uint8_rgb

        ref = self._require_ref(resource_id)
        kind = str(ref.meta.get("series_kind") or "")
        if kind in {"rtplan", "rtdose"}:
            raise InvalidFetchOptions(
                f"{kind.upper()} has no pixel preview; fetch view=metadata"
            )
        if kind in _MASK_KINDS or str(ref.meta.get("Modality") or "").upper() == "SEG":
            return self._fetch_mask_preview(resource_id, opts)

        series_uid = _series_uid(resource_id)
        scan = self._scan(series_uid)
        n_slices = scan_slice_count(scan)
        slice_index = _resolve_slice_index(ref, n_slices, opts)
        _check_slice_index(slice_index, n_slices)

        frame, slice_index, n_slices = scan_slice(scan, slice_index)
        if frame.size == 0:
            raise ValueError(f"Could not render DICOM series: {resource_id!r}")

        window = _resolve_window(ref, opts)
        image = to_uint8_rgb(frame, window=window)
        overlay_meta: dict[str, Any] = {"overlay": False}
        roi = _raw_roi(opts)
        if roi and not _truthy(opts.get("overlay")):
            raise InvalidFetchOptions("roi requires overlay=true on an image series")
        if _truthy(opts.get("overlay")):
            image, overlay_meta = self._overlay_on_image_series(
                ref, scan, slice_index, image, roi=roi
            )

        payload = to_llm_image_payload(
            id=resource_id,
            kind="dicom",
            image=image,
            meta={
                **overlay_meta,
                "slice_index": slice_index,
                "n_slices": n_slices,
                "slice_z_mm": scan_slice_z_mm(scan, slice_index),
                "window": window,
            },
        )
        return FetchResult(
            id=resource_id,
            modality=Modality.DICOM,
            view="preview",
            data=payload,
            bounds={
                "slice_index": slice_index,
                "n_slices": n_slices,
                "window": window,
                "truncated": False,
            },
            provenance={
                "SeriesInstanceUID": series_uid,
                "path": ref.path,
                "adapter": "dicom",
                "adapter_version": self.version,
            },
        )

    def _fetch_mask_preview(
        self, resource_id: str, opts: dict[str, Any]
    ) -> FetchResult:
        """Preview the referenced CT/MR with this RTSTRUCT's or SEG's ROIs overlaid."""
        from datactrl.utils.image import to_llm_image_payload, to_uint8_rgb

        ref = self._require_ref(resource_id)
        modality = str(ref.meta.get("Modality") or "").upper() or "RTSTRUCT"
        image_ref = self._referenced_image_series(ref)
        if image_ref is None:
            raise InvalidFetchOptions(
                f"{modality} has no referenced CT/MR series in this case; "
                "fetch view=metadata for ROI names"
            )

        ct_uid = _series_uid(image_ref.id)
        try:
            scan = self._scan(ct_uid)
        except ValueError as exc:
            raise InvalidFetchOptions(
                f"Referenced image series {image_ref.id!r} has no readable instances"
            ) from exc

        overlay_meta: dict[str, Any] = {"overlay": False}
        mask: MaskSource | None = None
        try:
            mask = self._mask(ref)
        except Exception as exc:
            overlay_meta["overlay_error"] = f"could not load {modality}: {exc}"[:200]

        roi = _match_roi(opts, _roi_names(mask, ref))
        n_slices = scan_slice_count(scan)
        if "slice_index" in opts and opts["slice_index"] is not None:
            slice_index = int(opts["slice_index"])
        else:
            located = (
                _mask_slice_index(mask, scan, roi or ref.meta.get("preferred_roi"))
                if mask is not None
                else None
            )
            if located is not None:
                slice_index = located
            elif (
                ref.meta.get("suggested_slice_index") is not None and modality != "SEG"
            ):
                slice_index = int(ref.meta["suggested_slice_index"])
            else:
                slice_index = _resolve_slice_index(image_ref, n_slices, {})
        _check_slice_index(slice_index, n_slices)

        frame, slice_index, n_slices = scan_slice(scan, slice_index)
        if frame.size == 0:
            raise ValueError(f"Could not render referenced series: {image_ref.id!r}")

        window = _resolve_window(image_ref, opts)
        image = to_uint8_rgb(frame, window=window)
        if mask is not None:
            image, overlay_meta = _draw_mask(
                image, scan, slice_index, mask, roi_filter=[roi] if roi else None
            )
        overlay_meta["referenced_series_id"] = image_ref.id
        overlay_meta["preferred_roi"] = ref.meta.get("preferred_roi")
        if roi:
            overlay_meta["roi"] = roi

        payload = to_llm_image_payload(
            id=resource_id,
            kind="dicom",
            image=image,
            meta={
                **overlay_meta,
                "slice_index": slice_index,
                "n_slices": n_slices,
                "slice_z_mm": scan_slice_z_mm(scan, slice_index),
                "window": window,
                "roi_names": list(ref.meta.get("roi_names") or []),
            },
        )
        return FetchResult(
            id=resource_id,
            modality=Modality.DICOM,
            view="preview",
            data=payload,
            bounds={
                "slice_index": slice_index,
                "n_slices": n_slices,
                "window": window,
                "truncated": False,
            },
            provenance={
                "SeriesInstanceUID": _series_uid(resource_id),
                "ReferencedSeriesUID": ct_uid,
                "path": ref.path,
                "adapter": "dicom",
                "adapter_version": self.version,
            },
        )

    # -----------------------------------------------------------------------
    # Linked RTSTRUCT and SEG
    # -----------------------------------------------------------------------

    def _annotate_rtstruct_slice(self, ref: ResourceRef) -> None:
        """Map the preferred ROI z onto the referenced image series.

        PET defaults to that slice too: a whole-body geometry guess usually
        misses the contoured uptake.
        """
        if str(ref.meta.get("series_kind") or "") != "rtstruct":
            return
        z_mm = ref.meta.get("preferred_roi_z_mm")
        image_ref = self._referenced_image_series(ref)
        if z_mm is None or image_ref is None:
            return
        idx = slice_index_for_z(image_ref.meta, float(z_mm))
        if idx is None:
            return
        ref.meta["suggested_slice_index"] = idx
        if str(image_ref.meta.get("Modality") or "").upper() == "PT":
            image_ref.meta["suggested_slice_index"] = idx

    def _overlay_on_image_series(
        self,
        image_ref: ResourceRef,
        scan: Scan,
        slice_index: int,
        image: Image.Image,
        *,
        roi: str | None = None,
    ) -> tuple[Image.Image, dict[str, Any]]:
        """Draw a linked RTSTRUCT (else SEG) onto an already-rendered image slice."""
        mask_ref = self._mask_ref_for_image(image_ref)
        if mask_ref is None:
            if roi:
                raise InvalidFetchOptions(
                    "No linked RTSTRUCT or SEG to select an ROI from"
                )
            return image, {"overlay": False}
        try:
            mask = self._mask(mask_ref)
        except Exception as exc:
            return image, {
                "overlay": False,
                "overlay_resource_id": mask_ref.id,
                "overlay_error": f"could not load overlay source: {exc}"[:200],
            }
        matched = _match_roi({"roi": roi} if roi else {}, _roi_names(mask, mask_ref))
        drawn, meta = _draw_mask(
            image,
            scan,
            slice_index,
            mask,
            roi_filter=[matched] if matched else None,
        )
        meta["overlay_resource_id"] = mask_ref.id
        if matched:
            meta["roi"] = matched
        return drawn, meta

    def _referenced_image_series(self, rt_ref: ResourceRef) -> ResourceRef | None:
        """Resolve the CT/MR/PT series this mask points at.

        The interlacer graph wins over the header reference.
        """
        series_uid = _series_uid(rt_ref.id)
        parent_uid: str | None = None
        try:
            node = self._require_interlacer().series_nodes[series_uid]
            parent_uid = _clean_uid(node.ReferencedSeriesUID)
        except KeyError:
            pass
        if not parent_uid:
            parent_uid = _clean_uid(
                rt_ref.meta.get("ReferencedSeriesUID")
                or rt_ref.meta.get("referenced_series_uid")
            )
        if not parent_uid:
            return None
        found = self._refs.get(make_resource_id(Modality.DICOM, parent_uid))
        if found is None:
            return None
        if str(found.meta.get("Modality") or "").upper() not in {"CT", "MR", "PT"}:
            return None
        return found

    def _mask_ref_for_image(self, image_ref: ResourceRef) -> ResourceRef | None:
        """Find an RTSTRUCT (preferred) or SEG that references this image series."""
        image_uid = _series_uid(image_ref.id)
        children = [
            child
            for child_id in image_ref.meta.get("child_series_ids") or []
            if (child := self._refs.get(child_id)) is not None
        ]
        for kind in _MASK_KINDS:
            for child in children:
                if str(child.meta.get("series_kind") or "") == kind:
                    return child
            for other in self._refs.values():
                if str(other.meta.get("series_kind") or "") != kind:
                    continue
                parent = other.meta.get("ReferencedSeriesUID") or other.meta.get(
                    "referenced_series_uid"
                )
                if parent and str(parent) == image_uid:
                    return other
        return None

    def _files_for_ref(self, ref: ResourceRef) -> list[str]:
        series_uid = _series_uid(ref.id)
        try:
            return ordered_series_files(self._require_crawler(), series_uid)
        except ValueError:
            return []

    def _scan(self, series_uid: str) -> Scan:
        """Load (or reuse) the imgtools Scan for an image series."""
        scan, _files = _cached(
            self._scans,
            series_uid,
            lambda: load_series_scan(self._require_crawler(), series_uid),
            _SCAN_CACHE_SIZE,
        )
        return scan

    def _mask(self, ref: ResourceRef) -> MaskSource:
        """Load (or reuse) the RTStructureSet / SEG behind a mask series."""
        series_uid = _series_uid(ref.id)

        def load() -> MaskSource:
            files = self._files_for_ref(ref)
            if not files:
                raise ValueError(f"no readable instances for {ref.id!r}")
            crawl_meta = _crawl_mask_meta(self._require_crawler(), series_uid)
            if str(ref.meta.get("series_kind") or "") == "seg":
                return load_seg(files[0], crawl_meta)
            return load_structure_set(files[0], crawl_meta)

        return _cached(self._masks, series_uid, load, _MASK_CACHE_SIZE)

    # -----------------------------------------------------------------------
    # Reference graph
    # -----------------------------------------------------------------------

    def tree(self) -> dict[str, Any]:
        """Return the patient/series reference forest as JSON."""
        interlacer = self._require_interlacer()
        patients: dict[str, list[dict[str, Any]]] = {}
        for root in interlacer.root_nodes:
            patients.setdefault(root.PatientID, []).append(_node_to_tree(root))
        return {
            "patients": [
                {"PatientID": patient_id, "roots": roots}
                for patient_id, roots in sorted(patients.items())
            ],
            "series_count": len(interlacer.series_nodes),
            "valid_queries": list(interlacer.valid_queries),
        }

    def related(self, resource_id: str) -> dict[str, Any]:
        """Return one-hop parent/children for a series."""
        self._require_ref(resource_id)
        series_uid = _series_uid(resource_id)
        node = self._require_interlacer().series_nodes[series_uid]
        parent_uid = _clean_uid(node.ReferencedSeriesUID)
        references: list[dict[str, Any]] = []
        if parent_uid and parent_uid in self._require_interlacer().series_nodes:
            parent = self._require_interlacer().series_nodes[parent_uid]
            references.append(_node_summary(parent, relation="parent"))

        referenced_by = [
            _node_summary(child, relation="child") for child in node.children
        ]
        return {
            "id": resource_id,
            "Modality": node.Modality,
            "ReferencedSeriesUID": parent_uid,
            "references": references,
            "referenced_by": referenced_by,
            "links": [link.to_dict() for link in self._structural_links(resource_id)],
        }

    def groups(self, query: str | None = None) -> dict[str, Any]:
        """Return linked modality groups from Interlacer.

        ``query`` is an imagetools query string such as ``CT,SEG``.
        When omitted, returns all valid query results for this case.
        """
        interlacer = self._require_interlacer()

        if query:
            chains = interlacer.query(query)
            queries = [query]
        else:
            chains = []
            queries = list(interlacer.valid_queries)
            for valid in queries:
                chains.extend(interlacer.query(valid))

        groups = [_chain_to_group(chain) for chain in chains]
        return {
            "query": query,
            "valid_queries": list(interlacer.valid_queries),
            "queries_used": queries,
            "groups": groups,
            "group_count": len(groups),
        }

    def _structural_links(self, resource_id: str) -> list[Link]:
        """Derive parent/child structural links from the interlacer graph."""
        series_uid = _series_uid(resource_id)
        node = self._require_interlacer().series_nodes[series_uid]
        links: list[Link] = []
        parent_uid = _clean_uid(node.ReferencedSeriesUID)
        if parent_uid and parent_uid in self._require_interlacer().series_nodes:
            parent = self._require_interlacer().series_nodes[parent_uid]
            links.append(
                Link(
                    from_id=resource_id,
                    to_id=make_resource_id(Modality.DICOM, parent_uid),
                    relation="references",
                    source="structural",
                    meta={"Modality": parent.Modality},
                )
            )
        for child in node.children:
            links.append(
                Link(
                    from_id=resource_id,
                    to_id=make_resource_id(Modality.DICOM, child.SeriesInstanceUID),
                    relation="referenced_by",
                    source="structural",
                    meta={"Modality": child.Modality},
                )
            )
        return links

    def _require_ref(self, resource_id: str) -> ResourceRef:
        """Look up an indexed series ref or raise ResourceNotFound."""
        try:
            return self._refs[resource_id]
        except KeyError as exc:
            raise ResourceNotFound(f"Unknown DICOM series: {resource_id!r}") from exc

    def _require_crawler(self) -> Crawler:
        """Return the imagetools crawler or raise when DICOM was not indexed."""
        if self._crawler is None:
            raise ResourceNotFound("No DICOM data indexed for this case")
        return self._crawler

    def _require_interlacer(self) -> Interlacer:
        """Return the imagetools interlacer or raise when DICOM was not indexed."""
        if self._interlacer is None:
            raise ResourceNotFound("No DICOM data indexed for this case")
        return self._interlacer


# ---------------------------------------------------------------------------
# Labels and slice options
# ---------------------------------------------------------------------------


def _series_metadata(adapter: DicomAdapter, ref: ResourceRef) -> dict[str, Any]:
    series_uid = _series_uid(ref.id)
    crawler_meta = _safe_series_info(adapter._require_crawler(), series_uid)
    return merged_series_metadata(ref.meta, crawler_meta)


def _series_label(
    modality: str | None,
    overlay: dict[str, Any],
    crawler_meta: dict[str, Any],
    series_uid: str,
) -> str:
    """Human label: series description, or ROI names for unlabeled RTSTRUCT/SEG."""
    kind = str(overlay.get("series_kind") or "")
    if kind in _MASK_KINDS:
        prefix = "RTSTRUCT" if kind == "rtstruct" else "SEG"
        struct_label = str(overlay.get("structure_set_label") or "").strip()
        names = [str(name) for name in (overlay.get("roi_names") or []) if name]
        if struct_label and struct_label.lower() not in {"unapproved", "unknown"}:
            return f"{prefix} ({struct_label})"
        if names:
            shown = ", ".join(names[:4])
            if len(names) > 4:
                shown += ", …"
            return f"{prefix} ({shown})"
        return prefix
    return str(
        overlay.get("SeriesDescription")
        or _clean_uid(crawler_meta.get("SeriesDescription"))
        or modality
        or series_uid[:16]
    )


def _resolve_slice_index(ref: ResourceRef, n_slices: int, opts: dict[str, Any]) -> int:
    if "slice_index" in opts and opts["slice_index"] is not None:
        return int(opts["slice_index"])
    suggested = ref.meta.get("suggested_slice_index")
    if suggested is not None:
        return max(0, min(n_slices - 1, int(suggested)))
    return n_slices // 2


def _check_slice_index(slice_index: int, n_slices: int) -> None:
    if slice_index < 0 or slice_index >= n_slices:
        raise InvalidFetchOptions(
            f"slice_index={slice_index} out of range; n_slices={n_slices} "
            f"(valid: 0..{n_slices - 1})"
        )


def _resolve_window(image_ref: ResourceRef, opts: dict[str, Any]) -> str:
    window = str(opts.get("window") or "")
    if window:
        return window
    modality = str(image_ref.meta.get("Modality") or "").upper()
    if modality == "CT":
        return "abdomen"
    if modality == "PT":
        return "pet"
    return "minmax"


# ---------------------------------------------------------------------------
# ROI selection and overlay
# ---------------------------------------------------------------------------


def _mask_slice_index(mask: MaskSource, scan: Scan, roi_name: Any) -> int | None:
    name = str(roi_name) if roi_name else None
    if isinstance(mask, SEG):
        return seg_slice_index(mask, scan, name)
    return rtstruct_slice_index(mask, scan, name)


def _draw_mask(
    image: Image.Image,
    scan: Scan,
    slice_index: int,
    mask: MaskSource,
    *,
    roi_filter: list[str] | None = None,
) -> tuple[Image.Image, dict[str, Any]]:
    if isinstance(mask, SEG):
        return overlay_seg(image, scan, slice_index, mask, roi_filter=roi_filter)
    return overlay_rtstruct(image, scan, slice_index, mask, roi_filter=roi_filter)


def _roi_names(mask: MaskSource | None, ref: ResourceRef) -> list[str]:
    if isinstance(mask, SEG):
        return [str(label) for label in mask.labels if label]
    if isinstance(mask, RTStructureSet):
        return [str(name) for name in mask.roi_names if name]
    return [str(name) for name in (ref.meta.get("roi_names") or []) if name]


def _raw_roi(opts: dict[str, Any]) -> str | None:
    raw = opts.get("roi")
    if raw is None:
        return None
    text = str(raw).strip()
    return text or None


def _match_roi(opts: dict[str, Any], names: list[str]) -> str | None:
    """Case-insensitive exact match, or None when ``roi`` was not requested."""
    text = _raw_roi(opts)
    if text is None:
        return None
    for name in names:
        if name.lower() == text.lower():
            return name
    available = ", ".join(names) or "(none)"
    raise InvalidFetchOptions(f"Unknown ROI {text!r}; available: {available}")


# ---------------------------------------------------------------------------
# Caches and ids
# ---------------------------------------------------------------------------


def _crawl_mask_meta(crawler: Crawler, series_uid: str) -> dict[str, Any]:
    meta = _safe_series_info(crawler, series_uid)
    return {
        key: value for key, value in meta.items() if key not in {"instances", "folder"}
    }


def _cached(
    cache: OrderedDict[str, _T], key: str, load: Callable[[], _T], size: int
) -> _T:
    if key in cache:
        cache.move_to_end(key)
        return cache[key]
    value = load()
    cache[key] = value
    while len(cache) > size:
        cache.popitem(last=False)
    return value


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _series_uid(resource_id: str) -> str:
    """Extract the DICOM series UID from a resource id."""
    modality, key = parse_resource_id(resource_id)
    if modality == Modality.DICOM:
        return key
    return resource_id


def _safe_series_info(crawler: Crawler, series_id: str) -> dict[str, Any]:
    """Return series metadata, or an empty dict when imagetools lookup fails."""
    try:
        return dict(crawler.get_series_info(series_id))
    except Exception:
        return {}


def _clean_uid(value: Any) -> str | None:
    """Normalize a DICOM UID value, dropping null/NaN sentinels."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


# ---------------------------------------------------------------------------
# Interlacer serialization
# ---------------------------------------------------------------------------


def _node_summary(node: SeriesNode, *, relation: str) -> dict[str, Any]:
    """Serialize one interlacer node with a relation label."""
    return {
        "id": make_resource_id(Modality.DICOM, node.SeriesInstanceUID),
        "series_id": node.SeriesInstanceUID,
        "Modality": node.Modality,
        "PatientID": node.PatientID,
        "StudyInstanceUID": node.StudyInstanceUID,
        "folder": node.folder,
        "relation": relation,
    }


def _node_to_tree(node: SeriesNode) -> dict[str, Any]:
    """Recursively serialize a series node and its children."""
    return {
        "id": make_resource_id(Modality.DICOM, node.SeriesInstanceUID),
        "series_id": node.SeriesInstanceUID,
        "Modality": node.Modality,
        "PatientID": node.PatientID,
        "StudyInstanceUID": node.StudyInstanceUID,
        "folder": node.folder,
        "ReferencedSeriesUID": _clean_uid(node.ReferencedSeriesUID),
        "children": [_node_to_tree(child) for child in node.children],
    }


def _chain_to_group(chain: list[SeriesNode]) -> dict[str, Any]:
    """Serialize one linked modality chain from Interlacer."""
    return {
        "modalities": [node.Modality for node in chain],
        "series": [
            {
                "id": make_resource_id(Modality.DICOM, node.SeriesInstanceUID),
                "series_id": node.SeriesInstanceUID,
                "Modality": node.Modality,
                "folder": node.folder,
                "ReferencedSeriesUID": _clean_uid(node.ReferencedSeriesUID),
            }
            for node in chain
        ],
    }


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


def _try_resolve_series_path(dicom_dir: Path, folder: str) -> Path | None:
    """Best-effort resolve of an imagetools series folder under ``dicom_dir``."""
    return try_resolve_series_path(dicom_dir, folder)


def _rel(path: Path, root: Path) -> str:
    """Return ``path`` relative to ``root``, or its string form if outside."""
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)
