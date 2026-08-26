"""DICOM adapter backed by med-imagetools, including series reference graphs."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from imgtools.dicom.crawl import Crawler
from imgtools.dicom.interlacer import Interlacer
from imgtools.utils.interlacer_utils import SeriesNode

from datactrl.adapters.base import resolve_fetch_options
from datactrl.dicom_index import (
    public_dicom_meta,
    rank_series,
    slice_index_for_z,
    summarize_series,
)
from datactrl.rtstruct import overlay_contours
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

_VIEWS = (
    ViewSpec(
        name="metadata",
        description="DICOM series metadata plus structural references",
        defaults={},
        max={},
    ),
    ViewSpec(
        name="preview",
        description=(
            "Single-slice PNG preview; set slice_index "
            "(default: mm-from-superior on long-FOV series, otherwise middle). "
            "RTSTRUCT previews the referenced CT with contours overlaid; "
            "pass overlay=true on a CT/MR series to draw linked RTSTRUCT ROIs."
        ),
        defaults={},
        max={},
    ),
)


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

    def views(self) -> Sequence[ViewSpec]:
        """Return supported DICOM fetch views."""
        return _VIEWS

    def ingest(self, root: Path, paths: Sequence[Path]) -> list[ResourceRef]:
        """Index DICOM series via imagetools.

        ``paths`` is a presence hint: if empty, skip crawl. When any DICOM
        files were classified, imagetools crawls ``root`` (series truth source).
        """
        self._refs.clear()
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
            overlay = summarize_series(series_path, series_uid)
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

    def describe(self, resource_id: str) -> DescribeResult:
        """Return metadata summary and structural links for one series."""
        ref = self._require_ref(resource_id)
        series_uid = _series_uid(resource_id)
        summary = public_dicom_meta(
            ref.meta, _safe_series_info(self._require_crawler(), series_uid)
        )
        links = self._structural_links(resource_id)
        return DescribeResult(
            id=resource_id,
            modality=Modality.DICOM,
            label=ref.label,
            path=ref.path,
            patient_id=ref.patient_id,
            views=[spec.name for spec in _VIEWS],
            view_specs=list(_VIEWS),
            summary=summary,
            links=links,
        )

    def fetch(
        self, resource_id: str, view: str | None = None, **opts: Any
    ) -> FetchResult:
        """Fetch ``metadata`` or a single-slice ``preview`` for a series."""
        resolved_view, merged = resolve_fetch_options(
            _VIEWS, view, opts, default_view="metadata"
        )
        if resolved_view == "preview":
            return self._fetch_preview(resource_id, merged)
        return self._fetch_metadata(resource_id)

    def _fetch_metadata(self, resource_id: str) -> FetchResult:
        """Build the metadata fetch payload for one series."""
        ref = self._require_ref(resource_id)
        series_uid = _series_uid(resource_id)
        crawler_meta = self._require_crawler().get_series_info(series_uid)
        links = self._structural_links(resource_id)
        return FetchResult(
            id=resource_id,
            modality=Modality.DICOM,
            view="metadata",
            data={
                "metadata": public_dicom_meta(ref.meta, dict(crawler_meta)),
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

    def _fetch_preview(self, resource_id: str, opts: dict[str, Any]) -> FetchResult:
        """Render one slice of a series as a base64 PNG preview."""
        from datactrl.dicom_index import ordered_series_files
        from datactrl.utils.image import to_llm_image_payload, to_uint8_rgb

        ref = self._require_ref(resource_id)
        kind = str(ref.meta.get("series_kind") or "")
        if kind in {"rtplan", "rtdose"}:
            raise InvalidFetchOptions(
                f"{kind.upper()} has no pixel preview; fetch view=metadata"
            )
        if kind == "rtstruct":
            return self._fetch_rtstruct_preview(resource_id, opts)

        series_uid = _series_uid(resource_id)
        series_path = _series_path(self._require_crawler(), series_uid)
        files = ordered_series_files(series_path, series_uid)
        if not files:
            raise ValueError(f"Could not render DICOM series: {resource_id!r}")

        n_slices = len(files)
        slice_index = _resolve_slice_index(ref, n_slices, opts)
        if slice_index < 0 or slice_index >= n_slices:
            raise InvalidFetchOptions(
                f"slice_index={slice_index} out of range; n_slices={n_slices} "
                f"(valid: 0..{n_slices - 1})"
            )

        frame = _read_preview_frame(files, slice_index)
        if frame is None:
            raise ValueError(f"Could not render DICOM series: {resource_id!r}")
        if frame.ndim == 3:
            n_slices = int(frame.shape[0])
            slice_index = max(0, min(n_slices - 1, slice_index))
            frame = frame[slice_index]
        if frame.ndim != 2 or frame.size == 0:
            raise ValueError(f"Could not render DICOM series: {resource_id!r}")

        window = str(opts.get("window") or "")
        if not window:
            window = "abdomen" if str(ref.meta.get("Modality") or "").upper() == "CT" else "minmax"

        image = to_uint8_rgb(frame, window=window)
        overlay_meta: dict[str, Any] = {"overlay": False}
        if _truthy(opts.get("overlay")):
            image, overlay_meta = self._overlay_on_image_series(ref, files, slice_index, image)

        payload = to_llm_image_payload(
            id=resource_id,
            kind="dicom",
            image=image,
            meta={
                **overlay_meta,
                "slice_index": slice_index,
                "n_slices": n_slices,
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

    def _fetch_rtstruct_preview(
        self, resource_id: str, opts: dict[str, Any]
    ) -> FetchResult:
        """Preview the referenced CT/MR with this RTSTRUCT's contours overlaid."""
        from datactrl.dicom_index import ordered_series_files
        from datactrl.utils.image import to_llm_image_payload, to_uint8_rgb

        ref = self._require_ref(resource_id)
        ct_ref = self._referenced_image_series(ref)
        if ct_ref is None:
            raise InvalidFetchOptions(
                "RTSTRUCT has no referenced CT/MR series in this case; "
                "fetch view=metadata for ROI names"
            )

        ct_uid = _series_uid(ct_ref.id)
        ct_path = _series_path(self._require_crawler(), ct_uid)
        files = ordered_series_files(ct_path, ct_uid)
        if not files:
            raise InvalidFetchOptions(
                f"Referenced image series {ct_ref.id!r} has no readable instances"
            )

        n_slices = len(files)
        if "slice_index" in opts and opts["slice_index"] is not None:
            slice_index = int(opts["slice_index"])
        else:
            preferred_z = ref.meta.get("preferred_roi_z_mm")
            mapped = (
                slice_index_for_z(ct_ref.meta, float(preferred_z))
                if preferred_z is not None
                else None
            )
            if mapped is not None:
                slice_index = mapped
            else:
                slice_index = _resolve_slice_index(ct_ref, n_slices, {})

        if slice_index < 0 or slice_index >= n_slices:
            raise InvalidFetchOptions(
                f"slice_index={slice_index} out of range; n_slices={n_slices} "
                f"(valid: 0..{n_slices - 1})"
            )

        frame = _read_preview_frame(files, slice_index)
        if frame is None:
            raise ValueError(f"Could not render referenced series: {ct_ref.id!r}")
        if frame.ndim == 3:
            n_slices = int(frame.shape[0])
            slice_index = max(0, min(n_slices - 1, slice_index))
            frame = frame[slice_index]
        if frame.ndim != 2 or frame.size == 0:
            raise ValueError(f"Could not render referenced series: {ct_ref.id!r}")

        window = str(opts.get("window") or "")
        if not window:
            window = (
                "abdomen"
                if str(ct_ref.meta.get("Modality") or "").upper() == "CT"
                else "minmax"
            )

        image = to_uint8_rgb(frame, window=window)
        rt_files = self._files_for_ref(ref)
        overlay_meta: dict[str, Any] = {"overlay": False}
        if rt_files:
            image, overlay_meta = overlay_contours(image, files[slice_index], rt_files[0])
        overlay_meta["referenced_series_id"] = ct_ref.id
        overlay_meta["preferred_roi"] = ref.meta.get("preferred_roi")

        payload = to_llm_image_payload(
            id=resource_id,
            kind="dicom",
            image=image,
            meta={
                **overlay_meta,
                "slice_index": slice_index,
                "n_slices": n_slices,
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

    def _annotate_rtstruct_slice(self, ref: ResourceRef) -> None:
        """Map the preferred ROI z onto the referenced CT/MR instance index."""
        if str(ref.meta.get("series_kind") or "") != "rtstruct":
            return
        z_mm = ref.meta.get("preferred_roi_z_mm")
        ct_ref = self._referenced_image_series(ref)
        if z_mm is None or ct_ref is None:
            return
        idx = slice_index_for_z(ct_ref.meta, float(z_mm))
        if idx is not None:
            ref.meta["suggested_slice_index"] = idx

    def _overlay_on_image_series(
        self,
        image_ref: ResourceRef,
        files: list[str],
        slice_index: int,
        image,
    ):
        """Draw a linked RTSTRUCT onto an already-rendered CT/MR slice."""
        rt_ref = self._rtstruct_for_image(image_ref)
        if rt_ref is None:
            return image, {"overlay": False}
        rt_files = self._files_for_ref(rt_ref)
        if not rt_files:
            return image, {"overlay": False}
        drawn, meta = overlay_contours(image, files[slice_index], rt_files[0])
        meta["overlay_resource_id"] = rt_ref.id
        return drawn, meta

    def _referenced_image_series(self, rt_ref: ResourceRef) -> ResourceRef | None:
        """Resolve the CT/MR series an RTSTRUCT points at."""
        parent = rt_ref.meta.get("ReferencedSeriesUID") or rt_ref.meta.get(
            "referenced_series_uid"
        )
        if parent:
            resource_id = make_resource_id(Modality.DICOM, str(parent))
            found = self._refs.get(resource_id)
            if found is not None:
                return found
        frame = rt_ref.meta.get("FrameOfReferenceUID")
        if not frame:
            return None
        for other in self._refs.values():
            if other.id == rt_ref.id:
                continue
            if other.meta.get("FrameOfReferenceUID") != frame:
                continue
            if str(other.meta.get("Modality") or "").upper() in {"CT", "MR", "PT"}:
                return other
        return None

    def _rtstruct_for_image(self, image_ref: ResourceRef) -> ResourceRef | None:
        """Find an RTSTRUCT that references this image series."""
        for child_id in image_ref.meta.get("child_series_ids") or []:
            child = self._refs.get(child_id)
            if child is not None and str(child.meta.get("series_kind") or "") == "rtstruct":
                return child
        image_uid = _series_uid(image_ref.id)
        for other in self._refs.values():
            if str(other.meta.get("series_kind") or "") != "rtstruct":
                continue
            parent = other.meta.get("ReferencedSeriesUID") or other.meta.get(
                "referenced_series_uid"
            )
            if parent and str(parent) == image_uid:
                return other
        return None

    def _files_for_ref(self, ref: ResourceRef) -> list[str]:
        from datactrl.dicom_index import ordered_series_files

        series_uid = _series_uid(ref.id)
        try:
            series_path = _series_path(self._require_crawler(), series_uid)
        except ValueError:
            return []
        return ordered_series_files(series_path, series_uid)

    def select(
        self,
        query: str,
        *,
        limit: int = 5,
        include_localizers: bool = False,
    ) -> dict[str, Any]:
        """Rank indexed series for a clinical query (no pixels)."""
        if not self._refs:
            raise ResourceNotFound("No DICOM data indexed for this case")
        return rank_series(
            list(self._refs.values()),
            query,
            limit=limit,
            include_localizers=include_localizers,
        )

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


def _series_label(
    modality: str | None,
    overlay: dict[str, Any],
    crawler_meta: dict[str, Any],
    series_uid: str,
) -> str:
    """Human label: series description, or ROI names for unlabeled RTSTRUCT."""
    kind = str(overlay.get("series_kind") or "")
    if kind == "rtstruct":
        struct_label = str(overlay.get("structure_set_label") or "").strip()
        names = [str(name) for name in (overlay.get("roi_names") or []) if name]
        if struct_label and struct_label.lower() not in {"unapproved", "unknown"}:
            return f"RTSTRUCT ({struct_label})"
        if names:
            shown = ", ".join(names[:4])
            if len(names) > 4:
                shown += ", …"
            return f"RTSTRUCT ({shown})"
        return "RTSTRUCT"
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


def _read_preview_frame(files: list[str], slice_index: int):
    """Read one instance (or a multi-frame volume) without loading the series."""
    import numpy as np
    import SimpleITK as sitk

    path = files[0] if len(files) == 1 else files[slice_index]
    image = sitk.ReadImage(path)
    frame = np.asarray(sitk.GetArrayFromImage(image))
    frame = np.squeeze(frame)
    return frame


def _try_resolve_series_path(dicom_dir: Path, folder: str) -> Path | None:
    """Best-effort resolve of an imagetools series folder under ``dicom_dir``."""
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


def _series_path(crawler: Crawler, series_id: str) -> Path:
    """Resolve a series folder from imagetools metadata.

    Imagetools sometimes stores ``folder`` relative to a parent crawl root
    (e.g. ``KiTS-00008/CT_...`` while ``dicom_dir`` is already ``.../KiTS-00008``).
    """
    metadata = crawler.get_series_info(series_id)
    folder = str(metadata.get("folder", ""))
    if not folder:
        raise ValueError(f"Missing series folder for {series_id!r}")

    dicom_dir = Path(crawler.dicom_dir)
    resolved = _try_resolve_series_path(dicom_dir, folder)
    if resolved is not None:
        return resolved

    raise ValueError(
        f"Could not resolve series folder {folder!r} for {series_id!r} "
        f"under {dicom_dir}"
    )


def _rel(path: Path, root: Path) -> str:
    """Return ``path`` relative to ``root``, or its string form if outside."""
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)
