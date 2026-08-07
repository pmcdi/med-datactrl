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
        description="Single-slice PNG preview; set slice_index (default: middle)",
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
            meta = _safe_series_info(crawler, series_uid)

            parent_uid = _clean_uid(node.ReferencedSeriesUID)
            child_modalities = sorted({child.Modality for child in node.children})
            child_ids = [
                make_resource_id(Modality.DICOM, child.SeriesInstanceUID)
                for child in node.children
            ]
            label = str(
                meta.get("SeriesDescription") or node.Modality or series_uid[:16]
            )
            patient_id = str(node.PatientID) if node.PatientID is not None else None

            raw_folder = str(node.folder or meta.get("folder", ""))
            series_path = _try_resolve_series_path(self._root, raw_folder)
            rel_path = (
                _rel(series_path, self._root) if series_path is not None else raw_folder
            )

            ref = ResourceRef(
                id=resource_id,
                modality=Modality.DICOM,
                path=rel_path,
                label=label,
                patient_id=patient_id,
                meta={
                    "Modality": node.Modality,
                    "PatientID": patient_id,
                    "StudyInstanceUID": node.StudyInstanceUID,
                    "SeriesInstanceUID": series_uid,
                    "ReferencedSeriesUID": parent_uid,
                    "child_series_ids": child_ids,
                    "child_modalities": child_modalities,
                },
            )
            self._refs[resource_id] = ref
            refs.append(ref)
        return refs

    def describe(self, resource_id: str) -> DescribeResult:
        """Return metadata summary and structural links for one series."""
        ref = self._require_ref(resource_id)
        series_uid = _series_uid(resource_id)
        meta = _safe_series_info(self._require_crawler(), series_uid)
        links = self._structural_links(resource_id)
        return DescribeResult(
            id=resource_id,
            modality=Modality.DICOM,
            label=ref.label,
            path=ref.path,
            patient_id=ref.patient_id,
            views=[spec.name for spec in _VIEWS],
            view_specs=list(_VIEWS),
            summary={
                "Modality": meta.get("Modality") or ref.meta.get("Modality"),
                "PatientID": meta.get("PatientID") or ref.meta.get("PatientID"),
                "StudyInstanceUID": meta.get("StudyInstanceUID")
                or ref.meta.get("StudyInstanceUID"),
                "SeriesInstanceUID": series_uid,
                "SeriesDescription": meta.get("SeriesDescription"),
                "instances": meta.get("instances") or meta.get("num_instances"),
                "metadata_keys": sorted(str(key) for key in meta.keys()),
            },
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
        self._require_ref(resource_id)
        series_uid = _series_uid(resource_id)
        meta = self._require_crawler().get_series_info(series_uid)
        links = self._structural_links(resource_id)
        return FetchResult(
            id=resource_id,
            modality=Modality.DICOM,
            view="metadata",
            data={
                "metadata": dict(meta),
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
        import numpy as np
        from imgtools.io.readers import read_dicom_series
        from imgtools.vizualize import ImageVisualizer

        from datactrl.utils.image import to_llm_image_payload, to_uint8_rgb

        self._require_ref(resource_id)
        series_uid = _series_uid(resource_id)
        series_path = _series_path(self._require_crawler(), series_uid)
        image, _ = read_dicom_series(path=str(series_path), series_id=series_uid)
        volume = np.asarray(ImageVisualizer.from_image(image).array)
        if volume.ndim != 3 or volume.size == 0:
            raise ValueError(f"Could not render DICOM series: {resource_id!r}")

        n_slices = int(volume.shape[0])
        if "slice_index" in opts and opts["slice_index"] is not None:
            slice_index = int(opts["slice_index"])
        else:
            slice_index = n_slices // 2

        if slice_index < 0 or slice_index >= n_slices:
            raise InvalidFetchOptions(
                f"slice_index={slice_index} out of range; n_slices={n_slices} "
                f"(valid: 0..{n_slices - 1})"
            )

        payload = to_llm_image_payload(
            id=resource_id,
            kind="dicom",
            image=to_uint8_rgb(volume[slice_index]),
            meta={
                "overlay": False,
                "slice_index": slice_index,
                "n_slices": n_slices,
                "window": "minmax",
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
                "truncated": False,
            },
            provenance={
                "SeriesInstanceUID": series_uid,
                "path": self._require_ref(resource_id).path,
                "adapter": "dicom",
                "adapter_version": self.version,
            },
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
