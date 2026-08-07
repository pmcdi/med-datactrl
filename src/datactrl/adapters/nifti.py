"""NIfTI adapter with optional imagetools pattern-based crawl enrichment."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
)

log = logging.getLogger(__name__)

_VIEWS = (
    ViewSpec(
        name="metadata",
        description="NIfTI header geometry plus optional crawl match metadata",
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

_CRAWL_CONTROL_COLS = frozenset(
    {
        "filepath",
        "file_type",
        "reference_id",
        "reference_scan",
    }
)
_FINGERPRINT_COLS = frozenset(
    {
        "class",
        "hash",
        "size",
        "ndim",
        "nvoxels",
        "spacing",
        "origin",
        "direction",
        "min",
        "max",
        "sum",
        "mean",
        "std",
        "variance",
        "dtype_str",
        "dtype_numpy",
    }
)


@dataclass(frozen=True)
class NiftiCrawlOptions:
    """Optional pattern-driven crawl via imagetools ``parse_nifti_dir``.

    When omitted from ``Case``, NIfTI files are still discovered by extension.
    When provided, matched files are enriched and mask→scan links are derived.
    """

    scan_name_pattern: str
    mask_name_pattern: str | None = None
    extensions: list[str] | None = None
    deep: bool = False
    n_jobs: int = 1
    output_dir: Path | None = None
    dataset_name: str | None = None
    force: bool = False


@dataclass
class _NiftiRecord:
    ref: ResourceRef
    path: Path
    file_type: str | None = None
    placeholders: dict[str, str] = field(default_factory=dict)
    reference_id: str | None = None
    reference_scan: str | None = None
    fingerprint: dict[str, Any] = field(default_factory=dict)


class NiftiAdapter:
    """NIfTI modality adapter with optional imagetools crawl enrichment."""

    modality = Modality.NIFTI
    version = "1"

    def __init__(self, crawl: NiftiCrawlOptions | None = None) -> None:
        """Create an adapter; optional ``crawl`` enables pattern matching."""
        self._crawl = crawl
        self._root: Path | None = None
        self._records: dict[str, _NiftiRecord] = {}
        self._links_by_id: dict[str, list[Link]] = {}
        self._crawl_result: Any | None = None

    def views(self) -> Sequence[ViewSpec]:
        """Return supported NIfTI fetch views."""
        return _VIEWS

    def ingest(self, root: Path, paths: Sequence[Path]) -> list[ResourceRef]:
        """Index classified NIfTI files; optionally enrich via imagetools crawl."""
        self._records.clear()
        self._links_by_id.clear()
        self._crawl_result = None
        self._root = root.resolve()

        if not paths:
            return []

        by_rel: dict[str, Path] = {}
        for path in sorted(paths):
            rel = _rel(path, self._root)
            by_rel[rel] = path.resolve()

        crawl_rows: dict[str, dict[str, Any]] = {}
        if self._crawl is not None:
            crawl_rows = self._run_crawl(self._root)

        refs: list[ResourceRef] = []
        for rel, path in sorted(by_rel.items()):
            resource_id = make_resource_id(Modality.NIFTI, _path_key(rel))
            row = crawl_rows.get(rel, {})
            file_type = _optional_str(row.get("file_type"))
            placeholders = {
                str(key): str(value)
                for key, value in row.items()
                if key not in _CRAWL_CONTROL_COLS
                and key not in _FINGERPRINT_COLS
                and _optional_str(value) is not None
            }
            fingerprint = {
                key: _jsonable(value)
                for key, value in row.items()
                if key in _FINGERPRINT_COLS and value is not None and not _is_nan(value)
            }
            reference_id = _optional_str(row.get("reference_id"))
            reference_scan = _optional_str(row.get("reference_scan"))
            compressed = rel.lower().endswith(".nii.gz")
            meta: dict[str, Any] = {
                "compressed": compressed,
                "extension": ".nii.gz" if compressed else ".nii",
            }
            if file_type:
                meta["file_type"] = file_type
            if placeholders:
                meta["placeholders"] = placeholders
            if reference_id:
                meta["reference_id"] = reference_id
            if reference_scan:
                meta["reference_scan"] = reference_scan

            ref = ResourceRef(
                id=resource_id,
                modality=Modality.NIFTI,
                path=rel,
                label=path.name,
                patient_id=placeholders.get("PatientID")
                or placeholders.get("patient_id"),
                meta=meta,
            )
            self._records[resource_id] = _NiftiRecord(
                ref=ref,
                path=path,
                file_type=file_type,
                placeholders=placeholders,
                reference_id=reference_id,
                reference_scan=reference_scan or None,
                fingerprint=fingerprint,
            )
            refs.append(ref)

        self._build_links()
        return refs

    def describe(self, resource_id: str) -> DescribeResult:
        """Return geometry summary and structural links for one volume."""
        record = self._require(resource_id)
        geometry = self._geometry(record.path)
        summary: dict[str, Any] = {
            "compressed": record.ref.meta.get("compressed"),
            "extension": record.ref.meta.get("extension"),
            **geometry,
        }
        if record.file_type:
            summary["file_type"] = record.file_type
        if record.placeholders:
            summary["placeholders"] = dict(record.placeholders)
        if record.reference_id:
            summary["reference_id"] = record.reference_id
        if record.fingerprint:
            summary["fingerprint"] = dict(record.fingerprint)

        return DescribeResult(
            id=resource_id,
            modality=Modality.NIFTI,
            label=record.ref.label,
            path=record.ref.path,
            patient_id=record.ref.patient_id,
            views=[spec.name for spec in _VIEWS],
            view_specs=list(_VIEWS),
            summary=summary,
            links=list(self._links_by_id.get(resource_id, [])),
        )

    def fetch(
        self, resource_id: str, view: str | None = None, **opts: Any
    ) -> FetchResult:
        """Fetch ``metadata`` or a single-slice ``preview`` for a volume."""
        resolved_view, merged = resolve_fetch_options(
            _VIEWS, view, opts, default_view="metadata"
        )
        if resolved_view == "preview":
            return self._fetch_preview(resource_id, merged)
        return self._fetch_metadata(resource_id)

    def _fetch_metadata(self, resource_id: str) -> FetchResult:
        """Build the metadata fetch payload for one volume."""
        record = self._require(resource_id)
        geometry = self._geometry(record.path)
        links = self._links_by_id.get(resource_id, [])
        data: dict[str, Any] = {
            "metadata": {
                "path": record.ref.path,
                "compressed": record.ref.meta.get("compressed"),
                "extension": record.ref.meta.get("extension"),
                **geometry,
            },
            "links": [link.to_dict() for link in links],
        }
        if record.file_type:
            data["metadata"]["file_type"] = record.file_type
        if record.placeholders:
            data["metadata"]["placeholders"] = dict(record.placeholders)
        if record.reference_id:
            data["metadata"]["reference_id"] = record.reference_id
        if record.reference_scan:
            data["metadata"]["reference_scan"] = record.reference_scan
        if record.fingerprint:
            data["metadata"]["fingerprint"] = dict(record.fingerprint)

        return FetchResult(
            id=resource_id,
            modality=Modality.NIFTI,
            view="metadata",
            data=data,
            bounds={"truncated": False},
            provenance={
                "path": record.ref.path,
                "adapter": "nifti",
                "adapter_version": self.version,
            },
        )

    def _fetch_preview(self, resource_id: str, opts: dict[str, Any]) -> FetchResult:
        """Render one axial slice of a volume as a base64 PNG preview."""
        import numpy as np
        import SimpleITK as sitk

        from datactrl.utils.image import to_llm_image_payload, to_uint8_rgb

        record = self._require(resource_id)
        image = _read_image(record.path)
        volume = sitk.GetArrayFromImage(image)
        if volume.ndim == 4:
            # Take first component / timepoint for preview.
            volume = volume[0]
        if volume.ndim != 3 or volume.size == 0:
            raise InvalidFetchOptions(
                f"Unsupported NIfTI dimensionality for preview: "
                f"shape={tuple(volume.shape)} (resource={resource_id!r})"
            )

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
            kind="nifti",
            image=to_uint8_rgb(np.asarray(volume[slice_index])),
            meta={
                "slice_index": slice_index,
                "n_slices": n_slices,
                "window": "minmax",
                "axis": 0,
            },
        )
        return FetchResult(
            id=resource_id,
            modality=Modality.NIFTI,
            view="preview",
            data=payload,
            bounds={
                "slice_index": slice_index,
                "n_slices": n_slices,
                "truncated": False,
            },
            provenance={
                "path": record.ref.path,
                "adapter": "nifti",
                "adapter_version": self.version,
            },
        )

    def _run_crawl(self, root: Path) -> dict[str, dict[str, Any]]:
        """Call imagetools ``parse_nifti_dir`` and index rows by relative path."""
        from imgtools.nifti.crawl import parse_nifti_dir

        assert self._crawl is not None
        output_dir = (
            Path(self._crawl.output_dir)
            if self._crawl.output_dir is not None
            else root / ".datactrl" / "nifti"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            result = parse_nifti_dir(
                nifti_dir=root,
                scan_name_pattern=self._crawl.scan_name_pattern,
                mask_name_pattern=self._crawl.mask_name_pattern,
                output_dir=output_dir,
                dataset_name=self._crawl.dataset_name,
                force=self._crawl.force,
                extensions=self._crawl.extensions,
                deep=self._crawl.deep,
                n_jobs=self._crawl.n_jobs,
            )
        except ValueError as exc:
            # All files unmatched — keep layout-agnostic discovery.
            log.warning("NIfTI crawl matched no files: %s", exc)
            return {}
        except FileNotFoundError as exc:
            log.warning("NIfTI crawl found no files: %s", exc)
            return {}

        self._crawl_result = result
        rows: dict[str, dict[str, Any]] = {}
        for row in result.index.to_dict(orient="records"):
            filepath = _optional_str(row.get("filepath"))
            if filepath:
                rows[filepath] = row
        if result.unmatched_files:
            log.info(
                "NIfTI crawl left %d unmatched file(s)",
                len(result.unmatched_files),
            )
        return rows

    def _build_links(self) -> None:
        """Derive mask→scan structural links from crawl ``reference_scan``."""
        path_to_id = {
            record.ref.path: resource_id
            for resource_id, record in self._records.items()
        }
        for resource_id, record in self._records.items():
            links: list[Link] = []
            if record.file_type == "mask" and record.reference_scan:
                scan_id = path_to_id.get(record.reference_scan)
                if scan_id is not None:
                    links.append(
                        Link(
                            from_id=resource_id,
                            to_id=scan_id,
                            relation="references",
                            source="structural",
                            meta={
                                "file_type": "mask",
                                "reference_scan": record.reference_scan,
                                "reference_id": record.reference_id,
                            },
                        )
                    )
                    self._links_by_id.setdefault(scan_id, []).append(
                        Link(
                            from_id=scan_id,
                            to_id=resource_id,
                            relation="referenced_by",
                            source="structural",
                            meta={
                                "file_type": "scan",
                                "mask_path": record.ref.path,
                                "reference_id": record.reference_id,
                            },
                        )
                    )
            if links:
                self._links_by_id.setdefault(resource_id, []).extend(links)

    def _geometry(self, path: Path) -> dict[str, Any]:
        """Read basic image geometry without loading voxel stats."""
        image = _read_image(path)
        size = list(image.GetSize())
        # SimpleITK size is (x, y, z[, t]); array shape is reversed.
        shape = list(reversed(size))
        return {
            "shape": shape,
            "size": size,
            "spacing": list(image.GetSpacing()),
            "origin": list(image.GetOrigin()),
            "direction": list(image.GetDirection()),
            "dimension": int(image.GetDimension()),
            "pixel_id": image.GetPixelIDTypeAsString(),
        }

    def _require(self, resource_id: str) -> _NiftiRecord:
        """Look up an indexed volume or raise ResourceNotFound."""
        try:
            return self._records[resource_id]
        except KeyError as exc:
            raise ResourceNotFound(f"Unknown NIfTI resource: {resource_id!r}") from exc


def _read_image(path: Path) -> Any:
    """Read an image via imagetools helper when available, else SimpleITK."""
    try:
        from imgtools.nifti.crawl.parse_niftis import read_image

        return read_image(path)
    except Exception:
        import SimpleITK as sitk

        return sitk.ReadImage(str(path))


def _rel(path: Path, root: Path) -> str:
    """Return ``path`` relative to ``root``, or its string form if outside."""
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _path_key(relative_path: str) -> str:
    """Hash a relative path into a stable NIfTI resource key."""
    return hashlib.sha1(relative_path.encode("utf-8")).hexdigest()[:12]


def _optional_str(value: Any) -> str | None:
    """Normalize a scalar to a stripped string, dropping empties/NaN."""
    if value is None or _is_nan(value):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def _is_nan(value: Any) -> bool:
    """Return True for float NaN sentinels."""
    try:
        import math

        return isinstance(value, float) and math.isnan(value)
    except Exception:
        return False


def _jsonable(value: Any) -> Any:
    """Coerce crawl fingerprint cells into JSON-friendly values."""
    if _is_nan(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value
