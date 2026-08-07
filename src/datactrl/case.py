"""Case access layer: open a root, discover modalities, describe/fetch resources."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

from datactrl.adapters.base import Adapter
from datactrl.adapters.dicom import DicomAdapter
from datactrl.adapters.notes import NotesAdapter
from datactrl.adapters.structured import StructuredAdapter
from datactrl.discovery.classify import classify_files
from datactrl.discovery.walk import iter_files
from datactrl.errors import ModalityUnavailable, ResourceNotFound
from datactrl.limits import DEFAULT_NOTE_CHARS, DEFAULT_TABLE_LIMIT
from datactrl.types import (
    DescribeResult,
    DiscoverReport,
    FetchResult,
    Link,
    Modality,
    ResourceRef,
)

log = logging.getLogger(__name__)


class Case:
    """Local medical-data case rooted at a filesystem directory.

    Canonical agent API: ``discover`` → ``describe`` → ``fetch(view=...)``.
    """

    def __init__(self, root: Path) -> None:
        """Bind a case to ``root`` without running discovery."""
        self.root = root.resolve()
        self.case_id = _case_id(self.root)

        self._dicom = DicomAdapter()
        self._structured = StructuredAdapter()
        self._notes = NotesAdapter()
        self._adapters: dict[Modality, Adapter] = {
            Modality.DICOM: self._dicom,
            Modality.STRUCTURED: self._structured,
            Modality.NOTE: self._notes,
        }

        self._index: dict[str, ResourceRef] = {}
        self._report: DiscoverReport | None = None
        self._asserted_links: list[Link] = []

    @classmethod
    def open(cls, root: str | Path) -> Case:
        """Open a directory and run an initial ``discover`` pass.

        Args:
            root: Filesystem path to the case directory.

        Returns:
            A discovered case ready for ``describe`` / ``fetch``.

        Raises:
            ValueError: If ``root`` is not a directory.
        """
        path = Path(root)
        if not path.is_dir():
            raise ValueError(f"Root is not a directory: {path}")

        case = cls(path)
        case.discover()
        return case

    def discover(self) -> DiscoverReport:
        """Scan the case root and index discoverable resources.

        Rebuilds the resource index and cached report. Safe to call again
        after filesystem changes.

        Returns:
            Discovery report with resources, modality counts, and unknown files.
        """
        log.debug("discovering case at %s", self.root)

        files = iter_files(self.root)
        log.info("found %d files under case root", len(files))

        buckets = classify_files(files)
        dicom_refs = list(self._dicom.ingest(self.root, buckets[Modality.DICOM]))
        structured_refs = list(
            self._structured.ingest(self.root, buckets[Modality.STRUCTURED])
        )
        note_refs = list(self._notes.ingest(self.root, buckets[Modality.NOTE]))

        resources = sorted(
            [*dicom_refs, *structured_refs, *note_refs],
            key=lambda ref: (ref.modality.value, ref.id),
        )
        unknown = [
            {"path": _rel(path, self.root), "reason": "unclassified"}
            for path in buckets[None]
        ]

        self._index = {ref.id: ref for ref in resources}
        self._report = DiscoverReport(
            case_id=self.case_id,
            root=self.root,
            resources=resources,
            unknown=unknown,
        )

        log.info(
            "discovered %d resources (dicom=%d structured=%d notes=%d unknown=%d)",
            len(resources),
            len(dicom_refs),
            len(structured_refs),
            len(note_refs),
            len(unknown),
        )
        return self._report

    @property
    def report(self) -> DiscoverReport:
        """Return the cached discovery report, running ``discover`` if needed."""
        if self._report is None:
            return self.discover()
        return self._report

    def describe(self, resource_id: str) -> dict[str, Any]:
        """Describe one resource, merging structural and asserted links.

        Args:
            resource_id: Stable id from ``discover`` (e.g. ``dicom:<uid>``).

        Returns:
            JSON-serializable summary with views, schema hints, and links.
        """
        ref = self._require(resource_id)
        adapter = self._adapters[ref.modality]
        result: DescribeResult = adapter.describe(resource_id)

        merged_links = list(result.links) + [
            link
            for link in self._asserted_links
            if link.from_id == resource_id or link.to_id == resource_id
        ]

        payload = result.to_dict()
        payload["links"] = [link.to_dict() for link in merged_links]
        return payload

    def fetch(
        self,
        resource_id: str,
        view: str | None = None,
        **opts: Any,
    ) -> dict[str, Any]:
        """Fetch a bounded view of a resource.

        Args:
            resource_id: Stable id from ``discover``.
            view: Adapter-specific view name; see ``describe`` for options.
            **opts: View options such as ``limit``, ``max_chars``, ``slice_index``.

        Returns:
            JSON-serializable payload with data, bounds, and provenance.
        """
        ref = self._require(resource_id)
        adapter = self._adapters[ref.modality]
        result: FetchResult = adapter.fetch(resource_id, view=view, **opts)
        return result.to_dict()

    def links(self, resource_id: str | None = None) -> list[dict[str, Any]]:
        """Return structural and asserted links for the case or one resource.

        Args:
            resource_id: When set, return links touching this resource only.
                When omitted, return deduplicated case-wide structural links.

        Returns:
            List of link dicts with ``from_id``, ``to_id``, ``relation``, ``source``.
        """
        structural: list[Link] = []

        if resource_id is None:
            for ref in self.report.dicom:
                described = self._dicom.describe(ref.id)
                structural.extend(described.links)
        else:
            ref = self._require(resource_id)

            if ref.modality == Modality.DICOM:
                structural.extend(self._dicom.describe(resource_id).links)

            structural.extend(
                link
                for link in self._asserted_links
                if link.from_id == resource_id or link.to_id == resource_id
            )
            return [link.to_dict() for link in structural]

        all_links = structural + list(self._asserted_links)

        # Deduplicate while preserving order
        seen: set[tuple[str, str, str, str]] = set()
        out: list[dict[str, Any]] = []
        for link in all_links:
            key = (link.from_id, link.to_id, link.relation, link.source)
            if key in seen:
                continue
            seen.add(key)
            out.append(link.to_dict())

        return out

    def assert_link(
        self,
        from_id: str,
        to_id: str,
        relation: str,
        *,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Add a session-only asserted link between two known resources.

        Args:
            from_id: Source resource id.
            to_id: Target resource id.
            relation: Free-form relation label (e.g. ``supports_diagnosis``).
            meta: Optional metadata stored on the link.

        Returns:
            The new link as a dict.
        """
        self._require(from_id)
        self._require(to_id)

        link = Link(
            from_id=from_id,
            to_id=to_id,
            relation=relation,
            source="asserted",
            meta=meta or {},
        )
        self._asserted_links.append(link)

        return link.to_dict()

    # --- Thin sugar / graph helpers (still go through adapters) ---

    def fetch_table(
        self,
        table_id: str,
        *,
        columns: list[str] | None = None,
        limit: int = DEFAULT_TABLE_LIMIT,
        offset: int = 0,
    ) -> dict[str, Any]:
        """Fetch bounded rows from a structured table resource."""
        self._ensure_modality(Modality.STRUCTURED)
        return self.fetch(
            table_id,
            view="rows",
            columns=columns,
            limit=limit,
            offset=offset,
        )

    def fetch_note(
        self,
        note_id: str,
        *,
        max_chars: int = DEFAULT_NOTE_CHARS,
        preview: bool = False,
    ) -> dict[str, Any]:
        """Fetch bounded text from a clinical note resource."""
        from datactrl.limits import DEFAULT_NOTE_PREVIEW_CHARS

        self._ensure_modality(Modality.NOTE)

        chars = DEFAULT_NOTE_PREVIEW_CHARS if preview else max_chars
        return self.fetch(note_id, view="text", max_chars=chars)

    def fetch_dicom_metadata(self, series_id: str) -> dict[str, Any]:
        """Fetch DICOM series metadata and structural references."""
        self._ensure_modality(Modality.DICOM)
        return self.fetch(series_id, view="metadata")

    def fetch_dicom_preview(
        self,
        series_id: str,
        *,
        slice_index: int | None = None,
    ) -> dict[str, Any]:
        """Fetch a single-slice PNG preview for a DICOM series."""
        self._ensure_modality(Modality.DICOM)

        opts: dict[str, Any] = {}
        if slice_index is not None:
            opts["slice_index"] = slice_index

        return self.fetch(series_id, view="preview", **opts)

    def dicom_tree(self) -> dict[str, Any]:
        """Return the DICOM patient/series reference forest."""
        self._ensure_modality(Modality.DICOM)
        return self._dicom.tree()

    def dicom_related(self, series_id: str) -> dict[str, Any]:
        """Return one-hop parent and child series for a DICOM resource."""
        self._ensure_modality(Modality.DICOM)
        self._require(series_id)
        return self._dicom.related(series_id)

    def dicom_groups(self, query: str | None = None) -> dict[str, Any]:
        """Return linked DICOM modality groups (e.g. ``CT,SEG``)."""
        self._ensure_modality(Modality.DICOM)
        return self._dicom.groups(query)

    def _require(self, resource_id: str) -> ResourceRef:
        """Look up ``resource_id`` in the index or raise ResourceNotFound."""
        _ = self.report  # ensure index exists

        try:
            return self._index[resource_id]
        except KeyError as exc:
            raise ResourceNotFound(f"Unknown resource: {resource_id!r}") from exc

    def _ensure_modality(self, modality: Modality) -> None:
        """Raise ModalityUnavailable when the case has no resources of ``modality``."""
        resources = [ref for ref in self.report.resources if ref.modality == modality]
        if not resources:
            raise ModalityUnavailable(
                f"No {modality.value} data available for this case"
            )


def _case_id(root: Path) -> str:
    """Derive a stable case id from the resolved root path."""
    digest = hashlib.sha1(str(root).encode("utf-8")).hexdigest()[:16]
    return f"case_{digest}"


def _rel(path: Path, root: Path) -> str:
    """Return ``path`` relative to ``root``, or its string form if outside."""
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)
