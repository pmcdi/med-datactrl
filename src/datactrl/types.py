"""Shared types for the medical data harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class Modality(StrEnum):
    """Supported medical data modalities."""

    DICOM = "dicom"
    STRUCTURED = "structured"
    NOTE = "note"


@dataclass(frozen=True)
class ViewSpec:
    """Declared fetch view for a modality/resource."""

    name: str
    description: str = ""
    defaults: dict[str, Any] = field(default_factory=dict)
    max: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the view spec for API responses."""
        return {
            "name": self.name,
            "description": self.description,
            "defaults": dict(self.defaults),
            "max": dict(self.max),
        }


@dataclass(frozen=True)
class Link:
    """Edge between resources.

    ``source="structural"`` comes from adapters (e.g. DICOM references).
    ``source="asserted"`` is an agent/user overlay on the Case.
    """

    from_id: str
    to_id: str
    relation: str
    source: str = "structural"
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the link for API responses."""
        return {
            "from_id": self.from_id,
            "to_id": self.to_id,
            "relation": self.relation,
            "source": self.source,
            "meta": dict(self.meta),
        }


@dataclass(frozen=True)
class ResourceRef:
    """Stable handle to one discovered resource."""

    id: str
    modality: Modality
    label: str
    path: str = ""
    patient_id: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the resource ref for API responses."""
        return {
            "id": self.id,
            "modality": self.modality.value,
            "label": self.label,
            "path": self.path,
            "patient_id": self.patient_id,
            "meta": dict(self.meta),
        }


@dataclass
class DescribeResult:
    """Adapter output for ``describe`` without full content."""

    id: str
    modality: Modality
    label: str
    views: list[str]
    path: str = ""
    patient_id: str | None = None
    summary: dict[str, Any] = field(default_factory=dict)
    links: list[Link] = field(default_factory=list)
    preview: str | None = None
    view_specs: list[ViewSpec] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the describe result for API responses."""
        return {
            "id": self.id,
            "modality": self.modality.value,
            "label": self.label,
            "path": self.path,
            "patient_id": self.patient_id,
            "views": list(self.views),
            "view_specs": [spec.to_dict() for spec in self.view_specs],
            "summary": dict(self.summary),
            "links": [link.to_dict() for link in self.links],
            "preview": self.preview,
        }


@dataclass
class FetchResult:
    """Adapter output for a bounded ``fetch`` view."""

    id: str
    modality: Modality
    view: str
    data: Any
    bounds: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the fetch result for API responses."""
        return {
            "id": self.id,
            "modality": self.modality.value,
            "view": self.view,
            "data": self.data,
            "bounds": dict(self.bounds),
            "provenance": dict(self.provenance),
        }


@dataclass
class DiscoverReport:
    """Result of scanning a case root."""

    case_id: str
    root: Path
    resources: list[ResourceRef] = field(default_factory=list)
    unknown: list[dict[str, str]] = field(default_factory=list)

    @property
    def dicom(self) -> list[ResourceRef]:
        """DICOM series resources."""
        return [ref for ref in self.resources if ref.modality == Modality.DICOM]

    @property
    def structured(self) -> list[ResourceRef]:
        """Structured table resources."""
        return [ref for ref in self.resources if ref.modality == Modality.STRUCTURED]

    @property
    def notes(self) -> list[ResourceRef]:
        """Clinical note resources."""
        return [ref for ref in self.resources if ref.modality == Modality.NOTE]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the discovery report for API responses."""
        return {
            "case_id": self.case_id,
            "root": str(self.root),
            "counts": {
                "dicom": len(self.dicom),
                "structured": len(self.structured),
                "notes": len(self.notes),
                "unknown": len(self.unknown),
                "total": len(self.resources),
            },
            "resources": [ref.to_dict() for ref in self.resources],
            "dicom": [ref.to_dict() for ref in self.dicom],
            "structured": [ref.to_dict() for ref in self.structured],
            "notes": [ref.to_dict() for ref in self.notes],
            "unknown": list(self.unknown),
        }


def make_resource_id(modality: Modality, key: str) -> str:
    """Build a namespaced resource id such as ``dicom:<uid>`` or ``note:<hash>``.

    Args:
        modality: Resource modality prefix.
        key: Stable key within the modality namespace.

    Returns:
        Namespaced id string.

    Raises:
        ValueError: If ``key`` is empty after stripping.
    """
    key = key.strip()
    if not key:
        raise ValueError("resource id key must be non-empty")
    return f"{modality.value}:{key}"


def parse_resource_id(resource_id: str) -> tuple[Modality | None, str]:
    """Split ``modality:key`` into its parts.

    Args:
        resource_id: Namespaced or legacy bare id.

    Returns:
        ``(modality, key)`` when prefixed, otherwise ``(None, resource_id)``.
    """
    if ":" not in resource_id:
        return None, resource_id
    prefix, key = resource_id.split(":", 1)
    try:
        return Modality(prefix), key
    except ValueError:
        return None, resource_id
