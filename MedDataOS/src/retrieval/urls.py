"""Stable HTTP paths for datactrl resources (citation cards)."""

from __future__ import annotations

from urllib.parse import quote


def resource_url(patient_id: str, resource_id: str, view: str) -> str:
    """Return a frontend-fetchable URL for one resource view."""
    return (
        f"/api/cases/{quote(patient_id, safe='')}/resource"
        f"?id={quote(resource_id, safe='')}&view={quote(view, safe='')}"
    )
