"""Adapter protocol and fetch-option validation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from datactrl.errors import InvalidFetchOptions, LimitExceeded, UnknownView
from datactrl.types import DescribeResult, FetchResult, Modality, ResourceRef, ViewSpec


class Adapter(Protocol):
    """Modality plugin: ingest files → describe/fetch bounded views."""

    modality: Modality
    version: str

    def views(self) -> Sequence[ViewSpec]:
        """Declare supported fetch views and their bounds."""

    def ingest(self, root: Path, paths: Sequence[Path]) -> Sequence[ResourceRef]:
        """Index resources under ``root``. Must be deterministic."""

    def describe(self, resource_id: str) -> DescribeResult:
        """Return schema/summary/links without full content."""

    def fetch(
        self, resource_id: str, view: str | None = None, **opts: Any
    ) -> FetchResult:
        """Return a bounded payload for ``view``."""


def view_map(specs: Sequence[ViewSpec]) -> dict[str, ViewSpec]:
    """Index view specs by name."""
    return {spec.name: spec for spec in specs}


def resolve_fetch_options(
    specs: Sequence[ViewSpec],
    view: str | None,
    opts: Mapping[str, Any],
    *,
    default_view: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Validate ``view``/opts against ViewSpecs.

    - Missing view → ``default_view`` or the sole declared view.
    - Unknown view → ``UnknownView`` with allowed list.
    - Numeric opts above ViewSpec.max → ``LimitExceeded`` (reject, do not clamp).
    - Declared defaults fill missing keys.
    """
    by_name = view_map(specs)
    if not by_name:
        raise UnknownView("Adapter declares no fetch views")

    resolved_view = view or default_view
    if resolved_view is None and len(by_name) == 1:
        resolved_view = next(iter(by_name))
    if resolved_view is None or resolved_view not in by_name:
        allowed = ", ".join(sorted(by_name))
        raise UnknownView(
            f"Unknown view {view!r}; allowed: {allowed}"
            if view
            else f"view is required; allowed: {allowed}"
        )

    spec = by_name[resolved_view]
    merged: dict[str, Any] = {**spec.defaults, **dict(opts)}

    for key, ceiling in spec.max.items():
        if key not in merged or merged[key] is None:
            continue
        value = merged[key]
        if isinstance(ceiling, (int, float)) and isinstance(value, (int, float)):
            if value < 0:
                raise InvalidFetchOptions(f"{key} must be non-negative, got {value!r}")
            if value > ceiling:
                raise LimitExceeded(
                    f"{key}={value} exceeds max {ceiling} for view {resolved_view!r}; "
                    f"retry with {key}<={ceiling}"
                )

    for key, value in list(merged.items()):
        if key in ("limit", "offset", "max_chars", "slice_index") and isinstance(
            value, (int, float)
        ):
            if value < 0:
                raise InvalidFetchOptions(f"{key} must be non-negative, got {value!r}")

    return resolved_view, merged
