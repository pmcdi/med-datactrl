"""Structured table/record adapter."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from datactrl.adapters.base import resolve_fetch_options
from datactrl.errors import ResourceNotFound
from datactrl.limits import DEFAULT_TABLE_LIMIT, MAX_TABLE_LIMIT
from datactrl.types import (
    DescribeResult,
    FetchResult,
    Modality,
    ResourceRef,
    ViewSpec,
    make_resource_id,
)


@dataclass
class _TableRecord:
    ref: ResourceRef
    path: Path
    format: str


_VIEWS = (
    ViewSpec(
        name="rows",
        description="Tabular rows with limit/offset/column projection",
        defaults={"limit": DEFAULT_TABLE_LIMIT, "offset": 0},
        max={"limit": MAX_TABLE_LIMIT},
    ),
)


class StructuredAdapter:
    """Structured table/record modality adapter."""

    modality = Modality.STRUCTURED
    version = "1"

    def __init__(self) -> None:
        """Create an empty adapter; call ``ingest`` before describe/fetch."""
        self._tables: dict[str, _TableRecord] = {}

    def views(self) -> Sequence[ViewSpec]:
        """Return supported structured fetch views."""
        return _VIEWS

    def ingest(self, root: Path, paths: Sequence[Path]) -> list[ResourceRef]:
        """Index structured files under ``root`` as table resources."""
        self._tables.clear()
        refs: list[ResourceRef] = []
        for path in sorted(paths):
            rel = _rel(path, root)
            table_id = make_resource_id(Modality.STRUCTURED, _path_key(rel))
            fmt = path.suffix.lower().lstrip(".") or "csv"
            columns = _sample_columns(path, fmt)
            ref = ResourceRef(
                id=table_id,
                modality=Modality.STRUCTURED,
                path=rel,
                label=path.name,
                meta={"format": fmt, "columns": columns},
            )
            self._tables[table_id] = _TableRecord(ref=ref, path=path, format=fmt)
            refs.append(ref)
        return refs

    def describe(self, resource_id: str) -> DescribeResult:
        """Return column names and row count for one table."""
        record = self._require(resource_id)
        rows = _load_rows(record.path, record.format)
        columns = (
            list(rows[0].keys()) if rows else list(record.ref.meta.get("columns", []))
        )
        return DescribeResult(
            id=resource_id,
            modality=Modality.STRUCTURED,
            label=record.ref.label,
            path=record.ref.path,
            views=[spec.name for spec in _VIEWS],
            view_specs=list(_VIEWS),
            summary={
                "format": record.format,
                "columns": columns,
                "row_count": len(rows),
            },
        )

    def fetch(
        self, resource_id: str, view: str | None = None, **opts: Any
    ) -> FetchResult:
        """Fetch bounded rows with optional column projection."""
        resolved_view, merged = resolve_fetch_options(
            _VIEWS, view, opts, default_view="rows"
        )
        record = self._require(resource_id)
        limit = int(merged.get("limit", DEFAULT_TABLE_LIMIT))
        offset = int(merged.get("offset", 0))
        columns = merged.get("columns")

        rows = _load_rows(record.path, record.format)
        total = len(rows)
        sliced = rows[offset : offset + limit]

        if columns:
            sliced = [{key: row.get(key) for key in columns} for row in sliced]

        out_columns = columns or (
            list(rows[0].keys()) if rows else list(record.ref.meta.get("columns", []))
        )

        return FetchResult(
            id=resource_id,
            modality=Modality.STRUCTURED,
            view=resolved_view,
            data={
                "columns": out_columns,
                "rows": sliced,
                "row_count": total,
            },
            bounds={
                "limit": limit,
                "offset": offset,
                "truncated": offset + limit < total,
            },
            provenance={
                "path": record.ref.path,
                "adapter": "structured",
                "adapter_version": self.version,
            },
        )

    def _require(self, resource_id: str) -> _TableRecord:
        """Look up an indexed table or raise ResourceNotFound."""
        try:
            return self._tables[resource_id]
        except KeyError as exc:
            raise ResourceNotFound(f"Unknown table: {resource_id!r}") from exc


def _sample_columns(path: Path, fmt: str) -> list[str]:
    """Read column names from the first row of a table file."""
    rows = _load_rows(path, fmt, max_rows=1)
    return list(rows[0].keys()) if rows else []


def _load_rows(
    path: Path, fmt: str, max_rows: int | None = None
) -> list[dict[str, Any]]:
    """Load rows from a supported structured file format."""
    if fmt == "csv":
        return _load_csv(path, delimiter=",", max_rows=max_rows)
    if fmt == "tsv":
        return _load_csv(path, delimiter="\t", max_rows=max_rows)
    if fmt == "json":
        return _load_json(path, max_rows=max_rows)
    if fmt == "jsonl":
        return _load_jsonl(path, max_rows=max_rows)
    if fmt == "parquet":
        return _load_parquet(path, max_rows=max_rows)
    raise ValueError(f"Unsupported structured format: {fmt!r}")


def _load_csv(
    path: Path,
    *,
    delimiter: str,
    max_rows: int | None,
) -> list[dict[str, Any]]:
    """Load delimiter-separated rows from a CSV/TSV file."""
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        rows: list[dict[str, Any]] = []
        for row in reader:
            rows.append(dict(row))
            if max_rows is not None and len(rows) >= max_rows:
                break
        return rows


def _load_json(path: Path, max_rows: int | None) -> list[dict[str, Any]]:
    """Load tabular rows from a JSON array or ``{data: [...]}`` wrapper."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        rows = payload["data"]
    elif isinstance(payload, list):
        rows = payload
    else:
        raise ValueError(f"JSON is not tabular: {path}")
    rows = [row for row in rows if isinstance(row, dict)]
    if max_rows is not None:
        rows = rows[:max_rows]
    return rows


def _load_jsonl(path: Path, max_rows: int | None) -> list[dict[str, Any]]:
    """Load object rows from a JSONL file."""
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
            if max_rows is not None and len(rows) >= max_rows:
                break
    return rows


def _load_parquet(path: Path, max_rows: int | None) -> list[dict[str, Any]]:
    """Load rows from a Parquet file (requires optional pyarrow extra)."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ValueError("Parquet support requires pyarrow") from exc
    table = pq.read_table(path)
    if max_rows is not None:
        table = table.slice(0, max_rows)
    return cast(list[dict[str, Any]], table.to_pylist())


def _rel(path: Path, root: Path) -> str:
    """Return ``path`` relative to ``root``, or its string form if outside."""
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _path_key(relative_path: str) -> str:
    """Hash a relative path into a stable structured resource key."""
    return hashlib.sha1(relative_path.encode("utf-8")).hexdigest()[:12]
