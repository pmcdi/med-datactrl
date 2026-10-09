"""Tool schemas and dispatch over Case (thin mirror of the Access API)."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from datactrl.case import Case

log = logging.getLogger(__name__)

_SESSIONS: dict[str, Case] = {}


def _tool(
    name: str,
    description: str,
    properties: dict[str, Any],
    required: list[str],
) -> dict[str, Any]:
    """Build one OpenAI-style function tool schema."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


_ROOT_OR_CASE = {
    "root": {
        "type": "string",
        "description": "Filesystem path to the case root",
    },
    "case_id": {
        "type": "string",
        "description": "Case id returned by discover (alternative to root)",
    },
}


TOOL_DEFINITIONS: list[dict[str, Any]] = [
    _tool(
        "discover",
        "Open or refresh a local case root and list discovered resources.",
        {"root": {"type": "string", "description": "Filesystem path to the case root"}},
        ["root"],
    ),
    _tool(
        "describe",
        "Describe one discovered resource (schema/views/links) without full content.",
        {
            **_ROOT_OR_CASE,
            "resource_id": {"type": "string"},
        },
        ["resource_id"],
    ),
    _tool(
        "fetch",
        "Fetch a bounded view of a resource (e.g. text, rows, metadata, preview).",
        {
            **_ROOT_OR_CASE,
            "resource_id": {"type": "string"},
            "view": {
                "type": "string",
                "description": "Fetch view; see describe().views",
            },
            "max_chars": {"type": "integer"},
            "limit": {"type": "integer"},
            "offset": {"type": "integer"},
            "columns": {"type": "array", "items": {"type": "string"}},
            "slice_index": {
                "type": "integer",
                "description": (
                    "DICOM preview slice index (default: adapter-suggested from "
                    "geometry; mm-from-end on long-FOV series, otherwise middle)"
                ),
            },
            "window": {
                "type": "string",
                "description": (
                    "CT display window: minmax, abdomen, liver, lung, bone, brain"
                ),
            },
            "overlay": {
                "type": "boolean",
                "description": (
                    "For CT/MR preview, draw linked RTSTRUCT contours on the slice. "
                    "RTSTRUCT preview always overlays on the referenced image series."
                ),
            },
            "start": {
                "type": "integer",
                "description": "Note span view: start character offset",
            },
            "end": {
                "type": "integer",
                "description": "Note span view: end character offset (exclusive)",
            },
            "context_chars": {
                "type": "integer",
                "description": "Note span view: characters of context on each side",
            },
            "page_index": {
                "type": "integer",
                "description": "PDF note page view: zero-based page index",
            },
        },
        ["resource_id"],
    ),
    _tool(
        "notes_search",
        (
            "Lexical (BM25) search over clinical note sections. Returns note ids, "
            "section names, character offsets and snippets; use fetch(view='span') "
            "to read the exact text."
        ),
        {
            **_ROOT_OR_CASE,
            "query": {"type": "string", "description": "Keywords or a question"},
            "limit": {"type": "integer", "description": "Max hits (default 8)"},
        },
        ["query"],
    ),
    _tool(
        "links",
        "List structural and asserted links for a resource (or entire case).",
        {
            **_ROOT_OR_CASE,
            "resource_id": {
                "type": "string",
                "description": (
                    "Optional resource id; omit for whole-case structural links"
                ),
            },
        },
        [],
    ),
    _tool(
        "assert_link",
        "Add a session-only asserted link between two known resources.",
        {
            **_ROOT_OR_CASE,
            "from_id": {"type": "string"},
            "to_id": {"type": "string"},
            "relation": {"type": "string"},
        },
        ["from_id", "to_id", "relation"],
    ),
    _tool(
        "dicom_tree",
        "Return the DICOM series reference forest (roots and children).",
        {**_ROOT_OR_CASE},
        [],
    ),
    _tool(
        "dicom_groups",
        (
            "Return linked modality groups (e.g. query='CT,SEG'). "
            "Omit query for all valid pairs."
        ),
        {
            **_ROOT_OR_CASE,
            "query": {
                "type": "string",
                "description": "Imagetools modality query such as CT,SEG",
            },
        },
        [],
    ),
    _tool(
        "dicom_select",
        (
            "Rank DICOM series for a clinical query using series description, "
            "protocol, image type, geometry, and RTSTRUCT ROI names. "
            "Does not return pixels. Contour/GTV questions select the RTSTRUCT; "
            "representative CT/MR slice questions select an acquisition. "
            "Use the selected resource_id with fetch(preview) or analyze_imaging."
        ),
        {
            **_ROOT_OR_CASE,
            "query": {
                "type": "string",
                "description": "Clinical question or anatomy/modality to find",
            },
            "limit": {
                "type": "integer",
                "description": "Max ranked candidates (default 5)",
            },
            "include_localizers": {
                "type": "boolean",
                "description": "Include scout/localizer series (default false)",
            },
        },
        ["query"],
    ),
]


def gemini_function_declarations() -> list[dict[str, Any]]:
    """Return Gemini-style function declarations (name/description/parameters)."""
    return [dict(tool["function"]) for tool in TOOL_DEFINITIONS]


def _session_key(root: str | Path) -> str:
    """Normalize a root path to a session cache key."""
    return str(Path(root).resolve())


def _case_from_root(root: str) -> Case:
    """Return a cached Case for ``root``, opening it on first use."""
    key = _session_key(root)
    if key not in _SESSIONS:
        _SESSIONS[key] = Case.open(key)
    return _SESSIONS[key]


def _case_from_args(
    *,
    root: str | None = None,
    case_id: str | None = None,
) -> Case:
    """Resolve a Case from ``root`` or a prior ``case_id``."""
    if root:
        return _case_from_root(root)

    if case_id:
        for case in _SESSIONS.values():
            if case.case_id == case_id:
                return case

        raise ValueError(f"Unknown case_id {case_id!r}; call discover(root=...) first")

    raise ValueError("Provide root or case_id")


def discover(root: str) -> dict[str, Any]:
    """Open a case root and return the discovery report as a dict."""
    key = _session_key(root)
    case = Case.open(key)
    _SESSIONS[key] = case
    return case.report.to_inventory()


def describe(
    resource_id: str,
    root: str | None = None,
    case_id: str | None = None,
) -> dict[str, Any]:
    """Describe one resource in an open case session."""
    return _case_from_args(root=root, case_id=case_id).describe(resource_id)


def fetch(
    resource_id: str,
    root: str | None = None,
    case_id: str | None = None,
    view: str | None = None,
    **opts: Any,
) -> dict[str, Any]:
    """Fetch a bounded resource view in an open case session."""
    clean = {key: value for key, value in opts.items() if value is not None}
    return _case_from_args(root=root, case_id=case_id).fetch(
        resource_id, view=view, **clean
    )


def links(
    root: str | None = None,
    case_id: str | None = None,
    resource_id: str | None = None,
) -> dict[str, Any]:
    """List links for a case or one resource in an open session."""
    case = _case_from_args(root=root, case_id=case_id)
    return {"links": case.links(resource_id)}


def assert_link(
    from_id: str,
    to_id: str,
    relation: str,
    root: str | None = None,
    case_id: str | None = None,
) -> dict[str, Any]:
    """Add a session-only asserted link in an open case session."""
    return _case_from_args(root=root, case_id=case_id).assert_link(
        from_id, to_id, relation
    )


def dicom_tree(
    root: str | None = None,
    case_id: str | None = None,
) -> dict[str, Any]:
    """Return the DICOM reference forest for an open case session."""
    return _case_from_args(root=root, case_id=case_id).dicom_tree()


def dicom_groups(
    root: str | None = None,
    case_id: str | None = None,
    query: str | None = None,
) -> dict[str, Any]:
    """Return linked DICOM modality groups for an open case session."""
    return _case_from_args(root=root, case_id=case_id).dicom_groups(query)


def dicom_select(
    query: str,
    root: str | None = None,
    case_id: str | None = None,
    limit: int = 5,
    include_localizers: bool = False,
) -> dict[str, Any]:
    """Rank DICOM series for a query in an open case session."""
    return _case_from_args(root=root, case_id=case_id).dicom_select(
        query, limit=limit, include_localizers=include_localizers
    )


def notes_search(
    query: str,
    root: str | None = None,
    case_id: str | None = None,
    limit: int = 8,
) -> dict[str, Any]:
    """Search note sections in an open case session."""
    return _case_from_args(root=root, case_id=case_id).notes_search(
        query, limit=int(limit)
    )


_HANDLERS: dict[str, Callable[..., Any]] = {
    "discover": discover,
    "describe": describe,
    "fetch": fetch,
    "links": links,
    "assert_link": assert_link,
    "dicom_tree": dicom_tree,
    "dicom_groups": dicom_groups,
    "dicom_select": dicom_select,
    "notes_search": notes_search,
}


def dispatch(name: str, arguments: str | dict[str, Any]) -> str:
    """Invoke a tool by name and return JSON text.

    Args:
        name: Tool name from ``TOOL_DEFINITIONS``.
        arguments: JSON string or dict of tool arguments.

    Returns:
        JSON-encoded tool result.

    Raises:
        ValueError: If ``name`` is unknown.
    """
    args = json.loads(arguments) if isinstance(arguments, str) else arguments
    log.debug("dispatch %s(%s)", name, args)

    handler = _HANDLERS.get(name)
    if handler is None:
        raise ValueError(f"Unknown tool: {name!r}")

    result = handler(**args)
    return json.dumps(result, default=str)


def reset_sessions() -> None:
    """Clear cached Case sessions (mainly for tests)."""
    _SESSIONS.clear()
