"""Patient-scoped datactrl tools for Gemini (no filesystem paths)."""

from __future__ import annotations

import json
from typing import Any

from google.genai import types

from src.retrieval.case_store import get_case
from src.retrieval.urls import resource_url

from datactrl.errors import DataCtrlError
from datactrl.tools import gemini_function_declarations

RETRIEVAL_TOOL_NAMES = {
    "discover",
    "describe",
    "fetch",
    "links",
    "assert_link",
    "dicom_tree",
    "dicom_groups",
    "dicom_select",
}

_PATIENT_ID_PROP = {
    "patient_id": {
        "type": "string",
        "description": "Patient identifier (e.g. TCGA-DD-A4NF)",
    }
}


def _rewrite_declaration(decl: dict[str, Any]) -> dict[str, Any]:
    """Swap root/case_id for patient_id so the model never sees filesystem paths."""
    params = dict(decl.get("parameters") or {})
    properties = dict(params.get("properties") or {})
    properties.pop("root", None)
    properties.pop("case_id", None)
    properties = {**_PATIENT_ID_PROP, **properties}

    required = [
        name
        for name in (params.get("required") or [])
        if name not in {"root", "case_id"}
    ]
    if "patient_id" not in required:
        required = ["patient_id", *required]

    return {
        "name": decl["name"],
        "description": decl.get("description", ""),
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
        },
    }


def retrieval_declarations() -> list[dict[str, Any]]:
    """Gemini function declarations for the datactrl retrieve layer."""
    return [_rewrite_declaration(item) for item in gemini_function_declarations()]


def _compact_fetch(patient_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Drop image bytes from LLM-facing fetch; point at the HTTP preview instead."""
    data = payload.get("data")
    if not isinstance(data, dict) or "image_base64" not in data:
        return payload
    resource_id = payload.get("id")
    compact = dict(data)
    compact.pop("image_base64", None)
    compact["web_path"] = resource_url(patient_id, str(resource_id), "preview")
    compact["note"] = (
        "Image bytes omitted from the tool result. "
        "Use analyze_imaging or the web_path URL."
    )
    return {**payload, "data": compact}


def dispatch_retrieval(
    name: str,
    patient_id: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """Run a datactrl tool against the patient's cached Case."""
    case = get_case(patient_id)
    args = {key: value for key, value in arguments.items() if key != "patient_id"}

    try:
        if name == "discover":
            return case.report.to_inventory()
        if name == "describe":
            return case.describe(args["resource_id"])
        if name == "fetch":
            resource_id = args.pop("resource_id")
            view = args.pop("view", None)
            return _compact_fetch(patient_id, case.fetch(resource_id, view=view, **args))
        if name == "links":
            return {"links": case.links(args.get("resource_id"))}
        if name == "assert_link":
            return case.assert_link(
                args["from_id"],
                args["to_id"],
                args["relation"],
                meta=args.get("meta"),
            )
        if name == "dicom_tree":
            return case.dicom_tree()
        if name == "dicom_groups":
            return case.dicom_groups(args.get("query"))
        if name == "dicom_select":
            return case.dicom_select(
                args["query"],
                limit=int(args.get("limit") or 5),
                include_localizers=bool(args.get("include_localizers") or False),
            )
    except (DataCtrlError, KeyError, TypeError, ValueError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}

    return {"error": f"Unknown retrieval tool: {name}"}


def retrieval_tool() -> types.Tool:
    """Gemini Tool wrapping the rewritten datactrl declarations."""
    return types.Tool(function_declarations=retrieval_declarations())


def dump_for_model(payload: dict[str, Any]) -> dict[str, Any]:
    """JSON-roundtrip so Gemini gets plain types."""
    return json.loads(json.dumps(payload, default=str))
