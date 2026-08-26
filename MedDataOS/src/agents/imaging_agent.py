"""DICOM imaging agent: bounded preview via datactrl, then a vision LLM."""

from __future__ import annotations

import base64

from src.agents.base_agent import BaseAgent, Finding
from src.core.gemini_client import GEMINI_MODEL, create_client
from src.retrieval.case_store import get_case
from src.retrieval.urls import resource_url

from datactrl.dicom_index import choose_preview_window, parse_query_intent, rank_series
from datactrl.errors import DataCtrlError

_ANALYSIS_PROMPT = """You are a radiology specialist looking at a single DICOM slice preview (not a full study read).

Question: {query}

Series: {label} ({modality})
Resource id: {resource_id}
Kind: {series_kind}
Slice {slice_index} of {n_slices}
Window: {window}
{overlay_block}
{warnings}

Describe only anatomy that is visible on this slice. If contours are overlaid, name structures from the ROI list and the drawn outlines — do not invent extra ROIs. If the series is a runoff/extremity protocol, do not assume a dedicated liver/abdomen acquisition. Do not invent organs that are not in the frame. Be explicit that this is one preview slice, not a complete interpretation."""


class ImagingAgent(BaseAgent):
    modality = "imaging"
    data_dir = "radiology"
    file_ext = ".png"

    def analyze(
        self,
        patient_id: str,
        query: str,
        resource_id: str | None = None,
        slice_index: int | None = None,
    ) -> Finding:
        case = get_case(patient_id)
        refs = list(case.report.dicom)
        if not refs:
            return Finding(
                agent=self.modality,
                file="imaging.txt",
                web_path="",
                summary="No DICOM series in this case.",
            )

        ranked = case.dicom_select(query, limit=5)
        chosen = _resolve_series(refs, ranked, resource_id)
        if chosen is None:
            return Finding(
                agent=self.modality,
                file="dicom_select.txt",
                web_path="",
                summary=_format_candidates(ranked),
            )

        series_id = chosen["id"]
        ref = next(item for item in refs if item.id == series_id)
        if slice_index is None:
            slice_index = chosen.get("suggested_slice_index")
        label = ref.label or series_id
        modality = (ref.meta.get("Modality") if ref else None) or "DICOM"
        series_kind = str(ref.meta.get("series_kind") or "acquisition")
        warnings = list(chosen.get("warnings") or [])
        intent = parse_query_intent(query)
        pixel_modality = "CT" if series_kind == "rtstruct" else str(modality)
        window = choose_preview_window(pixel_modality, intent)

        fetch_opts: dict = {"window": window}
        if slice_index is not None:
            fetch_opts["slice_index"] = int(slice_index)
        if series_kind == "rtstruct" or intent.contours:
            fetch_opts["overlay"] = True
        try:
            preview = case.fetch(series_id, view="preview", **fetch_opts)
        except (DataCtrlError, ValueError) as exc:
            return Finding(
                agent=self.modality,
                file=f"{label}.png",
                web_path=resource_url(patient_id, series_id, "preview"),
                summary=f"Could not render a preview for series {label}: {exc}",
            )
        data = preview.get("data") or {}
        encoded = data.get("image_base64")
        if not encoded:
            return Finding(
                agent=self.modality,
                file=f"{label}.png",
                web_path=resource_url(patient_id, series_id, "preview"),
                summary=f"Could not render a preview for series {label}.",
            )

        meta = data.get("meta") or {}
        from google.genai import types

        png = base64.b64decode(encoded)
        warning_text = ""
        if warnings:
            warning_text = "Warnings: " + "; ".join(warnings)
        overlay_block = _overlay_prompt_block(ref.meta, meta)
        prompt = _ANALYSIS_PROMPT.format(
            query=query,
            label=label,
            modality=modality,
            resource_id=series_id,
            series_kind=series_kind,
            slice_index=meta.get("slice_index", "?"),
            n_slices=meta.get("n_slices", "?"),
            window=meta.get("window", window),
            overlay_block=overlay_block,
            warnings=warning_text,
        )
        client = create_client()
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_bytes(data=png, mime_type="image/png"),
                        types.Part.from_text(text=prompt),
                    ],
                )
            ],
        )
        summary = (response.text or "").strip() or "Preview retrieved but no interpretation was produced."
        return Finding(
            agent=self.modality,
            file=f"{label}.png",
            web_path=resource_url(patient_id, series_id, "preview"),
            summary=summary,
        )


def _overlay_prompt_block(ref_meta: dict, fetch_meta: dict) -> str:
    """ROI names from the structure set plus what was drawn on this slice."""
    names = list(ref_meta.get("roi_names") or fetch_meta.get("roi_names") or [])
    drawn = list(fetch_meta.get("overlay_rois") or [])
    if not names and not drawn:
        return ""
    lines = []
    if names:
        lines.append("Structure-set ROI names: " + ", ".join(str(n) for n in names))
    if drawn:
        parts = [
            f"{item.get('name')} ({item.get('color')})"
            if item.get("color")
            else str(item.get("name"))
            for item in drawn
        ]
        lines.append("Contours drawn on this slice: " + ", ".join(parts))
    elif fetch_meta.get("overlay") is False:
        lines.append("No contours fell on this slice.")
    return "\n".join(lines)


def _resolve_series(
    refs: list,
    ranked: dict,
    resource_id: str | None,
) -> dict | None:
    """Honor an explicit resource_id; otherwise require a confident ranker pick."""
    if resource_id:
        ref = next((item for item in refs if item.id == resource_id), None)
        if ref is None:
            return None
        one = rank_series([ref], ranked.get("query") or "", limit=1, include_localizers=True)
        if one["candidates"]:
            return one["candidates"][0]
        return {
            "id": ref.id,
            "label": ref.label,
            "suggested_slice_index": ref.meta.get("suggested_slice_index"),
            "warnings": [],
        }
    if ranked.get("confident") and ranked.get("selected"):
        return ranked["selected"]
    return None


def _format_candidates(ranked: dict) -> str:
    """Fail closed: return ranked options instead of guessing a series."""
    intent = ranked.get("intent") or {}
    lines = [
        "No confident DICOM series match. Ranked candidates (call dicom_select, "
        "then analyze_imaging with resource_id):",
        f"intent={intent}",
    ]
    for item in ranked.get("candidates") or []:
        reasons = ", ".join(item.get("reasons") or [])
        lines.append(
            f"- {item.get('id')}: {item.get('label')} "
            f"({item.get('Modality')}, {item.get('series_kind')}, "
            f"n={item.get('instance_count')}, slice={item.get('suggested_slice_index')}) "
            f"score={item.get('score')} {reasons}"
        )
    if len(lines) == 2:
        lines.append("- (none)")
    return "\n".join(lines)
