"""
Main orchestrator for the multimodal medical agent system.
Coordinates between specialized agents using Gemini's function calling.
"""
import threading
from google.genai import types

from src.tools.definitions import tools_for_patient
from src.utils.logging import log_user, log_assistant, log_tool_call, log_tool_result, broadcast_knowledge_bus_update
from src.agents.prompts import build_system_prompt
from src.agents.knowledge_bus import build_knowledge_bus
from src.core.gemini_client import GEMINI_MODEL, create_client, create_config
from src.core.response_assembly import assemble_response
from src.agents.base_agent import Finding
from src.patients.catalog import DATACTRL_BACKEND, has_case
from src.retrieval.tools import RETRIEVAL_TOOL_NAMES, dispatch_retrieval, dump_for_model

from src.agents.clinical_notes_agent import ClinicalNotesAgent
from src.agents.chest_xray_agent import ChestXrayAgent
from src.agents.ecg_agent import EcgAgent
from src.agents.echo_agent import EchoAgent
from src.agents.heart_sounds_agent import HeartSoundsAgent
from src.agents.lab_results_agent import LabResultsAgent
from src.agents.medication_agent import MedicationAgent
from src.agents.imaging_agent import ImagingAgent

AGENT_REGISTRY = {
    "analyze_clinical_notes": ClinicalNotesAgent(),
    "analyze_chest_xray": ChestXrayAgent(),
    "analyze_ecg": EcgAgent(),
    "analyze_echo": EchoAgent(),
    "analyze_heart_sounds": HeartSoundsAgent(),
    "analyze_lab_results": LabResultsAgent(),
    "analyze_medication": MedicationAgent(),
}

DATACTRL_AGENTS = {
    "analyze_clinical_notes": AGENT_REGISTRY["analyze_clinical_notes"],
    "analyze_imaging": ImagingAgent(),
}


def _finding_payload(finding: Finding) -> dict:
    """Function response for the orchestrator model: summary plus compact provenance."""
    payload: dict = {"summary": finding.summary}
    if finding.evidence:
        payload["evidence"] = [
            {"n": e.get("n"), "note": e.get("label"), "date": e.get("note_date"), "quote": e.get("quote")}
            for e in finding.evidence
        ]
    if finding.conflicts:
        payload["conflicts"] = [
            {
                "attribute": c.get("attribute"),
                "kind": c.get("kind"),
                "severity": c.get("severity"),
                "values": [f"{v.get('value')} ({v.get('note_label')})" for v in c.get("values", [])],
                "rationale": c.get("rationale"),
            }
            for c in finding.conflicts
        ]
    return payload


def has_function_calls(response) -> bool:
    try:
        parts = response.candidates[-1].content.parts
        return any(p.function_call and p.function_call.name for p in parts)
    except (IndexError, AttributeError):
        return False


def run_pipeline(
    user_query: str,
    conversation_history: list = None,
    patient_id: str = None,
    patient_info: dict = None,
    on_knowledge_bus_complete: callable = None,
):
    client = create_client()
    backend = DATACTRL_BACKEND if has_case(patient_id) else "stub"
    system_prompt = build_system_prompt(patient_id, patient_info, backend=backend)
    config = create_config(tools_for_patient(patient_id), system_prompt)

    # Build conversation contents
    contents = []
    if conversation_history:
        for msg in conversation_history[:-1]:
            role = "user" if msg["role"] == "user" else "model"
            contents.append(types.Content(
                role=role,
                parts=[types.Part(text=msg["content"])],
            ))

    contents.append(types.Content(
        role="user", parts=[types.Part(text=user_query)],
    ))
    log_user(user_query)

    # Initial Gemini call
    response = client.models.generate_content(
        model=GEMINI_MODEL, contents=contents, config=config,
    )

    findings: list[Finding] = []
    imaging_calls = 0
    max_imaging_calls = 2
    seen_imaging: set[tuple[str, str]] = set()

    # Tool loop — handles parallel and sequential calls
    while has_function_calls(response):
        tool_calls = [
            part.function_call
            for part in response.candidates[-1].content.parts
            if part.function_call and part.function_call.name
        ]

        function_responses = []
        for tc in tool_calls:
            pid = tc.args.get("patient_id", patient_id)
            query = tc.args.get("query", user_query)

            if tc.name in RETRIEVAL_TOOL_NAMES:
                log_tool_call(tc.name, f"{pid}: {dict(tc.args)}")
                payload = dump_for_model(dispatch_retrieval(tc.name, pid, dict(tc.args)))
                log_tool_result(_brief_tool_result(payload))
                function_responses.append(
                    types.Part.from_function_response(name=tc.name, response=payload)
                )
                continue

            agents = DATACTRL_AGENTS if has_case(pid) else AGENT_REGISTRY
            agent = agents.get(tc.name) or AGENT_REGISTRY.get(tc.name)
            if not agent:
                log_tool_call(tc.name, "Unknown tool")
                function_responses.append(
                    types.Part.from_function_response(
                        name=tc.name, response={"error": f"Unknown tool: {tc.name}"},
                    )
                )
                continue

            log_tool_call(tc.name, f"{pid}: {query}")
            if tc.name == "analyze_imaging":
                resource_id = tc.args.get("resource_id")
                slice_index = tc.args.get("slice_index")
                if imaging_calls >= max_imaging_calls:
                    log_tool_result("analyze_imaging call cap reached")
                    function_responses.append(
                        types.Part.from_function_response(
                            name=tc.name,
                            response={
                                "error": (
                                    "analyze_imaging is limited to 2 calls per question. "
                                    "Use the existing preview unless you need a different "
                                    "series or slice_index."
                                )
                            },
                        )
                    )
                    continue
                key = (
                    str(resource_id or ""),
                    str(int(slice_index)) if slice_index is not None else "default",
                )
                same_series_default = any(
                    item[0] == key[0] and item[0]
                    for item in seen_imaging
                ) and slice_index is None
                if key in seen_imaging or same_series_default:
                    log_tool_result("duplicate analyze_imaging skipped")
                    function_responses.append(
                        types.Part.from_function_response(
                            name=tc.name,
                            response={
                                "error": (
                                    "This series/slice was already analyzed. "
                                    "Use that finding; do not call analyze_imaging again "
                                    "on the same preview."
                                )
                            },
                        )
                    )
                    continue
                seen_imaging.add(key)
                imaging_calls += 1
                finding = agent.analyze(
                    pid,
                    query,
                    resource_id=resource_id,
                    slice_index=int(slice_index) if slice_index is not None else None,
                )
            else:
                finding = agent.analyze(pid, query)
            findings.append(finding)
            log_tool_result(finding.summary)

            function_responses.append(
                types.Part.from_function_response(
                    name=tc.name, response=_finding_payload(finding),
                )
            )

        contents.append(response.candidates[-1].content)
        contents.append(types.Content(role="user", parts=function_responses))

        response = client.models.generate_content(
            model=GEMINI_MODEL, contents=contents, config=config,
        )

    # Extract final text
    gemini_text = response.candidates[-1].content.parts[-1].text

    # If no tools were called, return plain text
    if not findings:
        log_assistant(gemini_text)
        return gemini_text

    # Assemble response immediately with empty knowledge bus
    final_response = assemble_response(gemini_text, findings, {})
    log_assistant(final_response)

    # Fire knowledge bus async — will patch citations via WebSocket when done
    if len(findings) >= 2 and patient_id and patient_info:
        def _run_knowledge_bus():
            try:
                kb = build_knowledge_bus(findings, patient_id, patient_info)
                broadcast_knowledge_bus_update(findings, kb, on_complete=on_knowledge_bus_complete)
            except Exception as e:
                print(f"Knowledge bus error: {e}")
        threading.Thread(target=_run_knowledge_bus, daemon=True).start()

    return final_response


def _brief_tool_result(payload: dict) -> str:
    """Short log line so retrieval dumps do not flood the UI."""
    if "error" in payload:
        return str(payload["error"])
    if "counts" in payload:
        return f"inventory {payload['counts']}"
    if "id" in payload and "view" in payload:
        return f"{payload.get('view')} {payload.get('id')}"
    if payload.get("candidates") is not None:
        selected = payload.get("selected") or {}
        return (
            f"dicom_select confident={payload.get('confident')} "
            f"selected={selected.get('label') or selected.get('id')}"
        )
    if "label" in payload:
        return str(payload["label"])
    return str(list(payload.keys())[:8])
