"""
Main orchestrator for the multimodal medical agent system.
Coordinates between specialized agents using Gemini's function calling.
"""
import threading
from google.genai import types

from src.tools.definitions import tools
from src.utils.logging import log_user, log_assistant, log_tool_call, log_tool_result, broadcast_knowledge_bus_update
from src.agents.prompts import build_system_prompt
from src.agents.knowledge_bus import build_knowledge_bus
from src.core.gemini_client import create_client, create_config
from src.core.response_assembly import assemble_response
from src.agents.base_agent import Finding

from src.agents.clinical_notes_agent import ClinicalNotesAgent
from src.agents.chest_xray_agent import ChestXrayAgent
from src.agents.ecg_agent import EcgAgent
from src.agents.echo_agent import EchoAgent
from src.agents.heart_sounds_agent import HeartSoundsAgent
from src.agents.lab_results_agent import LabResultsAgent
from src.agents.medication_agent import MedicationAgent

GEMINI_MODEL = "gemini-3.1-pro-preview"

AGENT_REGISTRY = {
    "analyze_clinical_notes": ClinicalNotesAgent(),
    "analyze_chest_xray": ChestXrayAgent(),
    "analyze_ecg": EcgAgent(),
    "analyze_echo": EchoAgent(),
    "analyze_heart_sounds": HeartSoundsAgent(),
    "analyze_lab_results": LabResultsAgent(),
    "analyze_medication": MedicationAgent(),
}


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
    system_prompt = build_system_prompt(patient_id, patient_info)
    config = create_config(tools, system_prompt)

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

    # Tool loop — handles parallel and sequential calls
    while has_function_calls(response):
        tool_calls = [
            part.function_call
            for part in response.candidates[-1].content.parts
            if part.function_call and part.function_call.name
        ]

        function_responses = []
        for tc in tool_calls:
            agent = AGENT_REGISTRY.get(tc.name)
            if not agent:
                log_tool_call(tc.name, "Unknown tool")
                function_responses.append(
                    types.Part.from_function_response(
                        name=tc.name, response={"error": f"Unknown tool: {tc.name}"},
                    )
                )
                continue

            pid = tc.args.get("patient_id", patient_id)
            query = tc.args.get("query", user_query)
            log_tool_call(tc.name, f"{pid}: {query}")

            finding = agent.analyze(pid, query)
            findings.append(finding)
            log_tool_result(finding.summary)

            function_responses.append(
                types.Part.from_function_response(
                    name=tc.name, response={"summary": finding.summary},
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
