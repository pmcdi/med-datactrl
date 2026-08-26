import json
from google.genai import types
from src.agents.base_agent import Finding
from src.agents.prompts import KNOWLEDGE_BUS_PROMPT_TEMPLATE
from src.core.gemini_client import create_client

GEMINI_MODEL = "gemini-3.1-pro-preview"


def _build_knowledge_bus_prompt(findings: list[Finding], patient_id: str, patient_info: dict) -> str:
    numbered = "\n".join(
        f"{i+1}. {f.agent}: {f.summary}" for i, f in enumerate(findings)
    )
    return KNOWLEDGE_BUS_PROMPT_TEMPLATE.format(
        patient_name=patient_info.get("name", patient_id),
        patient_id=patient_id,
        age=patient_info.get("age", "?"),
        sex=patient_info.get("sex", "?"),
        conditions=", ".join(patient_info.get("conditions", [])) or "None",
        numbered_findings_list=numbered,
    )


def _empty_bus(findings: list[Finding]) -> dict:
    return {f.agent: {"supported_by": [], "contradicted_by": []} for f in findings}


def _build_response_schema(findings: list[Finding]) -> dict:
    cross_ref_item = {
        "type": "object",
        "properties": {
            "agent": {"type": "string"},
            "finding": {"type": "string"},
            "reason": {"type": "string"},
        },
        "required": ["agent", "finding", "reason"],
    }
    agent_entry = {
        "type": "object",
        "properties": {
            "supported_by": {"type": "array", "items": cross_ref_item},
            "contradicted_by": {"type": "array", "items": cross_ref_item},
        },
        "required": ["supported_by", "contradicted_by"],
    }
    return {
        "type": "object",
        "properties": {f.agent: agent_entry for f in findings},
        "required": [f.agent for f in findings],
    }


def build_knowledge_bus(findings: list[Finding], patient_id: str, patient_info: dict) -> dict:
    prompt = _build_knowledge_bus_prompt(findings, patient_id, patient_info)
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=_build_response_schema(findings),
    )
    try:
        client = create_client()
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
            config=config,
        )
        bus_data = json.loads(response.text)
    except Exception:
        return _empty_bus(findings)

    result = _empty_bus(findings)
    for f in findings:
        if f.agent in bus_data and isinstance(bus_data[f.agent], dict):
            result[f.agent] = {
                "supported_by": bus_data[f.agent].get("supported_by", []),
                "contradicted_by": bus_data[f.agent].get("contradicted_by", []),
            }
    return result
