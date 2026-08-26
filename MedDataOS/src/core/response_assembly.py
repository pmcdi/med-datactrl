# src/core/response_assembly.py
import json
from src.agents.base_agent import Finding


def assemble_response(gemini_text: str, findings: list[Finding], knowledge_bus: dict) -> str:
    citations = []
    for i, finding in enumerate(findings):
        citations.append({
            "id": str(i + 1),
            "agent": finding.agent,
            "file": finding.file,
            "web_path": finding.web_path,
            "summary": finding.summary,
            "knowledge_bus": knowledge_bus.get(finding.agent, {
                "supported_by": [],
                "contradicted_by": [],
            }),
        })
    return json.dumps({"response": gemini_text, "citations": citations}, ensure_ascii=False)
