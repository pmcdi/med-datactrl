import json

from src.agents.base_agent import Finding
from src.core.response_assembly import assemble_response


def test_single_citation_empty_knowledge_bus():
    finding = Finding(
        agent="ecg",
        file="P0001.svg",
        web_path="/multimodal-data/ecg/P0001.svg",
        summary="Sinus rhythm 72 bpm",
    )
    result = json.loads(assemble_response("Normal ECG.", [finding], {}))

    assert result["response"] == "Normal ECG."
    assert len(result["citations"]) == 1

    cite = result["citations"][0]
    assert cite["id"] == "1"
    assert cite["agent"] == "ecg"
    assert cite["file"] == "P0001.svg"
    assert cite["web_path"] == "/multimodal-data/ecg/P0001.svg"
    assert cite["summary"] == "Sinus rhythm 72 bpm"
    assert cite["knowledge_bus"] == {"supported_by": [], "contradicted_by": []}


def test_multiple_citations_with_knowledge_bus():
    findings = [
        Finding(agent="ecg", file="P0001.svg", web_path="/multimodal-data/ecg/P0001.svg", summary="ST depression V4-V6"),
        Finding(agent="chest_xray", file="P0001.png", web_path="/multimodal-data/chest-xray/P0001.png", summary="Cardiomegaly"),
    ]
    knowledge_bus = {
        "ecg": {"supported_by": ["chest_xray"], "contradicted_by": []},
        "chest_xray": {"supported_by": ["ecg"], "contradicted_by": []},
    }

    result = json.loads(assemble_response("Cardiac findings.", findings, knowledge_bus))

    assert len(result["citations"]) == 2
    assert result["citations"][0]["id"] == "1"
    assert result["citations"][1]["id"] == "2"
    assert result["citations"][0]["knowledge_bus"]["supported_by"] == ["chest_xray"]
    assert result["citations"][1]["knowledge_bus"]["supported_by"] == ["ecg"]


def test_valid_json_with_special_characters():
    finding = Finding(
        agent="clinical_notes",
        file="P0001.txt",
        web_path="/multimodal-data/clinical-notes/P0001.txt",
        summary='Patient said "I feel fine" \\ BP 120/80',
    )
    raw = assemble_response('He reported "no pain" \\ stable', [finding], {})

    # Must be valid JSON
    result = json.loads(raw)
    assert '"I feel fine"' in result["citations"][0]["summary"]
    assert "\\" in result["citations"][0]["summary"]
    assert '"no pain"' in result["response"]
