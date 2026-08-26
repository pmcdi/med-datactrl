import json
from unittest.mock import MagicMock, patch
from src.agents.base_agent import Finding
from src.agents.knowledge_bus import _build_knowledge_bus_prompt, build_knowledge_bus, _empty_bus


PATIENT_ID = "P0001"
PATIENT_INFO = {
    "name": "Alice Cardio",
    "age": 62,
    "sex": "F",
    "conditions": ["heart failure", "hypertension"],
}

FINDINGS = [
    Finding(agent="ecg", file="P0001.svg", web_path="/multimodal-data/ecg/P0001.svg", summary="ST depression V4-V6"),
    Finding(agent="chest_xray", file="P0001.png", web_path="/multimodal-data/chest_xray/P0001.png", summary="Cardiomegaly present"),
]


def test_prompt_builder_includes_patient_name():
    prompt = _build_knowledge_bus_prompt(FINDINGS, PATIENT_ID, PATIENT_INFO)
    assert "Alice Cardio" in prompt
    assert PATIENT_ID in prompt


def test_prompt_builder_includes_findings():
    prompt = _build_knowledge_bus_prompt(FINDINGS, PATIENT_ID, PATIENT_INFO)
    assert "ST depression V4-V6" in prompt
    assert "Cardiomegaly present" in prompt
    assert "ecg" in prompt
    assert "chest_xray" in prompt


def test_prompt_builder_numbers_findings():
    prompt = _build_knowledge_bus_prompt(FINDINGS, PATIENT_ID, PATIENT_INFO)
    assert "1." in prompt
    assert "2." in prompt


def test_empty_bus_structure():
    result = _empty_bus(FINDINGS)
    assert "ecg" in result
    assert "chest_xray" in result
    for agent_data in result.values():
        assert agent_data == {"supported_by": [], "contradicted_by": []}


def test_build_knowledge_bus_calls_gemini_and_parses_json():
    valid_response = {
        "ecg": {"supported_by": ["chest_xray"], "contradicted_by": []},
        "chest_xray": {"supported_by": [], "contradicted_by": []},
    }
    mock_response = MagicMock()
    mock_response.text = json.dumps(valid_response)

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    with patch("src.agents.knowledge_bus.create_client", return_value=mock_client):
        result = build_knowledge_bus(FINDINGS, PATIENT_ID, PATIENT_INFO)

    mock_client.models.generate_content.assert_called_once()
    assert result["ecg"]["supported_by"] == ["chest_xray"]
    assert result["ecg"]["contradicted_by"] == []
    assert result["chest_xray"]["supported_by"] == []


def test_build_knowledge_bus_handles_malformed_response():
    mock_response = MagicMock()
    mock_response.text = "this is not valid json {"

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    with patch("src.agents.knowledge_bus.create_client", return_value=mock_client):
        result = build_knowledge_bus(FINDINGS, PATIENT_ID, PATIENT_INFO)

    # Should return empty bus on parse failure
    assert result == _empty_bus(FINDINGS)


def test_build_knowledge_bus_handles_api_exception():
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = Exception("API error")

    with patch("src.agents.knowledge_bus.create_client", return_value=mock_client):
        result = build_knowledge_bus(FINDINGS, PATIENT_ID, PATIENT_INFO)

    assert result == _empty_bus(FINDINGS)


def test_build_knowledge_bus_handles_partial_response():
    # Only ecg in response, chest_xray missing
    partial_response = {
        "ecg": {"supported_by": ["chest_xray"], "contradicted_by": []},
    }
    mock_response = MagicMock()
    mock_response.text = json.dumps(partial_response)

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    with patch("src.agents.knowledge_bus.create_client", return_value=mock_client):
        result = build_knowledge_bus(FINDINGS, PATIENT_ID, PATIENT_INFO)

    # ecg should be populated, chest_xray should have empty bus defaults
    assert result["ecg"]["supported_by"] == ["chest_xray"]
    assert result["chest_xray"] == {"supported_by": [], "contradicted_by": []}
