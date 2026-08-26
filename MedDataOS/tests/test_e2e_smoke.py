# tests/test_e2e_smoke.py
"""
Smoke test: verifies the full pipeline can assemble a valid citation response
from stub agents without actually calling Gemini.
"""
import json
from unittest.mock import patch, MagicMock
from src.core.orchestrator import run_pipeline

@patch("src.core.orchestrator.create_client")
def test_e2e_single_modality(mock_orch_client):
    """Simulate Gemini calling analyze_ecg and composing a response.
    Knowledge bus is NOT called (single modality), so no need to mock it."""
    mock_client = MagicMock()
    mock_orch_client.return_value = mock_client

    # First call: Gemini returns a function call
    fc_part = MagicMock()
    fc_part.function_call = MagicMock()
    fc_part.function_call.name = "analyze_ecg"
    fc_part.function_call.args = {"patient_id": "P0001", "query": "cardiac workup"}
    response1 = MagicMock()
    response1.candidates = [MagicMock()]
    response1.candidates[-1].content.parts = [fc_part]

    # Second call: Gemini returns text with citation
    text_part = MagicMock()
    text_part.function_call = None
    text_part.text = "The ECG [cite:1] shows normal sinus rhythm."
    response2 = MagicMock()
    response2.candidates = [MagicMock()]
    response2.candidates[-1].content.parts = [text_part]

    mock_client.models.generate_content.side_effect = [response1, response2]

    patient_info = {"name": "James Carter", "age": 58, "sex": "M",
                    "blood_type": "A+", "allergies": ["Penicillin"],
                    "conditions": ["HTN"], "data_dates": {"ecg": "2026-02-03"}}

    result = run_pipeline("What does the ECG show?", patient_id="P0001", patient_info=patient_info)
    parsed = json.loads(result)

    assert "response" in parsed
    assert "[cite:1]" in parsed["response"]
    assert len(parsed["citations"]) == 1
    assert parsed["citations"][0]["agent"] == "ecg"
    assert parsed["citations"][0]["id"] == "1"
    assert "P0001.svg" in parsed["citations"][0]["file"]
