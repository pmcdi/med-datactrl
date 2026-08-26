from unittest.mock import MagicMock, patch
from src.core.orchestrator import AGENT_REGISTRY, has_function_calls, run_pipeline


def test_agent_registry_has_all_seven_entries():
    expected = {
        "analyze_clinical_notes",
        "analyze_chest_xray",
        "analyze_ecg",
        "analyze_echo",
        "analyze_heart_sounds",
        "analyze_lab_results",
        "analyze_medication",
    }
    assert set(AGENT_REGISTRY.keys()) == expected


def test_has_function_calls_true():
    fc = MagicMock()
    fc.function_call.name = "analyze_ecg"

    part_text = MagicMock()
    part_text.function_call = None

    response = MagicMock()
    response.candidates = [MagicMock()]
    response.candidates[-1].content.parts = [part_text, fc]

    assert has_function_calls(response) is True


def test_has_function_calls_false_text_only():
    part = MagicMock()
    part.function_call = None

    response = MagicMock()
    response.candidates = [MagicMock()]
    response.candidates[-1].content.parts = [part]

    assert has_function_calls(response) is False


def test_has_function_calls_false_empty():
    response = MagicMock()
    response.candidates = []

    assert has_function_calls(response) is False


@patch("src.core.orchestrator.create_client")
@patch("src.core.orchestrator.log_user")
@patch("src.core.orchestrator.log_assistant")
def test_run_pipeline_plain_text_no_tools(mock_log_asst, mock_log_user, mock_create_client):
    """When Gemini returns text without function calls, run_pipeline returns plain text."""
    # Build a mock response with no function calls
    text_part = MagicMock()
    text_part.function_call = None
    text_part.text = "Hello, how can I help?"

    candidate = MagicMock()
    candidate.content.parts = [text_part]

    response = MagicMock()
    response.candidates = [candidate]

    client = MagicMock()
    client.models.generate_content.return_value = response
    mock_create_client.return_value = client

    result = run_pipeline("Hi there")

    assert result == "Hello, how can I help?"
    mock_log_user.assert_called_once_with("Hi there")
    mock_log_asst.assert_called_once_with("Hello, how can I help?")
