from google.genai import types
from src.tools.definitions import TOOL_DECLARATIONS, tools

EXPECTED_TOOL_NAMES = {
    "analyze_clinical_notes",
    "analyze_chest_xray",
    "analyze_ecg",
    "analyze_echo",
    "analyze_heart_sounds",
    "analyze_lab_results",
    "analyze_medication",
}


def test_seven_tools_declared():
    assert len(TOOL_DECLARATIONS) == 7


def test_all_expected_tool_names_present():
    names = {t["name"] for t in TOOL_DECLARATIONS}
    assert names == EXPECTED_TOOL_NAMES


def test_each_tool_has_patient_id_and_query():
    for tool in TOOL_DECLARATIONS:
        props = tool["parameters"]["properties"]
        assert "patient_id" in props, f"{tool['name']} missing patient_id"
        assert "query" in props, f"{tool['name']} missing query"
        assert tool["parameters"]["required"] == ["patient_id", "query"]


def test_tools_object_is_types_tool():
    assert isinstance(tools, types.Tool)


def test_each_tool_has_description():
    for tool in TOOL_DECLARATIONS:
        assert tool["description"], f"{tool['name']} has empty description"
