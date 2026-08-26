from src.agents.prompts import BASE_SYSTEM_INSTRUCTION, KNOWLEDGE_BUS_PROMPT_TEMPLATE, build_system_prompt


def test_base_system_instruction_contains_tool_names():
    tool_names = [
        "analyze_clinical_notes",
        "analyze_chest_xray",
        "analyze_ecg",
        "analyze_echo",
        "analyze_heart_sounds",
        "analyze_lab_results",
        "analyze_medication",
    ]
    for name in tool_names:
        assert name in BASE_SYSTEM_INSTRUCTION, f"{name} missing from BASE_SYSTEM_INSTRUCTION"


def test_base_system_instruction_contains_cite_format():
    assert "[cite:N]" in BASE_SYSTEM_INSTRUCTION


def test_build_system_prompt_without_patient_returns_base():
    result = build_system_prompt(None, None)
    assert result == BASE_SYSTEM_INSTRUCTION


def test_build_system_prompt_without_patient_id_returns_base():
    result = build_system_prompt(None, {"name": "Alice"})
    assert result == BASE_SYSTEM_INSTRUCTION


def test_build_system_prompt_without_patient_info_returns_base():
    result = build_system_prompt("P0001", None)
    assert result == BASE_SYSTEM_INSTRUCTION


def test_build_system_prompt_with_patient_includes_patient_data():
    patient_info = {
        "name": "John Doe",
        "age": 55,
        "sex": "M",
        "blood_type": "A+",
        "allergies": ["penicillin"],
        "conditions": ["hypertension", "diabetes"],
        "data_dates": {
            "ecg": "2024-01-15",
            "chest_xray": "2024-01-10",
        },
    }
    result = build_system_prompt("P0001", patient_info)
    assert "John Doe" in result
    assert "P0001" in result
    assert "55" in result
    assert "A+" in result
    assert "penicillin" in result
    assert "hypertension" in result
    assert "diabetes" in result
    assert BASE_SYSTEM_INSTRUCTION in result


def test_build_system_prompt_with_patient_includes_data_dates():
    patient_info = {
        "name": "Jane Smith",
        "age": 40,
        "sex": "F",
        "blood_type": "O-",
        "allergies": [],
        "conditions": [],
        "data_dates": {
            "ecg": "2024-03-01",
            "lab_results": "2024-03-05",
        },
    }
    result = build_system_prompt("P0002", patient_info)
    assert "ecg: 2024-03-01" in result
    assert "lab_results: 2024-03-05" in result


def test_knowledge_bus_prompt_template_has_all_placeholders():
    placeholders = [
        "{patient_name}",
        "{patient_id}",
        "{age}",
        "{sex}",
        "{conditions}",
        "{numbered_findings_list}",
    ]
    for placeholder in placeholders:
        assert placeholder in KNOWLEDGE_BUS_PROMPT_TEMPLATE, f"{placeholder} missing from template"
