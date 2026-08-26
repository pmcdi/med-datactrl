import json
from pathlib import Path

import pytest

from src.agents.clinical_notes_agent import ClinicalNotesAgent
from src.agents.chest_xray_agent import ChestXrayAgent
from src.agents.ecg_agent import EcgAgent
from src.agents.echo_agent import EchoAgent
from src.agents.heart_sounds_agent import HeartSoundsAgent
from src.agents.lab_results_agent import LabResultsAgent
from src.agents.medication_agent import MedicationAgent

STUBS_DIR = Path(__file__).resolve().parents[2] / "stubs"

ALL_PATIENTS = [f"P{i:04d}" for i in range(1, 11)]

MODALITIES = [
    "clinical_notes",
    "chest_xray",
    "ecg",
    "echo",
    "heart_sounds",
    "lab_results",
    "medication",
]

ALL_AGENTS = [
    ClinicalNotesAgent(),
    ChestXrayAgent(),
    EcgAgent(),
    EchoAgent(),
    HeartSoundsAgent(),
    LabResultsAgent(),
    MedicationAgent(),
]


@pytest.mark.parametrize("patient_id", ALL_PATIENTS)
def test_fixture_file_exists(patient_id):
    fixture_path = STUBS_DIR / f"{patient_id}.json"
    assert fixture_path.exists(), f"Missing fixture: {fixture_path}"


@pytest.mark.parametrize("patient_id", ALL_PATIENTS)
def test_fixture_has_all_modalities(patient_id):
    with open(STUBS_DIR / f"{patient_id}.json") as f:
        data = json.load(f)
    for modality in MODALITIES:
        assert modality in data, f"{patient_id} missing modality: {modality}"


@pytest.mark.parametrize("patient_id", ALL_PATIENTS)
def test_fixture_summaries_present_and_non_trivial(patient_id):
    with open(STUBS_DIR / f"{patient_id}.json") as f:
        data = json.load(f)
    for modality in MODALITIES:
        summary = data[modality].get("summary", "")
        assert isinstance(summary, str), f"{patient_id}/{modality}: summary is not a string"
        assert len(summary) > 20, (
            f"{patient_id}/{modality}: summary too short ({len(summary)} chars): {summary!r}"
        )


@pytest.mark.parametrize("patient_id", ALL_PATIENTS)
@pytest.mark.parametrize("agent", ALL_AGENTS, ids=[a.modality for a in ALL_AGENTS])
def test_agent_can_load_fixture(patient_id, agent):
    from src.agents.base_agent import Finding
    result = agent.analyze(patient_id, "test query")
    assert isinstance(result, Finding)
    assert result.agent == agent.modality
    assert len(result.summary) > 20
