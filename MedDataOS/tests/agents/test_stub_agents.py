import pytest

from src.agents.base_agent import BaseAgent, Finding
from src.agents.clinical_notes_agent import ClinicalNotesAgent
from src.agents.chest_xray_agent import ChestXrayAgent
from src.agents.echo_agent import EchoAgent
from src.agents.heart_sounds_agent import HeartSoundsAgent
from src.agents.lab_results_agent import LabResultsAgent
from src.agents.medication_agent import MedicationAgent


AGENT_PARAMS = [
    (ClinicalNotesAgent, "clinical_notes", "clinical-notes", ".txt"),
    (ChestXrayAgent,     "chest_xray",     "chest-xray",    ".png"),
    (EchoAgent,          "echo",           "echo",          ".mp4"),
    (HeartSoundsAgent,   "heart_sounds",   "heart-sounds",  ".wav"),
    (LabResultsAgent,    "lab_results",    "lab-results",   ".png"),
    (MedicationAgent,    "medication",     "medications",   ".csv"),
]


@pytest.mark.parametrize("agent_cls,modality,data_dir,file_ext", AGENT_PARAMS)
def test_agent_properties(agent_cls, modality, data_dir, file_ext):
    agent = agent_cls()
    assert agent.modality == modality
    assert agent.data_dir == data_dir
    assert agent.file_ext == file_ext


@pytest.mark.parametrize("agent_cls,modality,data_dir,file_ext", AGENT_PARAMS)
def test_agent_is_base_agent_subclass(agent_cls, modality, data_dir, file_ext):
    assert issubclass(agent_cls, BaseAgent)


@pytest.mark.parametrize("agent_cls,modality,data_dir,file_ext", AGENT_PARAMS)
def test_agent_analyze_returns_finding(agent_cls, modality, data_dir, file_ext):
    agent = agent_cls()
    result = agent.analyze("P0001", "test query")
    assert isinstance(result, Finding)


@pytest.mark.parametrize("agent_cls,modality,data_dir,file_ext", AGENT_PARAMS)
def test_agent_analyze_finding_fields(agent_cls, modality, data_dir, file_ext):
    agent = agent_cls()
    result = agent.analyze("P0001", "test query")
    assert result.agent == modality
    assert result.file == f"P0001{file_ext}"
    assert result.web_path == f"/multimodal-data/{data_dir}/P0001{file_ext}"
    assert len(result.summary) > 20
