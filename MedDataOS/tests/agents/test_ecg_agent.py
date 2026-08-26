from src.agents.base_agent import BaseAgent, Finding
from src.agents.ecg_agent import EcgAgent


def test_ecg_agent_properties():
    agent = EcgAgent()
    assert agent.modality == "ecg"
    assert agent.data_dir == "ecg"
    assert agent.file_ext == ".svg"


def test_ecg_agent_is_base_agent_subclass():
    assert issubclass(EcgAgent, BaseAgent)


def test_ecg_agent_get_file_path():
    agent = EcgAgent()
    assert agent.get_file_path("P0001") == "multimodal-data/ecg/P0001.svg"


def test_ecg_agent_get_web_path():
    agent = EcgAgent()
    assert agent.get_web_path("P0001") == "/multimodal-data/ecg/P0001.svg"


def test_ecg_agent_analyze_returns_finding():
    agent = EcgAgent()
    result = agent.analyze("P0001", "What does the ECG show?")
    assert isinstance(result, Finding)


def test_ecg_agent_analyze_finding_fields():
    agent = EcgAgent()
    result = agent.analyze("P0001", "What does the ECG show?")
    assert result.agent == "ecg"
    assert result.file == "P0001.svg"
    assert result.web_path == "/multimodal-data/ecg/P0001.svg"
    assert "72 bpm" in result.summary
    assert "ST-segment depression" in result.summary
