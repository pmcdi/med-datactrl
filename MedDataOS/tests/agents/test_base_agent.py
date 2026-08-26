from dataclasses import fields
from src.agents.base_agent import BaseAgent, Finding


class DummyAgent(BaseAgent):
    modality = "dummy"
    data_dir = "dummy"
    file_ext = ".txt"

    def analyze(self, patient_id: str, query: str) -> Finding:
        return Finding(
            agent=self.modality,
            file=f"{patient_id}{self.file_ext}",
            web_path=self.get_web_path(patient_id),
            summary="dummy summary",
        )


def test_finding_is_dataclass():
    field_names = {f.name for f in fields(Finding)}
    assert field_names == {"agent", "file", "web_path", "summary"}


def test_finding_instantiation():
    f = Finding(agent="ecg", file="P0001.svg", web_path="/multimodal-data/ecg/P0001.svg", summary="Normal sinus rhythm")
    assert f.agent == "ecg"
    assert f.file == "P0001.svg"
    assert f.web_path == "/multimodal-data/ecg/P0001.svg"
    assert f.summary == "Normal sinus rhythm"


def test_base_agent_is_abstract():
    import inspect
    assert inspect.isabstract(BaseAgent)


def test_dummy_agent_get_file_path():
    agent = DummyAgent()
    assert agent.get_file_path("P0001") == "multimodal-data/dummy/P0001.txt"


def test_dummy_agent_get_web_path():
    agent = DummyAgent()
    assert agent.get_web_path("P0001") == "/multimodal-data/dummy/P0001.txt"


def test_dummy_agent_analyze_returns_finding():
    agent = DummyAgent()
    result = agent.analyze("P0001", "What is the result?")
    assert isinstance(result, Finding)
    assert result.agent == "dummy"
    assert result.file == "P0001.txt"
    assert result.web_path == "/multimodal-data/dummy/P0001.txt"
    assert result.summary == "dummy summary"
