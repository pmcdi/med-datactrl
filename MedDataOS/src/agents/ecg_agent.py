import json
from pathlib import Path

from src.agents.base_agent import BaseAgent, Finding

STUBS_DIR = Path(__file__).resolve().parents[2] / "stubs"


class EcgAgent(BaseAgent):
    modality = "ecg"
    data_dir = "ecg"
    file_ext = ".svg"

    def analyze(self, patient_id: str, query: str) -> Finding:
        with open(STUBS_DIR / f"{patient_id}.json") as f:
            stubs = json.load(f)
        data = stubs[self.modality]
        return Finding(
            agent=self.modality,
            file=f"{patient_id}{self.file_ext}",
            web_path=self.get_web_path(patient_id),
            summary=data["summary"],
        )
