from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Finding:
    agent: str      # "ecg", "chest_xray", etc.
    file: str       # "P0001.svg"
    web_path: str   # "/multimodal-data/ecg/P0001.svg"
    summary: str    # "Sinus rhythm 72 bpm, ST depression V4-V6"


class BaseAgent(ABC):
    modality: str
    data_dir: str
    file_ext: str

    @abstractmethod
    def analyze(self, patient_id: str, query: str) -> Finding: ...

    def get_file_path(self, patient_id: str) -> str:
        return f"multimodal-data/{self.data_dir}/{patient_id}{self.file_ext}"

    def get_web_path(self, patient_id: str) -> str:
        return f"/multimodal-data/{self.data_dir}/{patient_id}{self.file_ext}"
