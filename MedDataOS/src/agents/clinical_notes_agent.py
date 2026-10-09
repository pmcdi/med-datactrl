import json
from pathlib import Path

from src.agents.base_agent import BaseAgent, Finding

STUBS_DIR = Path(__file__).resolve().parents[2] / "stubs"


class ClinicalNotesAgent(BaseAgent):
    """Clinical notes agent.

    datactrl cases use the compile-then-navigate pipeline in ``src.agents.notes``
    (grounded fact ledger, timeline, conflicts, verified span citations).
    Demo patients return their stub summary.
    """

    modality = "clinical_notes"
    data_dir = "clinical-notes"
    file_ext = ".txt"

    def analyze(self, patient_id: str, query: str) -> Finding:
        from src.patients.catalog import has_case

        if has_case(patient_id):
            return self._analyze_case(patient_id, query)
        return self._analyze_stub(patient_id, query)

    def _analyze_stub(self, patient_id: str, query: str) -> Finding:
        with open(STUBS_DIR / f"{patient_id}.json") as f:
            stubs = json.load(f)
        data = stubs[self.modality]
        return Finding(
            agent=self.modality,
            file=f"{patient_id}{self.file_ext}",
            web_path=self.get_web_path(patient_id),
            summary=data["summary"],
        )

    def _analyze_case(self, patient_id: str, query: str) -> Finding:
        from src.agents.notes.agent import answer_question
        from src.retrieval.case_store import get_case

        case = get_case(patient_id)
        web_path = f"/api/cases/{patient_id}/notes"
        if not case.report.notes:
            return Finding(
                agent=self.modality,
                file="clinical_notes.txt",
                web_path=web_path,
                summary="No clinical notes were discovered for this case.",
            )
        result = answer_question(patient_id, query, case=case)
        stats = {k: v for k, v in result.stats.items() if k != "dropped_claims"}
        print(f"[notes] {patient_id} mode={result.mode} stats={stats}")
        return Finding(
            agent=self.modality,
            file="clinical_notes.txt",
            web_path=web_path,
            summary=result.summary,
            evidence=result.evidence,
            conflicts=result.conflicts,
        )
