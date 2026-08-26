import json
from pathlib import Path

from src.agents.base_agent import BaseAgent, Finding

STUBS_DIR = Path(__file__).resolve().parents[2] / "stubs"

_ANALYSIS_PROMPT = """You are a clinical notes specialist. Given bounded note text from a patient case, answer the clinician's question.

Question: {query}

Notes:
{notes}

Write a concise clinical summary (4-8 sentences). Include specific values, dates, and note filenames when they matter. Do not invent findings that are not in the notes."""


class ClinicalNotesAgent(BaseAgent):
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
        from src.core.gemini_client import GEMINI_MODEL, create_client
        from src.retrieval.case_store import get_case

        case = get_case(patient_id)
        notes = list(case.report.notes)
        if not notes:
            return Finding(
                agent=self.modality,
                file="clinical_notes.txt",
                web_path=f"/api/cases/{patient_id}/notes",
                summary="No clinical notes were discovered for this case.",
            )

        chunks: list[str] = []
        for note in notes:
            fetched = case.fetch(note.id, view="text")
            body = fetched["data"].get("text") or ""
            chunks.append(f"## {note.label} ({note.path})\n{body}")
        combined = "\n\n".join(chunks)
        if len(combined) > 16_000:
            combined = combined[:16_000] + "\n\n[truncated]"

        client = create_client()
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=_ANALYSIS_PROMPT.format(query=query, notes=combined),
        )
        summary = (response.text or "").strip() or "Notes were retrieved but no summary was produced."
        return Finding(
            agent=self.modality,
            file="clinical_notes.txt",
            web_path=f"/api/cases/{patient_id}/notes",
            summary=summary,
        )
