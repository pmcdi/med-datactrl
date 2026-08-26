"""Patient catalog: stub demo patients vs datactrl-backed case roots.

Filesystem paths stay here. Orchestrator tools receive ``patient_id`` only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
MEDDATAOS_ROOT = WORKSPACE_ROOT / "MedDataOS"

DATACTRL_BACKEND = "datactrl"
STUB_BACKEND = "stub"

# Add a new case by appending an entry: id, case_root, demographics, modality labels.
# Agents and retrieval tools pick this up automatically via has_case() / get_patient().
DATACTRL_PATIENTS: dict[str, dict[str, Any]] = {
    "TCGA-DD-A4NF": {
        "id": "TCGA-DD-A4NF",
        "name": "TCGA-DD-A4NF",
        "age": 72,
        "sex": "M",
        "blood_type": None,
        "allergies": [],
        "conditions": ["HCV cirrhosis", "HCC", "Hypertension"],
        "data_dates": {
            "clinical-notes": "2011-03-14",
            "radiology": "2011-03",
        },
        "modalities": [
            {"label": "Notes", "date": "2011-03-14"},
            {"label": "Radiology", "date": "2011-03"},
        ],
        "backend": DATACTRL_BACKEND,
        "case_root": WORKSPACE_ROOT / "data" / "TCGA-DD-A4NF",
    },
    "RADCURE-0006": {
        "id": "RADCURE-0006",
        "name": "RADCURE-0006",
        "age": 87,
        "sex": "M",
        "blood_type": None,
        "allergies": [],
        "conditions": ["Glottic SCC T1bN0M0"],
        "data_dates": {
            "clinical-notes": "2005-12-15",
            "radiology": "2006-01",
        },
        "modalities": [
            {"label": "Notes", "date": "2005-12-15"},
            {"label": "Radiology", "date": "2006-01"},
        ],
        "backend": DATACTRL_BACKEND,
        "case_root": WORKSPACE_ROOT / "data" / "RADCURE-0006",
    },
}


def get_patient(patient_id: str | None) -> dict[str, Any] | None:
    """Return a catalog record, or None when the id is not datactrl-backed."""
    if not patient_id:
        return None
    return DATACTRL_PATIENTS.get(patient_id)


def has_case(patient_id: str | None) -> bool:
    """True when this patient is served from a datactrl case root."""
    record = get_patient(patient_id)
    return bool(record and record.get("backend") == DATACTRL_BACKEND)


def case_root(patient_id: str) -> Path:
    """Return the filesystem root for a datactrl patient."""
    record = get_patient(patient_id)
    if record is None:
        raise KeyError(f"Unknown datactrl patient: {patient_id!r}")
    root = Path(record["case_root"])
    if not root.is_dir():
        raise FileNotFoundError(f"Case root is not a directory: {root}")
    return root


def list_datactrl_patients() -> list[dict[str, Any]]:
    """Sidebar-safe patient dicts (no filesystem paths)."""
    out: list[dict[str, Any]] = []
    for record in DATACTRL_PATIENTS.values():
        out.append(
            {
                "id": record["id"],
                "name": record.get("name", record["id"]),
                "age": record.get("age"),
                "sex": record.get("sex"),
                "blood_type": record.get("blood_type"),
                "allergies": record.get("allergies", []),
                "conditions": record.get("conditions", []),
                "modalities": record.get("modalities", []),
                "data_dates": record.get("data_dates", {}),
                "backend": DATACTRL_BACKEND,
            }
        )
    return out
