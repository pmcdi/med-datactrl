"""Tool layer exposing the Controller to LLM agents."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datactrl import Controller

DATA_ROOT = Path(__file__).parent.parent / "data" / "rawdata"

PATIENT_DIRS: dict[str, Path] = {
    "KiTS-00008": DATA_ROOT / "KiTS-00008",
    "Adrenal_Ki67_Seg_002": DATA_ROOT / "Adrenal_Ki67_Seg_002",
    "STS_002": DATA_ROOT / "STS_002",
    "TCGA-G3-AAV7": DATA_ROOT / "TCGA-G3-AAV7",
    "TCGA-DD-A4NF": DATA_ROOT / "TCGA-DD-A4NF",
}


def _patient_parameter() -> dict[str, Any]:
    return {"type": "string", "enum": list(PATIENT_DIRS)}


def _tool(
    name: str,
    description: str,
    properties: dict[str, Any],
    required: list[str],
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


TOOL_DEFINITIONS: list[dict[str, Any]] = [
    _tool(
        "get_dicom_patient_info",
        "Return DICOM patient/series hierarchy including child series.",
        {"patient": _patient_parameter()},
        ["patient"],
    ),
    _tool(
        "list_dicom_atoms",
        "List DICOM metadata getter atoms available for a patient dataset.",
        {"patient": _patient_parameter()},
        ["patient"],
    ),
    _tool(
        "call_dicom_atom",
        "Fetch one DICOM metadata field by atom name for a series.",
        {
            "patient": _patient_parameter(),
            "name": {"type": "string"},
            "series_id": {"type": "string"},
        },
        ["patient", "name", "series_id"],
    ),
    _tool(
        "list_clinical_notes",
        "List clinical note summaries (note_id, filename, preview) for a patient.",
        {"patient": _patient_parameter()},
        ["patient"],
    ),
    _tool(
        "get_clinical_note",
        "Return the full text or parsed JSON for one clinical note.",
        {
            "patient": _patient_parameter(),
            "note_id": {"type": "string"},
        },
        ["patient", "note_id"],
    ),
    _tool(
        "get_pathology_info",
        "Return pathology slide summaries (slide_id, dimensions, vendor, etc.).",
        {"patient": _patient_parameter()},
        ["patient"],
    ),
    _tool(
        "list_slide_atoms",
        "List pathology slide metadata atom getters available for a patient.",
        {"patient": _patient_parameter()},
        ["patient"],
    ),
    _tool(
        "call_slide_atom",
        "Fetch one pathology slide metadata field by atom name.",
        {
            "patient": _patient_parameter(),
            "name": {"type": "string"},
            "slide_id": {"type": "string"},
        },
        ["patient", "name", "slide_id"],
    ),
]


class ControllerToolSession:
    """Stateful controller access for one benchmark run."""

    def __init__(self) -> None:
        self._controllers: dict[str, Controller] = {}
        self.atom_calls: list[dict[str, str]] = []
        self.tool_calls: list[dict[str, Any]] = []
        self._tool_handlers = {
            "get_dicom_patient_info": self.get_dicom_patient_info,
            "list_dicom_atoms": self.list_dicom_atoms,
            "call_dicom_atom": self.call_dicom_atom,
            "list_clinical_notes": self.list_clinical_notes,
            "get_clinical_note": self.get_clinical_note,
            "get_pathology_info": self.get_pathology_info,
            "list_slide_atoms": self.list_slide_atoms,
            "call_slide_atom": self.call_slide_atom,
        }

    def _controller(self, patient: str) -> Controller:
        if patient not in PATIENT_DIRS:
            msg = f"Unknown patient {patient!r}"
            raise ValueError(msg)
        if patient not in self._controllers:
            self._controllers[patient] = Controller.from_patient_directory(
                PATIENT_DIRS[patient]
            )
        return self._controllers[patient]

    def get_dicom_patient_info(self, patient: str) -> dict[str, Any]:
        return self._controller(patient).get_dicom_patient_info()

    def list_dicom_atoms(self, patient: str) -> list[str]:
        return self._controller(patient).list_dicom_atoms()

    def call_dicom_atom(self, patient: str, name: str, series_id: str) -> Any:
        self.atom_calls.append(
            {"patient": patient, "name": name, "series_id": series_id}
        )
        return self._controller(patient).call_dicom_atom(name, series_id)

    def list_clinical_notes(self, patient: str) -> list[dict[str, Any]]:
        return self._controller(patient).list_clinical_notes()

    def get_clinical_note(self, patient: str, note_id: str) -> str | dict[str, Any]:
        return self._controller(patient).get_clinical_note(note_id)

    def get_pathology_info(self, patient: str) -> list[dict[str, Any]]:
        return self._controller(patient).get_pathology_info()

    def list_slide_atoms(self, patient: str) -> list[str]:
        return self._controller(patient).list_slide_atoms()

    def call_slide_atom(self, patient: str, name: str, slide_id: str) -> Any:
        self.atom_calls.append(
            {"patient": patient, "name": name, "slide_id": slide_id}
        )
        return self._controller(patient).call_slide_atom(name, slide_id)

    def dispatch(self, name: str, arguments: str | dict[str, Any]) -> str:
        args = json.loads(arguments) if isinstance(arguments, str) else arguments
        self.tool_calls.append({"name": name, "arguments": args})
        handler = self._tool_handlers.get(name)
        if handler is None:
            raise ValueError(f"Unknown tool: {name!r}")
        result = handler(**args)

        return json.dumps(result, default=str)

    def reset_trace(self) -> None:
        self.atom_calls = []
        self.tool_calls = []
