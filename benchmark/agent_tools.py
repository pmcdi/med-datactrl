"""Tool layer exposing the Controller to LLM agents."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datactrl import Controller

DATA_ROOT = Path(__file__).parent.parent / "data"

PATIENT_DIRS: dict[str, Path] = {
    "KiTS-00008": DATA_ROOT / "KiTS-00008",
    "Adrenal_Ki67_Seg_002": DATA_ROOT / "Adrenal_Ki67_Seg_002",
    "STS_002": DATA_ROOT / "STS_002",
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
        "get_patient_info",
        "Return patient/series hierarchy including child series.",
        {"patient": _patient_parameter()},
        ["patient"],
    ),
    _tool(
        "list_atoms",
        "List metadata getter atoms available for a patient dataset.",
        {"patient": _patient_parameter()},
        ["patient"],
    ),
    _tool(
        "call_atom",
        "Fetch one metadata field by atom name for a series.",
        {
            "patient": _patient_parameter(),
            "name": {"type": "string"},
            "series_id": {"type": "string"},
        },
        ["patient", "name", "series_id"],
    ),
]


class ControllerToolSession:
    """Stateful controller access for one benchmark run."""

    def __init__(self) -> None:
        self._controllers: dict[str, Controller] = {}
        self.atom_calls: list[dict[str, str]] = []
        self.tool_calls: list[dict[str, Any]] = []
        self._tool_handlers = {
            "get_patient_info": self.get_patient_info,
            "list_atoms": self.list_atoms,
            "call_atom": self.call_atom,
        }

    def _controller(self, patient: str) -> Controller:
        if patient not in PATIENT_DIRS:
            msg = f"Unknown patient {patient!r}"
            raise ValueError(msg)
        if patient not in self._controllers:
            self._controllers[patient] = Controller.from_dicom_directory(
                PATIENT_DIRS[patient]
            )
        return self._controllers[patient]

    def get_patient_info(self, patient: str) -> dict[str, Any]:
        return self._controller(patient).get_patient_info()

    def list_atoms(self, patient: str) -> list[str]:
        return self._controller(patient).list_atoms()

    def call_atom(self, patient: str, name: str, series_id: str) -> Any:
        self.atom_calls.append(
            {"patient": patient, "name": name, "series_id": series_id}
        )
        return self._controller(patient).call_atom(name, series_id)

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
