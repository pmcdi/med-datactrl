"""Raw pydicom tool layer for benchmark comparison (no controller)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pydicom
from pydicom.dataset import Dataset

from benchmark.agent_tools import PATIENT_DIRS, _patient_parameter, _tool

MAX_FILES_PER_QUESTION = 100


def _resolve_path(patient: str, path: str) -> Path:
    root = PATIENT_DIRS[patient].resolve()
    file_path = (root / path).resolve()
    if not str(file_path).startswith(str(root)):
        msg = f"Path must stay inside patient directory: {path!r}"
        raise ValueError(msg)
    if not file_path.is_file():
        msg = f"Not a file: {path!r}"
        raise ValueError(msg)
    return file_path


def _element_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, pydicom.sequence.Sequence):
        return [_dataset_to_dict(item) for item in value[:10]]
    if isinstance(value, Dataset):
        return _dataset_to_dict(value)
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value[:20]]
    if isinstance(value, bytes):
        return f"<bytes len={len(value)}>"
    return str(value)


def _dataset_to_dict(ds: Dataset) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for elem in ds:
        key = elem.keyword or str(elem.tag)
        if not key or key == "PixelData":
            continue
        result[key] = _element_value(elem.value)
    return result


TOOL_DEFINITIONS: list[dict[str, Any]] = [
    _tool(
        "list_dir",
        "List series folders and files at the top level of a patient directory.",
        {"patient": _patient_parameter()},
        ["patient"],
    ),
    _tool(
        "find_series_samples",
        "Return one sample DICOM file path per series folder, optionally filtered by modality.",
        {
            "patient": _patient_parameter(),
            "modality": {"type": "string"},
        },
        ["patient"],
    ),
    _tool(
        "read_dicom",
        "Read DICOM metadata tags from one file. Path is relative to the patient directory.",
        {
            "patient": _patient_parameter(),
            "path": {"type": "string"},
        },
        ["patient", "path"],
    ),
]


class PydicomToolSession:
    """Filesystem + pydicom access for one benchmark question."""

    def __init__(self) -> None:
        self.files_opened: list[str] = []
        self.tool_calls: list[dict[str, Any]] = []
        self._tool_handlers = {
            "list_dir": self.list_dir,
            "find_series_samples": self.find_series_samples,
            "read_dicom": self.read_dicom,
        }

    def _patient_root(self, patient: str) -> Path:
        if patient not in PATIENT_DIRS:
            raise ValueError(f"Unknown patient {patient!r}")
        return PATIENT_DIRS[patient]

    def _check_file_limit(self) -> None:
        if len(self.files_opened) >= MAX_FILES_PER_QUESTION:
            msg = f"File read limit reached ({MAX_FILES_PER_QUESTION} per question)"
            raise ValueError(msg)

    def list_dir(self, patient: str) -> dict[str, Any]:
        root = self._patient_root(patient)
        entries = []
        for path in sorted(root.iterdir()):
            if path.name.startswith("."):
                continue
            entry: dict[str, Any] = {
                "name": path.name,
                "type": "dir" if path.is_dir() else "file",
            }
            if path.is_dir():
                dcm_files = list(path.glob("*.dcm"))
                entry["dicom_file_count"] = len(dcm_files)
            entries.append(entry)
        return {"patient": patient, "root": str(root), "entries": entries}

    def find_series_samples(
        self,
        patient: str,
        modality: str | None = None,
    ) -> list[dict[str, Any]]:
        root = self._patient_root(patient)
        samples: list[dict[str, Any]] = []

        for series_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            dcm_files = sorted(series_dir.glob("*.dcm"))
            if not dcm_files:
                continue

            sample_path = dcm_files[0].relative_to(root).as_posix()
            meta = self.read_dicom(patient, sample_path)
            series_modality = meta.get("Modality", "")

            if modality and series_modality.upper() != modality.upper():
                continue

            samples.append(
                {
                    "folder": series_dir.name,
                    "sample_path": sample_path,
                    "modality": series_modality,
                    "series_instance_uid": meta.get("SeriesInstanceUID", ""),
                    "study_instance_uid": meta.get("StudyInstanceUID", ""),
                }
            )

        return samples

    def read_dicom(self, patient: str, path: str) -> dict[str, Any]:
        self._check_file_limit()
        file_path = _resolve_path(patient, path)
        self.files_opened.append(path)

        ds = pydicom.dcmread(file_path, stop_before_pixels=True)
        return _dataset_to_dict(ds)

    def dispatch(self, name: str, arguments: str | dict[str, Any]) -> str:
        args = json.loads(arguments) if isinstance(arguments, str) else arguments
        self.tool_calls.append({"name": name, "arguments": args})
        handler = self._tool_handlers.get(name)
        if handler is None:
            raise ValueError(f"Unknown tool: {name!r}")
        result = handler(**args)
        return json.dumps(result, default=str)

    def reset_trace(self) -> None:
        self.files_opened = []
        self.tool_calls = []
