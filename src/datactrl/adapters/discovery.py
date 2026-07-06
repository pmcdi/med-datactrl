"""File discovery helpers for patient directories."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger("datactrl")

DICOM_EXTENSION = ".dcm"
SLIDE_EXTENSIONS = {".svs", ".tif", ".tiff"}
NOTE_EXTENSIONS = {".txt", ".json", ".md", ".pdf"}


def iter_patient_files(patient_dir: Path) -> list[Path]:
    """Return visible files under a patient directory.

    Hidden paths (for example ``.imgtools/`` cache files) are excluded.
    """

    return [
        path
        for path in patient_dir.rglob("*")
        if path.is_file() and not _is_hidden_path(path, patient_dir)
    ]


def discover_modalities(patient_dir: Path) -> set[str]:
    """Infer available modalities from file extensions under ``patient_dir``."""

    if not patient_dir.is_dir():
        raise ValueError(f"Patient directory does not exist: {patient_dir}")

    modalities: set[str] = set()
    for path in iter_patient_files(patient_dir):
        modality = _modality_for_suffix(path.suffix.lower())
        if modality is not None:
            modalities.add(modality)

    logger.info(
        "Discovered modalities from file paths: %s",
        ", ".join(sorted(modalities)) if modalities else "<none>",
    )
    return modalities


def _modality_for_suffix(suffix: str) -> str | None:
    if suffix == DICOM_EXTENSION:
        return "dicom"
    if suffix in SLIDE_EXTENSIONS:
        return "pathology"
    if suffix in NOTE_EXTENSIONS:
        return "clinical"
    return None


def _is_hidden_path(path: Path, patient_dir: Path) -> bool:
    relative = path.relative_to(patient_dir)
    return any(part.startswith(".") for part in relative.parts)
