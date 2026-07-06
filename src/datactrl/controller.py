"""Orchestrate multimodal patient data access."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, TypeVar

from datactrl.adapters import (
    ClinicalAdapter, 
    DicomAdapter, 
    PathologyAdapter, 
    discover_modalities
)

logger = logging.getLogger("datactrl")
logging.basicConfig(level=logging.INFO)

AdapterT = TypeVar("AdapterT")


class Controller:
    """Entry point for reading DICOM, pathology, and clinical data for one patient."""

    def __init__(
        self,
        *,
        patient_dir: Path | None = None,
        dicom: DicomAdapter | None = None,
        pathology: PathologyAdapter | None = None,
        clinical: ClinicalAdapter | None = None,
    ):
        self.patient_dir = patient_dir
        self._dicom = dicom
        self._pathology = pathology
        self._clinical = clinical

        # Exposed for callers that still need direct DICOM crawler access.
        self.crawler = dicom.crawler if dicom is not None else None
        self.interlacer = dicom.interlacer if dicom is not None else None
        self.atoms = dicom.atoms if dicom is not None else {}

    @classmethod
    def from_patient_directory(cls, directory: str | Path) -> Controller:
        """Build a controller by scanning a patient directory for supported files."""

        patient_dir = Path(directory)
        logger.info("Initializing Controller from patient directory: %s", patient_dir)

        modalities = discover_modalities(patient_dir)
        if not modalities:
            raise ValueError(
                f"No supported files found in patient directory: {patient_dir}"
            )

        return cls(
            patient_dir=patient_dir,
            dicom=(
                DicomAdapter.from_directory(patient_dir)
                if "dicom" in modalities
                else None
            ),
            pathology=PathologyAdapter.from_patient_directory(patient_dir),
            clinical=ClinicalAdapter.from_patient_directory(patient_dir),
        )

    # DICOM

    def get_dicom_patient_info(self) -> dict[str, list[dict[str, Any]]]:
        """Return DICOM series trees grouped by patient ID."""

        return self._require(self._dicom, "DICOM").get_dicom_patient_info()

    def get_dicom_series_metadata(self, series_id: str) -> dict[str, Any]:
        """Return the full metadata dict for one DICOM series."""

        return self._require(self._dicom, "DICOM").get_dicom_series_metadata(series_id)

    def get_dicom_image(self, series_id: str) -> dict[str, Any]:
        """Return a basic LLM-ready DICOM preview image payload."""

        return self._require(self._dicom, "DICOM").get_dicom_image(series_id)

    def get_dicom_overlay_image(self, series_id: str) -> dict[str, Any]:
        """Return a basic overlay-marked DICOM preview payload."""

        return self._require(self._dicom, "DICOM").get_dicom_overlay_image(series_id)

    def list_dicom_atoms(self) -> list[str]:
        """Return registered DICOM atom getter names."""

        return self._require(self._dicom, "DICOM").list_dicom_atoms()

    def call_dicom_atom(self, name: str, series_id: str) -> Any:
        """Fetch one DICOM metadata field for a series."""

        return self._require(self._dicom, "DICOM").call_dicom_atom(name, series_id)

    # Pathology

    def get_pathology_info(self) -> list[dict[str, Any]]:
        """Return summaries for all pathology slides in the patient directory."""

        return self._require(self._pathology, "pathology").list_slides()

    def get_slide_metadata(self, slide_id: str) -> dict[str, Any]:
        """Return OpenSlide properties for one slide."""

        return self._require(self._pathology, "pathology").get_slide_metadata(slide_id)

    def get_pathology_image(self, slide_id: str) -> dict[str, Any]:
        """Return a basic LLM-ready pathology preview image payload."""

        return self._require(self._pathology, "pathology").get_pathology_image(slide_id)

    def get_pathology_overlay_image(self, slide_id: str) -> dict[str, Any]:
        """Return a basic overlay-marked pathology preview payload."""

        return self._require(self._pathology, "pathology").get_pathology_overlay_image(
            slide_id
        )

    def list_slide_atoms(self) -> list[str]:
        """Return registered pathology slide atom getter names."""

        return self._require(self._pathology, "pathology").list_slide_atoms()

    def call_slide_atom(self, name: str, slide_id: str) -> Any:
        """Fetch one pathology metadata field for a slide."""

        return self._require(self._pathology, "pathology").call_slide_atom(
            name,
            slide_id,
        )

    # Clinical

    def list_clinical_notes(self) -> list[dict[str, Any]]:
        """Return summaries for all clinical notes in the patient directory."""

        return self._require(self._clinical, "clinical").list_notes()

    def get_clinical_note(self, note_id: str) -> str | dict[str, Any]:
        """Return the full text or parsed JSON for one clinical note."""

        return self._require(self._clinical, "clinical").get_note(note_id)

    @staticmethod
    def _require(adapter: AdapterT | None, label: str) -> AdapterT:
        if adapter is None:
            raise ValueError(f"No {label} data available for this patient case")
        return adapter
