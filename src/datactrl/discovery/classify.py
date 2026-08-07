"""Classify files into modalities without requiring a case layout."""

from __future__ import annotations

from pathlib import Path

from datactrl.discovery.sniff import json_is_tabular, looks_like_dicom
from datactrl.types import Modality

STRUCTURED_EXTENSIONS = {".csv", ".tsv", ".parquet"}
NOTE_EXTENSIONS = {".txt", ".md", ".pdf"}
JSON_EXTENSIONS = {".json", ".jsonl"}


def classify_file(path: Path) -> Modality | None:
    """Return a modality for ``path``, or None if unknown."""
    name = path.name.lower()
    suffix = path.suffix.lower()

    if name.endswith(".nii.gz") or suffix == ".nii":
        return Modality.NIFTI

    if suffix in STRUCTURED_EXTENSIONS:
        return Modality.STRUCTURED

    if suffix in NOTE_EXTENSIONS:
        return Modality.NOTE

    if suffix in JSON_EXTENSIONS:
        if suffix == ".jsonl" or json_is_tabular(path):
            return Modality.STRUCTURED
        return Modality.NOTE

    if suffix == ".dcm" or looks_like_dicom(path):
        return Modality.DICOM

    return None


def classify_files(paths: list[Path]) -> dict[Modality | None, list[Path]]:
    """Bucket paths by modality (None = unknown)."""
    buckets: dict[Modality | None, list[Path]] = {
        Modality.DICOM: [],
        Modality.STRUCTURED: [],
        Modality.NOTE: [],
        Modality.NIFTI: [],
        None: [],
    }
    for path in paths:
        buckets[classify_file(path)].append(path)
    return buckets
