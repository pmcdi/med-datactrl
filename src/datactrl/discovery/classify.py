"""Classify files into modalities without requiring a case layout."""

from __future__ import annotations

from pathlib import Path

from datactrl.discovery.sniff import json_is_tabular, looks_like_dicom
from datactrl.types import Modality

STRUCTURED_EXTENSIONS = {".csv", ".tsv", ".parquet"}
NOTE_EXTENSIONS = {".txt", ".md", ".pdf"}
JSON_EXTENSIONS = {".json", ".jsonl"}
# Skip DICOM magic/pydicom sniff for known non-DICOM binaries (e.g. WSI).
SKIP_DICOM_SNIFF_EXTENSIONS = {
    ".svs",
    ".tif",
    ".tiff",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".mp4",
    ".wav",
    ".svg",
    ".bin",
    ".npy",
    ".npz",
    ".h5",
    ".hdf5",
    ".nii",
    ".zip",
    ".gz",
}


def classify_file(path: Path) -> Modality | None:
    """Return a modality for ``path``, or None if unknown."""
    suffix = path.suffix.lower()

    if suffix in STRUCTURED_EXTENSIONS:
        return Modality.STRUCTURED

    if suffix in NOTE_EXTENSIONS:
        return Modality.NOTE

    if suffix in JSON_EXTENSIONS:
        if suffix == ".jsonl" or json_is_tabular(path):
            return Modality.STRUCTURED
        return Modality.NOTE

    if suffix == ".dcm":
        return Modality.DICOM

    if suffix in SKIP_DICOM_SNIFF_EXTENSIONS:
        return None

    if looks_like_dicom(path):
        return Modality.DICOM

    return None


def classify_files(paths: list[Path]) -> dict[Modality | None, list[Path]]:
    """Bucket paths by modality (None = unknown)."""
    buckets: dict[Modality | None, list[Path]] = {
        Modality.DICOM: [],
        Modality.STRUCTURED: [],
        Modality.NOTE: [],
        None: [],
    }
    for path in paths:
        buckets[classify_file(path)].append(path)
    return buckets
