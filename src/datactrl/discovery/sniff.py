"""Content sniffing helpers for classification."""

from __future__ import annotations

import json
from pathlib import Path

from datactrl.limits import DEFAULT_JSON_SNIFF_BYTES

DICOM_MAGIC = b"DICM"


def looks_like_dicom(path: Path) -> bool:
    """Return True when the file has a DICOM Part 10 preamble or parses as DICOM."""
    try:
        with path.open("rb") as handle:
            header = handle.read(132)
    except OSError:
        return False

    if len(header) >= 132 and header[128:132] == DICOM_MAGIC:
        return True

    # Extensionless DICOM: try a lightweight pydicom read when available.
    try:
        import pydicom

        pydicom.dcmread(str(path), stop_before_pixels=True, force=False)
        return True
    except Exception:
        return False


def json_is_tabular(path: Path) -> bool:
    """Heuristic: array-of-objects or {data: [...objects...]} counts as structured."""
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")[
            :DEFAULT_JSON_SNIFF_BYTES
        ]
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError):
        return False

    if isinstance(payload, list):
        return bool(payload) and all(isinstance(row, dict) for row in payload[:20])

    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, list) and data:
            return all(isinstance(row, dict) for row in data[:20])
    return False
