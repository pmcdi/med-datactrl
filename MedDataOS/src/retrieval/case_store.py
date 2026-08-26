"""Lazy Case sessions keyed by patient_id."""

from __future__ import annotations

import sys
import threading
from pathlib import Path

_DATACTRL_SRC = Path(__file__).resolve().parents[3] / "src"
if str(_DATACTRL_SRC) not in sys.path:
    sys.path.insert(0, str(_DATACTRL_SRC))

from datactrl import Case  # noqa: E402
from datactrl.errors import DataCtrlError  # noqa: E402

from src.patients.catalog import case_root, has_case

_LOCK = threading.Lock()
_CASES: dict[str, Case] = {}


def get_case(patient_id: str) -> Case:
    """Return a cached Case for ``patient_id``, opening it on first use."""
    if not has_case(patient_id):
        raise KeyError(f"Patient {patient_id!r} is not a datactrl case")

    with _LOCK:
        cached = _CASES.get(patient_id)
        if cached is not None:
            return cached
        case = Case.open(case_root(patient_id))
        _CASES[patient_id] = case
        return case


def reset_cases() -> None:
    """Drop cached cases (tests)."""
    with _LOCK:
        _CASES.clear()


def warmup(patient_id: str) -> None:
    """Open a case in the background so the first chat is not a cold crawl."""
    try:
        get_case(patient_id)
    except (KeyError, FileNotFoundError, DataCtrlError, OSError) as exc:
        print(f"Case warmup failed for {patient_id}: {exc}")
