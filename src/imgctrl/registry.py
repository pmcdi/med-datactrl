from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any, Callable

from imgtools.dicom.dicom_metadata import get_extractor
from imgtools.dicom.dicom_metadata.extractor_base import (
    ModalityMetadataExtractor,
)

if TYPE_CHECKING:
    from imgctrl.controller import Controller

AtomFn = Callable[["Controller", str], Any]
EMPTY_VALUE = ""


def _get_series_meta(controller: Controller, series_id: str) -> dict[str, Any]:
    return controller.crawler.get_series_info(series_id)


def _make_base_getter(tag: str) -> AtomFn:
    """Create a generic atom getter for a base metadata tag."""

    def getter(ctrl: Controller, series_id: str) -> Any:
        return _get_series_meta(ctrl, series_id).get(tag, EMPTY_VALUE)

    getter.__name__ = f"get_{tag}"
    return getter


def _make_modality_getter(modality: str, tag: str) -> AtomFn:
    """Create a modality-scoped atom getter for a specific tag."""

    def getter(ctrl: Controller, series_id: str) -> Any:
        meta = _get_series_meta(ctrl, series_id)
        actual = meta.get("Modality", EMPTY_VALUE)
        if actual.upper() != modality.upper():
            msg = f"Series {series_id!r} is {actual!r}, expected {modality!r}"
            raise ValueError(msg)
        return meta.get(tag, EMPTY_VALUE)

    getter.__name__ = f"get_{modality.lower()}_{tag}"
    return getter


def _present_keys_by_modality(
    controller: Controller,
) -> tuple[set[str], dict[str, set[str]]]:
    """Collect all present metadata keys globally and per modality."""

    present_keys: set[str] = set()
    keys_by_modality: dict[str, set[str]] = defaultdict(set)

    for series in controller.crawler.crawl_db_raw.values():
        for subseries in series.values():
            modality = str(subseries.get("Modality", EMPTY_VALUE)).upper()
            for key in subseries:
                present_keys.add(key)
                if modality:
                    keys_by_modality[modality].add(key)

    return present_keys, keys_by_modality


def register_atoms(controller: Controller) -> dict[str, AtomFn]:
    """Register atoms for tags that are present in this controller."""

    present_keys, keys_by_modality = _present_keys_by_modality(controller)
    atoms: dict[str, AtomFn] = {}

    for tag in sorted(ModalityMetadataExtractor.base_tags & present_keys):
        atoms[f"get_{tag}"] = _make_base_getter(tag)

    for modality in sorted(keys_by_modality):
        extractor = get_extractor(modality)
        modality_keys = extractor.modality_tags | set(
            extractor.computed_fields.keys()
        )
        for tag in sorted(modality_keys & keys_by_modality[modality]):
            atoms[f"get_{modality.lower()}_{tag}"] = _make_modality_getter(
                modality,
                tag,
            )

    return atoms
