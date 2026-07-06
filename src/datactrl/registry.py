"""Register deterministic metadata getters ("atoms") for adapters."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any, Callable

from imgtools.dicom.dicom_metadata import get_extractor
from imgtools.dicom.dicom_metadata.extractor_base import (
    ModalityMetadataExtractor,
)

if TYPE_CHECKING:
    from datactrl.adapters.dicom import DicomAdapter
    from datactrl.adapters.pathology import PathologyAdapter

AtomFn = Callable[["DicomAdapter", str], Any]
SlideAtomFn = Callable[["PathologyAdapter", str], Any]
EMPTY_VALUE = ""


def register_atoms(adapter: DicomAdapter) -> dict[str, AtomFn]:
    """Register DICOM atom getters for metadata tags present in ``adapter``."""

    present_keys, keys_by_modality = _present_dicom_keys(adapter)
    atoms: dict[str, AtomFn] = {}

    for tag in sorted(ModalityMetadataExtractor.base_tags & present_keys):
        atoms[f"get_{tag}"] = _make_base_getter(tag)

    for modality in sorted(keys_by_modality):
        extractor = get_extractor(modality)
        modality_keys = extractor.modality_tags | set(extractor.computed_fields)
        for tag in sorted(modality_keys & keys_by_modality[modality]):
            atoms[f"get_{modality.lower()}_{tag}"] = _make_modality_getter(
                modality,
                tag,
            )

    atoms["get_dicom_image"] = _get_dicom_image
    atoms["get_dicom_overlay_image"] = _get_dicom_overlay_image

    return atoms


def register_slide_atoms(adapter: PathologyAdapter) -> dict[str, SlideAtomFn]:
    """Register pathology atom getters for OpenSlide properties in ``adapter``."""

    present_keys: set[str] = set()
    for slide_id in adapter.slides:
        present_keys.update(adapter.get_slide_metadata(slide_id))

    atoms = {
        f"get_pathology_{_normalize_property_key(key)}": _make_slide_getter(key)
        for key in sorted(present_keys)
    }
    atoms["get_pathology_image"] = _get_pathology_image
    atoms["get_pathology_overlay_image"] = _get_pathology_overlay_image
    return atoms


def _present_dicom_keys(
    adapter: DicomAdapter,
) -> tuple[set[str], dict[str, set[str]]]:
    present_keys: set[str] = set()
    keys_by_modality: dict[str, set[str]] = defaultdict(set)

    for series in adapter.crawler.crawl_db_raw.values():
        for subseries in series.values():
            modality = str(subseries.get("Modality", EMPTY_VALUE)).upper()
            for key in subseries:
                present_keys.add(key)
                if modality:
                    keys_by_modality[modality].add(key)

    return present_keys, keys_by_modality


def _make_base_getter(tag: str) -> AtomFn:
    def getter(adapter: DicomAdapter, series_id: str) -> Any:
        return adapter.crawler.get_series_info(series_id).get(tag, EMPTY_VALUE)

    getter.__name__ = f"get_{tag}"
    return getter


def _make_modality_getter(modality: str, tag: str) -> AtomFn:
    def getter(adapter: DicomAdapter, series_id: str) -> Any:
        meta = adapter.crawler.get_series_info(series_id)
        actual = meta.get("Modality", EMPTY_VALUE)
        if actual.upper() != modality.upper():
            msg = f"Series {series_id!r} is {actual!r}, expected {modality!r}"
            raise ValueError(msg)
        return meta.get(tag, EMPTY_VALUE)

    getter.__name__ = f"get_{modality.lower()}_{tag}"
    return getter


def _make_slide_getter(property_key: str) -> SlideAtomFn:
    def getter(adapter: PathologyAdapter, slide_id: str) -> Any:
        return adapter.get_slide_metadata(slide_id).get(property_key, EMPTY_VALUE)

    getter.__name__ = f"get_pathology_{_normalize_property_key(property_key)}"
    return getter


def _normalize_property_key(key: str) -> str:
    return key.replace(".", "_").replace(" ", "_")


def _get_dicom_image(adapter: DicomAdapter, series_id: str) -> Any:
    return adapter.get_dicom_image(series_id)


def _get_dicom_overlay_image(adapter: DicomAdapter, series_id: str) -> Any:
    return adapter.get_dicom_overlay_image(series_id)


def _get_pathology_image(adapter: PathologyAdapter, slide_id: str) -> Any:
    return adapter.get_pathology_image(slide_id)


def _get_pathology_overlay_image(adapter: PathologyAdapter, slide_id: str) -> Any:
    return adapter.get_pathology_overlay_image(slide_id)
