from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import openslide

from datactrl.adapters.discovery import SLIDE_EXTENSIONS, iter_patient_files
from datactrl.registry import register_slide_atoms
from datactrl.utils.image import to_llm_image_payload


@dataclass(frozen=True)
class SlideRecord:
    slide_id: str
    path: Path
    filename: str


class PathologyAdapter:
    """Discover and query whole-slide pathology images for one patient directory."""

    def __init__(self, patient_dir: Path, slides: dict[str, SlideRecord]):
        self.patient_dir = patient_dir
        self.slides = slides
        self._handles: dict[str, openslide.OpenSlide] = {}
        self.slide_atoms = register_slide_atoms(self)

    @classmethod
    def from_patient_directory(cls, patient_dir: Path) -> PathologyAdapter | None:
        """Return an adapter when slide files are present, otherwise ``None``."""

        slides = _discover_slides(patient_dir)
        if not slides:
            return None
        return cls(patient_dir, slides)

    def list_slides(self) -> list[dict[str, Any]]:
        """Return one summary dict per indexed slide."""

        return [_slide_summary(self, slide_id) for slide_id in sorted(self.slides)]

    def get_slide_metadata(self, slide_id: str) -> dict[str, Any]:
        """Return all OpenSlide properties for one slide."""

        self._require_slide(slide_id)
        return dict(self._open_slide(slide_id).properties)

    def get_pathology_image(self, slide_id: str) -> dict[str, Any]:
        """Return a basic LLM-ready thumbnail preview for one pathology slide."""

        slide = self._open_slide(slide_id)
        image = slide.get_thumbnail((1024, 1024)).convert("RGB")
        return to_llm_image_payload(
            id=slide_id,
            kind="pathology",
            image=image,
            meta={"overlay": False},
        )

    def get_pathology_overlay_image(self, slide_id: str) -> dict[str, Any]:
        """Return a basic overlay-marked pathology preview for one slide."""

        payload = self.get_pathology_image(slide_id)
        payload["meta"] = {"overlay": True}
        return payload

    def list_slide_atoms(self) -> list[str]:
        """Return sorted names of registered slide atom getters."""

        return sorted(self.slide_atoms)

    def call_slide_atom(self, name: str, slide_id: str) -> Any:
        """Invoke one registered slide atom getter."""

        try:
            getter = self.slide_atoms[name]
        except KeyError:
            raise ValueError(f"Unknown slide atom: {name!r}") from None
        return getter(self, slide_id)

    def _open_slide(self, slide_id: str) -> openslide.OpenSlide:
        if slide_id not in self._handles:
            record = self._require_slide(slide_id)
            self._handles[slide_id] = openslide.OpenSlide(record.path)
        return self._handles[slide_id]

    def _require_slide(self, slide_id: str) -> SlideRecord:
        try:
            return self.slides[slide_id]
        except KeyError:
            raise ValueError(f"Unknown slide: {slide_id!r}") from None


def _discover_slides(patient_dir: Path) -> dict[str, SlideRecord]:
    slides: dict[str, SlideRecord] = {}
    for path in sorted(iter_patient_files(patient_dir)):
        if path.suffix.lower() not in SLIDE_EXTENSIONS:
            continue
        slides[path.stem] = SlideRecord(
            slide_id=path.stem,
            path=path,
            filename=path.name,
        )
    return slides


def _slide_summary(adapter: PathologyAdapter, slide_id: str) -> dict[str, Any]:
    record = adapter.slides[slide_id]
    slide = adapter._open_slide(slide_id)
    return {
        "slide_id": slide_id,
        "filename": record.filename,
        "path": str(record.path.relative_to(adapter.patient_dir)),
        "dimensions": slide.dimensions,
        "level_count": slide.level_count,
        "vendor": slide.properties.get("openslide.vendor", ""),
    }
