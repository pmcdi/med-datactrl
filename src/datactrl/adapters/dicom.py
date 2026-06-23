from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from imgtools.dicom.crawl import Crawler
from imgtools.dicom.interlacer import Interlacer
from imgtools.utils.interlacer_utils import SeriesNode

from datactrl.registry import register_atoms


class DicomAdapter:
    """Crawl, index, and query DICOM series for one patient directory."""

    def __init__(self, crawler: Crawler):
        self.crawler = crawler
        self.interlacer = Interlacer(crawler.index)
        self.atoms = register_atoms(self)

    @classmethod
    def from_directory(cls, directory: str | Path) -> DicomAdapter:
        """Crawl ``directory`` and return an adapter over the indexed series."""

        crawler = Crawler(Path(directory))
        crawler.crawl()
        return cls(crawler)

    def get_dicom_patient_info(self) -> dict[str, list[dict[str, Any]]]:
        """Return series trees grouped by DICOM patient ID."""

        patient_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for root in self.interlacer.root_nodes:
            patient_groups[root.PatientID].append(_series_node_to_dict(root))

        return dict(sorted(patient_groups.items()))

    def get_dicom_series_metadata(self, series_id: str) -> dict[str, Any]:
        """Return the indexed metadata dict for one series."""

        return self.crawler.get_series_info(series_id)

    def list_dicom_atoms(self) -> list[str]:
        """Return sorted names of registered DICOM atom getters."""

        return sorted(self.atoms)

    def call_dicom_atom(self, name: str, series_id: str) -> Any:
        """Invoke one registered DICOM atom getter."""

        try:
            getter = self.atoms[name]
        except KeyError:
            raise ValueError(f"Unknown DICOM atom: {name!r}") from None
        return getter(self, series_id)


def _series_node_to_dict(node: SeriesNode) -> dict[str, Any]:
    return {
        "modality": node.Modality,
        "series_id": node.SeriesInstanceUID,
        "study_id": node.StudyInstanceUID,
        "children": [_series_node_to_dict(child) for child in node.children],
    }
