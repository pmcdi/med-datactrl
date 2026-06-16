import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from imgtools.dicom.crawl import Crawler
from imgtools.dicom.interlacer import Interlacer
from imgtools.utils.interlacer_utils import SeriesNode

from datactrl.registry import register_atoms

logger = logging.getLogger("datactrl")
logging.basicConfig(level=logging.INFO)

class Controller:
    def __init__(self, crawler: Crawler):
        self.crawler = crawler
        self.interlacer = Interlacer(crawler.index)
        self.atoms = register_atoms(self)

    @classmethod
    def from_dicom_directory(cls, directory: str | Path) -> "Controller":
        logger.info(f"Initalizing Controller from DICOM directory: {directory}")
        crawler = Crawler(Path(directory))
        crawler.crawl()

        return cls(crawler)
    
    def get_patient_info(self) -> dict[str, list[dict[str, Any]]]:
        
        def _series_node_to_dict(node: SeriesNode) -> dict[str, Any]:
            return {
                "modality": node.Modality,
                "series_id": node.SeriesInstanceUID,
                "study_id": node.StudyInstanceUID,
                "children": [
                    _series_node_to_dict(child) for child in node.children
                ],
            }

        patient_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for root in self.interlacer.root_nodes:
            patient_groups[root.PatientID].append(
                _series_node_to_dict(root)
            )

        return dict(sorted(patient_groups.items(), key=lambda item: item[0]))

    def get_series_metadata(self, series_id: str) -> dict[str, Any]:
        return self.crawler.get_series_info(series_id)

    def call_atom(self, name: str, series_id: str) -> Any:
        try:
            fn = self.atoms[name]
        except KeyError:
            raise ValueError(f"Unknown atom: {name!r}") from None
        return fn(self, series_id)

    def list_atoms(self) -> list[str]:
        return sorted(self.atoms)