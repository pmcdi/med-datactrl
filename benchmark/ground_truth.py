"""Compute and verify ground-truth answers for clinical benchmark questions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datactrl import Controller

BENCHMARK_DIR = Path(__file__).parent
QUESTIONS_PATH = BENCHMARK_DIR / "clinical_questions.json"
DATA_ROOT = BENCHMARK_DIR.parent / "data"

PATIENT_DIRS = {
    "KiTS-00008": DATA_ROOT / "KiTS-00008",
    "Adrenal_Ki67_Seg_002": DATA_ROOT / "Adrenal_Ki67_Seg_002",
    "STS_002": DATA_ROOT / "STS_002",
}


def _load_controller(patient: str) -> Controller:
    return Controller.from_patient_directory(PATIENT_DIRS[patient])


def _collect_nodes(controller: Controller) -> list[dict[str, Any]]:
    patients = controller.get_dicom_patient_info()
    trees = patients[next(iter(patients))]

    nodes: list[dict[str, Any]] = []

    def walk(node: dict[str, Any]) -> None:
        nodes.append(node)
        for child in node["children"]:
            walk(child)

    for root in trees:
        walk(root)
    return nodes


def _ct_with_seg_child(trees: list[dict[str, Any]]) -> dict[str, Any] | None:
    for root in trees:
        if root["modality"] == "CT" and any(
            child["modality"] == "SEG" for child in root["children"]
        ):
            return root
    return None


def _seg_child(ct_root: dict[str, Any]) -> dict[str, Any] | None:
    for child in ct_root["children"]:
        if child["modality"] == "SEG":
            return child
    return None


def _find_rtstruct_by_name(
    controller: Controller,
    nodes: list[dict[str, Any]],
    name_substring: str,
) -> dict[str, Any] | None:
    for node in nodes:
        if node["modality"] != "RTSTRUCT":
            continue
        name = controller.call_dicom_atom(
            "get_rtstruct_StructureSetName",
            node["series_id"],
        )
        if name_substring in name:
            return node
    return None


def _modality_for_series(
    controller: Controller,
    nodes: list[dict[str, Any]],
    series_id: str,
) -> str:
    for node in nodes:
        if node["series_id"] == series_id:
            return node["modality"]
    return controller.call_dicom_atom("get_Modality", series_id)


def compute_answer(question_id: str) -> dict[str, Any]:
    """Compute the expected answer for a benchmark question."""

    questions = json.loads(QUESTIONS_PATH.read_text())
    question = next(q for q in questions if q["id"] == question_id)
    patient = question["patient"]
    controller = _load_controller(patient)
    patients = controller.get_dicom_patient_info()
    trees = patients[next(iter(patients))]
    nodes = _collect_nodes(controller)

    if question_id == "K-R1":
        ct = trees[0]
        sid = ct["series_id"]
        return {
            "body_part": controller.call_dicom_atom("get_BodyPartExamined", sid),
            "contrast_agent": controller.call_dicom_atom("get_ct_ContrastBolusAgent", sid),
            "contrast_volume": controller.call_dicom_atom("get_ct_ContrastBolusVolume", sid),
            "protocol_name": controller.call_dicom_atom("get_ProtocolName", sid),
        }

    if question_id == "K-R2":
        seg = trees[0]["children"][0]
        sid = seg["series_id"]
        return {
            "roi_descriptions": controller.call_dicom_atom("get_seg_ROIDescriptions", sid),
            "roi_names": controller.call_dicom_atom("get_seg_ROINames", sid),
            "segmentation_type": controller.call_dicom_atom("get_seg_SegmentationType", sid),
        }

    if question_id == "K-R3":
        ct = trees[0]
        seg = ct["children"][0]
        ct_sid = ct["series_id"]
        seg_sid = seg["series_id"]
        spacing = controller.call_dicom_atom("get_seg_SegSpacing", seg_sid)
        return {
            "referenced_series_matches_ct": controller.call_dicom_atom(
                "get_seg_ReferencedSeriesUID",
                seg_sid,
            )
            == ct_sid,
            "ct_slice_thickness": controller.call_dicom_atom("get_SliceThickness", ct_sid),
            "seg_spacing_z": spacing[2] if spacing else None,
        }

    if question_id == "K-R4":
        ct = trees[0]
        sid = ct["series_id"]
        return {
            "body_part": controller.call_dicom_atom("get_BodyPartExamined", sid),
            "kvp": controller.call_dicom_atom("get_ct_KVP", sid),
            "slice_thickness": controller.call_dicom_atom("get_SliceThickness", sid),
            "convolution_kernel": controller.call_dicom_atom("get_ct_ConvolutionKernel", sid),
            "patient_position": controller.call_dicom_atom("get_PatientPosition", sid),
        }

    if question_id == "A-R1":
        ct = _ct_with_seg_child(trees)
        seg = _seg_child(ct) if ct else None
        ref_sid = controller.call_dicom_atom(
            "get_seg_ReferencedSeriesUID",
            seg["series_id"],
        )
        protocol = controller.call_dicom_atom("get_ProtocolName", ref_sid)
        return {
            "segmentation_source_protocol": protocol,
            "phase": "contrast-enhanced"
            if "SURE START" in protocol
            else "non-contrast",
        }

    if question_id == "A-R2":
        ct_roots = [root for root in trees if root["modality"] == "CT"]
        protocols = [
            controller.call_dicom_atom("get_ProtocolName", root["series_id"])
            for root in ct_roots
        ]
        contrast = next(p for p in protocols if "SURE START" in p)
        non_contrast = next(p for p in protocols if "NON CON" in p)
        return {
            "ct_phase_count": len(ct_roots),
            "contrast_protocol": contrast,
            "non_contrast_protocol": non_contrast,
        }

    if question_id == "A-R3":
        ct = _ct_with_seg_child(trees)
        seg = _seg_child(ct) if ct else None
        ref_sid = controller.call_dicom_atom(
            "get_seg_ReferencedSeriesUID",
            seg["series_id"],
        )
        return {
            "roi_names": controller.call_dicom_atom("get_seg_ROINames", seg["series_id"]),
            "roi_descriptions": controller.call_dicom_atom(
                "get_seg_ROIDescriptions",
                seg["series_id"],
            ),
            "body_part": controller.call_dicom_atom("get_BodyPartExamined", ref_sid),
            "slice_thickness": controller.call_dicom_atom("get_SliceThickness", ref_sid),
        }

    if question_id == "A-R4":
        ct_roots = [root for root in trees if root["modality"] == "CT"]
        kvp_values = [
            controller.call_dicom_atom("get_ct_KVP", root["series_id"]) for root in ct_roots
        ]
        exposure_values = [
            controller.call_dicom_atom("get_ct_Exposure", root["series_id"])
            for root in ct_roots
        ]
        return {
            "kvp_values": kvp_values,
            "exposure_values": exposure_values,
            "lowest_exposure": min(exposure_values, key=float),
        }

    if question_id == "S-R1":
        rt_pet = _find_rtstruct_by_name(controller, nodes, "RTstruct_PET")
        rt_ct = _find_rtstruct_by_name(controller, nodes, "RTstruct_CT")
        roi_names = controller.call_dicom_atom(
            "get_rtstruct_ROINames",
            rt_pet["series_id"],
        )
        return {
            "pet_structure_set_name": controller.call_dicom_atom(
                "get_rtstruct_StructureSetName",
                rt_pet["series_id"],
            ),
            "ct_structure_set_name": controller.call_dicom_atom(
                "get_rtstruct_StructureSetName",
                rt_ct["series_id"],
            ),
            "roi_names": roi_names,
            "num_rois": controller.call_dicom_atom(
                "get_rtstruct_NumROIs",
                rt_pet["series_id"],
            ),
        }

    if question_id == "S-R2":
        rt_pet = _find_rtstruct_by_name(controller, nodes, "RTstruct_PET")
        ref_sid = controller.call_dicom_atom(
            "get_rtstruct_ReferencedSeriesUID",
            rt_pet["series_id"],
        )
        return {
            "structure_set_name": controller.call_dicom_atom(
                "get_rtstruct_StructureSetName",
                rt_pet["series_id"],
            ),
            "referenced_modality": _modality_for_series(controller, nodes, ref_sid),
        }

    if question_id == "S-R3":
        rt_ct = _find_rtstruct_by_name(controller, nodes, "RTstruct_CT")
        ref_sid = controller.call_dicom_atom(
            "get_rtstruct_ReferencedSeriesUID",
            rt_ct["series_id"],
        )
        return {
            "structure_set_name": controller.call_dicom_atom(
                "get_rtstruct_StructureSetName",
                rt_ct["series_id"],
            ),
            "kvp": controller.call_dicom_atom("get_ct_KVP", ref_sid),
            "slice_thickness": controller.call_dicom_atom("get_SliceThickness", ref_sid),
        }

    if question_id == "S-R4":
        ct = next(node for node in nodes if node["modality"] == "CT")
        pt = next(node for node in nodes if node["modality"] == "PT")
        mr_nodes = [node for node in nodes if node["modality"] == "MR"]
        return {
            "ct_slice_thickness": controller.call_dicom_atom(
                "get_SliceThickness",
                ct["series_id"],
            ),
            "pt_slice_thickness": controller.call_dicom_atom(
                "get_SliceThickness",
                pt["series_id"],
            ),
            "mr_slice_thicknesses": [
                controller.call_dicom_atom("get_SliceThickness", node["series_id"])
                for node in mr_nodes
            ],
        }

    if question_id == "S-R5":
        mr_nodes = [node for node in nodes if node["modality"] == "MR"]
        thickest = max(
            mr_nodes,
            key=lambda node: float(
                controller.call_dicom_atom("get_SliceThickness", node["series_id"])
            ),
        )
        thickest_sid = thickest["series_id"]
        max_thickness = controller.call_dicom_atom("get_SliceThickness", thickest_sid)
        for node in nodes:
            if node["modality"] != "RTSTRUCT":
                continue
            ref_sid = controller.call_dicom_atom(
                "get_rtstruct_ReferencedSeriesUID",
                node["series_id"],
            )
            if ref_sid == thickest_sid:
                return {
                    "max_mr_slice_thickness": max_thickness,
                    "structure_set_name": controller.call_dicom_atom(
                        "get_rtstruct_StructureSetName",
                        node["series_id"],
                    ),
                }
        raise ValueError("No RTSTRUCT references thickest MR series")

    if question_id == "S-R6":
        pt = next(node for node in nodes if node["modality"] == "PT")
        sid = pt["series_id"]
        return {
            "body_part": controller.call_dicom_atom("get_BodyPartExamined", sid),
            "protocol_name": controller.call_dicom_atom("get_ProtocolName", sid),
            "decay_correction": controller.call_dicom_atom("get_pt_DecayCorrection", sid),
        }

    if question_id == "S-R7":
        rt_t1 = _find_rtstruct_by_name(controller, nodes, "RTstruct_T1")
        ref_sid = controller.call_dicom_atom(
            "get_rtstruct_ReferencedSeriesUID",
            rt_t1["series_id"],
        )
        return {
            "structure_set_name": controller.call_dicom_atom(
                "get_rtstruct_StructureSetName",
                rt_t1["series_id"],
            ),
            "referenced_modality": _modality_for_series(controller, nodes, ref_sid),
            "roi_names": controller.call_dicom_atom(
                "get_rtstruct_ROINames",
                rt_t1["series_id"],
            ),
        }

    raise ValueError(f"Unknown question id: {question_id}")


def verify_questions() -> list[dict[str, Any]]:
    """Verify stored expected answers against computed ground truth."""

    questions = json.loads(QUESTIONS_PATH.read_text())
    results: list[dict[str, Any]] = []

    for question in questions:
        computed = compute_answer(question["id"])
        expected = question["expected"]
        passed = computed == expected
        results.append(
            {
                "id": question["id"],
                "passed": passed,
                "expected": expected,
                "computed": computed,
            }
        )

    return results


if __name__ == "__main__":
    results = verify_questions()
    passed = sum(1 for result in results if result["passed"])
    print(f"Verified {passed}/{len(results)} questions")

    for result in results:
        status = "PASS" if result["passed"] else "FAIL"
        print(f"  [{status}] {result['id']}")
        if not result["passed"]:
            print(f"    expected: {result['expected']}")
            print(f"    computed: {result['computed']}")
