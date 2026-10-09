from google.genai import types

from src.patients.catalog import has_case
from src.retrieval.tools import retrieval_declarations


def _make_tool(name: str, description: str, extra_properties: dict | None = None) -> dict:
    properties = {
        "patient_id": {
            "type": "string",
            "description": "Patient identifier (e.g., 'P0001')",
        },
        "query": {
            "type": "string",
            "description": "Clinical question or focus area for the analysis",
        },
    }
    if extra_properties:
        properties.update(extra_properties)
    return {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": ["patient_id", "query"],
        },
    }


TOOL_DECLARATIONS = [
    _make_tool("analyze_clinical_notes", "Answer a question from the patient's clinical notes with source-grounded evidence. Timeline-aware (resolves relative dates), negation-aware, and reports documented discrepancies between notes (e.g. imaging vs pathology) with the more reliable source."),
    _make_tool("analyze_chest_xray", "Analyze chest X-ray image for thoracic pathologies including cardiac silhouette, lung fields, mediastinum, and bony structures."),
    _make_tool("analyze_ecg", "Analyze 12-lead ECG for rhythm, rate, intervals, axis, ST/T-wave changes, and conduction abnormalities."),
    _make_tool("analyze_echo", "Analyze echocardiogram video for LV function, wall motion, valvular assessment, and pericardial evaluation."),
    _make_tool("analyze_heart_sounds", "Analyze heart auscultation audio for S1/S2 quality, murmurs, gallops, rubs, and extra heart sounds."),
    _make_tool("analyze_lab_results", "Analyze laboratory results for abnormal values, clinically significant trends, and critical findings."),
    _make_tool("analyze_medication", "Analyze medication regimen for drug list, interactions, contraindications, and therapeutic adequacy."),
]

IMAGING_TOOL = _make_tool(
    "analyze_imaging",
    (
        "Interpret one DICOM preview. Pass resource_id and suggested_slice_index "
        "from dicom_select. RTSTRUCT previews overlay contours on the referenced CT. "
        "Call once per question unless you need a different series or slice. "
        "Do not guess a series."
    ),
    extra_properties={
        "resource_id": {
            "type": "string",
            "description": "datactrl resource id from dicom_select (e.g. dicom:<SeriesInstanceUID>)",
        },
        "slice_index": {
            "type": "integer",
            "description": "Preview slice from dicom_select.suggested_slice_index",
        },
    },
)

tools = types.Tool(function_declarations=TOOL_DECLARATIONS)


def tools_for_patient(patient_id: str | None) -> types.Tool:
    """Stub cardiac tools for demo patients; retrieval + notes/imaging for datactrl cases."""
    if has_case(patient_id):
        notes = next(item for item in TOOL_DECLARATIONS if item["name"] == "analyze_clinical_notes")
        return types.Tool(
            function_declarations=[
                *retrieval_declarations(),
                notes,
                IMAGING_TOOL,
            ]
        )
    return tools
