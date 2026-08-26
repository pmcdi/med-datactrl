from google.genai import types


def _make_tool(name: str, description: str) -> dict:
    return {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": {
                "patient_id": {
                    "type": "string",
                    "description": "Patient identifier (e.g., 'P0001')",
                },
                "query": {
                    "type": "string",
                    "description": "Clinical question or focus area for the analysis",
                },
            },
            "required": ["patient_id", "query"],
        },
    }


TOOL_DECLARATIONS = [
    _make_tool("analyze_clinical_notes", "Extract key findings from clinical notes including diagnoses, symptoms, history, and clinical relationships."),
    _make_tool("analyze_chest_xray", "Analyze chest X-ray image for thoracic pathologies including cardiac silhouette, lung fields, mediastinum, and bony structures."),
    _make_tool("analyze_ecg", "Analyze 12-lead ECG for rhythm, rate, intervals, axis, ST/T-wave changes, and conduction abnormalities."),
    _make_tool("analyze_echo", "Analyze echocardiogram video for LV function, wall motion, valvular assessment, and pericardial evaluation."),
    _make_tool("analyze_heart_sounds", "Analyze heart auscultation audio for S1/S2 quality, murmurs, gallops, rubs, and extra heart sounds."),
    _make_tool("analyze_lab_results", "Analyze laboratory results for abnormal values, clinically significant trends, and critical findings."),
    _make_tool("analyze_medication", "Analyze medication regimen for drug list, interactions, contraindications, and therapeutic adequacy."),
]

tools = types.Tool(function_declarations=TOOL_DECLARATIONS)
