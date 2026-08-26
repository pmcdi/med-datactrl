"""
FastAPI server for real-time log broadcasting.
Serves the web UI and handles WebSocket connections.
"""
import sys
import os
import json
from pathlib import Path
import signal
import socket

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import uvicorn
import asyncio
import threading
from uuid import uuid4
from typing import Optional
from datetime import datetime, timezone
import shutil

from src.utils.logging import WebSocketManager, set_websocket_manager
from src.core.orchestrator import run_pipeline

app = FastAPI(title="MedDataOS Monitor")

# Global server reference for cleanup
_server = None

# Initialize WebSocket manager
ws_manager = WebSocketManager()
set_websocket_manager(ws_manager)

# Session storage (in-memory, persists until server restart)
sessions = {}  # session_id -> {"history": [...], "processing": bool}

# Per-patient pinned citations (patient_id -> list of {pin_id, citation})
patient_pins = {}

patient_reviews = {}   # {patient_id: {"agent|web_path": {"0": True, ...}}}


def _cite(cid, agent, file, web_path, summary, knowledge_bus=None, review=None):
    """Helper to build a citation dict."""
    c = {"id": str(cid), "agent": agent, "file": file, "web_path": web_path, "summary": summary}
    if knowledge_bus:
        c["knowledge_bus"] = knowledge_bus
    if review:
        c["review"] = review
    return c


# Review state helpers
_reviewed = {"0": True, "1": True, "2": True, "3": True}
_partial = {"0": True, "1": True, "2": False, "3": False}

# Reusable P0001 citation builders (call with sequential id)
def _c_notes(cid):
    return _cite(cid, "clinical_notes", "P0001.txt", "/multimodal-data/clinical-notes/P0001.txt",
                 "58-year-old male presenting with 2\u20133 week history of exertional chest pressure. "
                 "Discomfort is pressure-like, rated 4/10, and resolves within 5 minutes of rest. "
                 "Mild dyspnea on exertion and increased fatigue are also reported. "
                 "No rest pain, palpitations, or syncope. Penicillin allergy documented.",
                 knowledge_bus={"supported_by": [
                     {"agent": "ecg", "finding": "Lateral ST changes in V4\u2013V6", "reason": "cardiac origin confirmed"},
                     {"agent": "echo", "finding": "Wall motion abnormality", "reason": "structural basis for symptoms"},
                 ], "contradicted_by": []},
                 review=_reviewed)

def _c_xray(cid):
    return _cite(cid, "chest_xray", "P0001.png", "/multimodal-data/chest-xray/P0001.png",
                 "No acute cardiopulmonary abnormality identified on this study. "
                 "Cardiac silhouette is within normal limits with no cardiomegaly. "
                 "Lung fields are clear with no consolidation, effusion, or pulmonary vascular congestion. "
                 "Mediastinal contours and bony thorax appear unremarkable.",
                 knowledge_bus={"supported_by": [
                     {"agent": "clinical_notes", "finding": "No orthopnea or decompensation signs", "reason": "consistent with clear radiograph"},
                 ], "contradicted_by": []},
                 review=_reviewed)

def _c_ecg(cid):
    return _cite(cid, "ecg", "P0001.svg", "/multimodal-data/ecg/P0001.svg",
                 "Sinus rhythm at 72 bpm with normal PR and QRS intervals. "
                 "Subtle ST-segment depression noted in lateral leads V4\u2013V6. "
                 "T-wave flattening observed in leads I and aVL. "
                 "Findings are consistent with lateral ischemia and warrant clinical correlation.",
                 knowledge_bus={"supported_by": [
                     {"agent": "echo", "finding": "Lateral wall motion abnormality", "reason": "same vascular territory"},
                     {"agent": "clinical_notes", "finding": "Exertional chest pressure", "reason": "symptom correlation"},
                 ], "contradicted_by": []},
                 review=_reviewed)

def _c_heart(cid):
    return _cite(cid, "heart_sounds", "P0001.wav", "/multimodal-data/heart-sounds/P0001.wav",
                 "S4 gallop present, indicating impaired left ventricular compliance. "
                 "Grade 2/6 systolic murmur audible at the apex, non-radiating. "
                 "S1 and S2 are otherwise normal in character. "
                 "No pericardial rub or diastolic murmur detected.",
                 knowledge_bus={"supported_by": [
                     {"agent": "echo", "finding": "LVEF 45\u201350%", "reason": "confirms ventricular dysfunction"},
                 ], "contradicted_by": []},
                 review={"0": True, "1": True, "2": True, "3": False})

def _c_echo(cid):
    return _cite(cid, "echo", "P0001.mp4", "/multimodal-data/echo/P0001.mp4",
                 "Left ventricular ejection fraction is mildly reduced at approximately 45\u201350%. "
                 "Regional wall motion abnormality identified in the lateral wall territory. "
                 "No significant valvular pathology or pericardial effusion observed. "
                 "Right ventricular size and function appear preserved.",
                 knowledge_bus={"supported_by": [
                     {"agent": "ecg", "finding": "ST depression V4\u2013V6", "reason": "same vascular territory"},
                     {"agent": "heart_sounds", "finding": "S4 gallop", "reason": "confirms impaired compliance"},
                 ], "contradicted_by": []},
                 review=_partial)

def _c_labs(cid):
    return _cite(cid, "lab_results", "P0001.png", "/multimodal-data/lab-results/P0001.png",
                 "Recurrent hyperkalemia documented with peaks of 5.9, 6.2, and 5.8 mmol/L over Oct 2025\u2013Jan 2026. "
                 "Creatinine is trending at the upper limit of normal, suggesting mild renal impairment. "
                 "Troponin levels are within the reference range on most recent draw. "
                 "CBC and hepatic function panels are unremarkable.",
                 knowledge_bus={"supported_by": [
                     {"agent": "medication", "finding": "Lisinopril 20mg (ACE inhibitor)", "reason": "explains K+ retention"},
                 ], "contradicted_by": []},
                 review=_partial)

def _c_meds(cid):
    return _cite(cid, "medication", "P0001.csv", "/multimodal-data/medications/P0001.csv",
                 "Lisinopril 20mg daily (ACE inhibitor) is the likely driver of recurrent hyperkalemia. "
                 "Cardiovascular regimen includes Amlodipine 5mg, Atorvastatin 20mg, and Aspirin 81mg. "
                 "Diabetes managed with Metformin 1000mg twice daily and Semaglutide 0.5mg weekly. "
                 "Albuterol PRN is prescribed for asthma. Penicillin allergy is documented.",
                 knowledge_bus={"supported_by": [],
                 "contradicted_by": [
                     {"agent": "lab_results", "finding": "K+ 6.2 mmol/L", "reason": "unsafe to continue Lisinopril at current dose"},
                 ]})


# Reusable P0002 citation builders
def _c2_notes(cid):
    return _cite(cid, "clinical_notes", "P0002.txt", "/multimodal-data/clinical-notes/P0002.txt",
                 "34-year-old female presenting with 5-day history of fever peaking at 39.2\u00b0C. "
                 "Productive cough with yellowish-green sputum and right-sided pleuritic chest pain. "
                 "Progressive dyspnea over 3 days, now with minimal exertion, SpO2 94% on room air. "
                 "Known history of mild intermittent asthma, well-controlled. No drug allergies.",
                 knowledge_bus={"supported_by": [
                     {"agent": "chest_xray", "finding": "RLL consolidation", "reason": "imaging confirms clinical pneumonia diagnosis"},
                     {"agent": "lab_results", "finding": "Elevated WBC and CRP", "reason": "inflammatory markers support active infection"},
                 ], "contradicted_by": []},
                 review=_reviewed)

def _c2_xray(cid):
    return _cite(cid, "chest_xray", "P0002.png", "/multimodal-data/chest-xray/P0002.png",
                 "Right lower lobe consolidation identified on PA and lateral views. "
                 "Findings consistent with community-acquired pneumonia. "
                 "No pleural effusion or pneumothorax. "
                 "Cardiac silhouette is within normal limits.",
                 knowledge_bus={"supported_by": [
                     {"agent": "clinical_notes", "finding": "RLL crackles and dullness to percussion", "reason": "exam localizes to same region"},
                     {"agent": "lab_results", "finding": "Elevated WBC", "reason": "confirms infectious etiology"},
                 ], "contradicted_by": []},
                 review=_reviewed)

def _c2_ecg(cid):
    return _cite(cid, "ecg", "P0002.svg", "/multimodal-data/ecg/P0002.svg",
                 "Sinus tachycardia at 112 bpm with normal PR and QRS intervals. "
                 "No ST-segment or T-wave abnormalities identified. "
                 "No right heart strain pattern (no S1Q3T3, no right axis deviation). "
                 "Tachycardia likely infectious or febrile in origin.",
                 knowledge_bus={"supported_by": [
                     {"agent": "clinical_notes", "finding": "Fever 39.2\u00b0C", "reason": "tachycardia explained by febrile state"},
                     {"agent": "echo", "finding": "Normal biventricular function", "reason": "rules out cardiac cause of tachycardia"},
                 ], "contradicted_by": []},
                 review=_reviewed)

def _c2_heart(cid):
    return _cite(cid, "heart_sounds", "P0002.wav", "/multimodal-data/heart-sounds/P0002.wav",
                 "Normal S1 and S2 heart sounds with regular rhythm. "
                 "No murmurs, gallops, or rubs detected. "
                 "No wheezing transmitted to the cardiac exam. "
                 "Findings consistent with a non-cardiac cause for symptoms.",
                 knowledge_bus={"supported_by": [
                     {"agent": "echo", "finding": "Normal valves, no effusion", "reason": "structural confirmation of normal auscultation"},
                 ], "contradicted_by": []},
                 review=_reviewed)

def _c2_echo(cid):
    return _cite(cid, "echo", "P0002.mp4", "/multimodal-data/echo/P0002.mp4",
                 "Normal left ventricular function with preserved ejection fraction. "
                 "No valvular abnormality or regurgitation identified. "
                 "No pericardial effusion. "
                 "Right ventricular size and function appear normal.",
                 knowledge_bus={"supported_by": [
                     {"agent": "heart_sounds", "finding": "No murmurs or gallops", "reason": "acoustic confirms structural normality"},
                     {"agent": "ecg", "finding": "No ST changes or strain", "reason": "electrically normal heart"},
                 ], "contradicted_by": []},
                 review=_reviewed)

def _c2_labs(cid):
    return _cite(cid, "lab_results", "P0002.png", "/multimodal-data/lab-results/P0002.png",
                 "Elevated WBC and CRP consistent with acute bacterial infection. "
                 "Procalcitonin mildly elevated, supporting bacterial etiology. "
                 "Renal and hepatic panels within normal limits. "
                 "CBC otherwise unremarkable with no anemia or thrombocytopenia.",
                 knowledge_bus={"supported_by": [
                     {"agent": "chest_xray", "finding": "RLL consolidation", "reason": "imaging correlates with lab infection markers"},
                     {"agent": "clinical_notes", "finding": "Fever and productive cough", "reason": "clinical presentation matches lab picture"},
                 ], "contradicted_by": []},
                 review=_partial)

def _c2_meds(cid):
    return _cite(cid, "medication", "P0002.csv", "/multimodal-data/medications/P0002.csv",
                 "Albuterol inhaler PRN for mild intermittent asthma. "
                 "Amoxicillin 875mg twice daily and Azithromycin 500mg for community-acquired pneumonia. "
                 "Acetaminophen PRN for fever management. "
                 "No known drug allergies documented.",
                 knowledge_bus={"supported_by": [
                     {"agent": "lab_results", "finding": "Normal renal and hepatic panels", "reason": "safe for current antibiotic dosing"},
                 ], "contradicted_by": []},
                 review={"0": True, "1": True, "2": True, "3": False})


# Reusable P0003 citation builders
def _c3_notes(cid):
    return _cite(cid, "clinical_notes", "P0003.txt", "/multimodal-data/clinical-notes/P0003.txt",
                 "45-year-old male presenting for annual wellness exam. No acute complaints. "
                 "Mild exercise intolerance attributed to deconditioning. GAD well-controlled on Sertraline 50mg. "
                 "Family history of hypertension (father) and colorectal cancer (maternal uncle). "
                 "Sulfa drug allergy documented. Vitals normal, exam unremarkable.",
                 knowledge_bus={"supported_by": [
                     {"agent": "ecg", "finding": "Normal sinus rhythm", "reason": "no cardiac etiology for exercise intolerance"},
                     {"agent": "lab_results", "finding": "Borderline LDL and fasting glucose", "reason": "metabolic risk factors consistent with deconditioning"},
                 ], "contradicted_by": []},
                 review=_reviewed)

def _c3_xray(cid):
    return _cite(cid, "chest_xray", "P0003.png", "/multimodal-data/chest-xray/P0003.png",
                 "No acute cardiopulmonary abnormality identified on PA and lateral views. "
                 "Cardiac silhouette is within normal limits with no cardiomegaly. "
                 "Lung fields are clear bilaterally with no consolidation, effusion, or mass. "
                 "Mediastinal contours and bony thorax appear unremarkable.",
                 knowledge_bus={"supported_by": [
                     {"agent": "clinical_notes", "finding": "Lungs clear to auscultation", "reason": "exam and imaging concordant"},
                 ], "contradicted_by": []},
                 review=_reviewed)

def _c3_ecg(cid):
    return _cite(cid, "ecg", "P0003.svg", "/multimodal-data/ecg/P0003.svg",
                 "Normal sinus rhythm at 74 bpm with normal PR, QRS, and QTc intervals. "
                 "No ST-segment or T-wave abnormalities identified. "
                 "No axis deviation or evidence of chamber enlargement. "
                 "No conduction delay or arrhythmia detected.",
                 knowledge_bus={"supported_by": [
                     {"agent": "heart_sounds", "finding": "Normal S1/S2, no murmurs", "reason": "electrically and acoustically normal"},
                 ], "contradicted_by": []},
                 review=_reviewed)

def _c3_heart(cid):
    return _cite(cid, "heart_sounds", "P0003.wav", "/multimodal-data/heart-sounds/P0003.wav",
                 "Normal S1 and S2 heart sounds with regular rate and rhythm. "
                 "No murmurs, gallops, or rubs detected. "
                 "No extra heart sounds or pericardial friction rub. "
                 "Findings consistent with a structurally and functionally normal heart.",
                 knowledge_bus={"supported_by": [
                     {"agent": "echo", "finding": "Normal LVEF and valves", "reason": "structural confirmation"},
                 ], "contradicted_by": []},
                 review=_reviewed)

def _c3_echo(cid):
    return _cite(cid, "echo", "P0003.mp4", "/multimodal-data/echo/P0003.mp4",
                 "Normal left ventricular ejection fraction estimated at 60\u201365%. "
                 "No regional wall motion abnormalities identified. "
                 "Normal valve morphology and function with no pericardial effusion. "
                 "Right ventricular size and function appear preserved.",
                 knowledge_bus={"supported_by": [
                     {"agent": "ecg", "finding": "No chamber enlargement on ECG", "reason": "electrical and structural concordance"},
                     {"agent": "heart_sounds", "finding": "No murmurs or gallops", "reason": "acoustic confirmation"},
                 ], "contradicted_by": []},
                 review=_reviewed)

def _c3_labs(cid):
    return _cite(cid, "lab_results", "P0003.png", "/multimodal-data/lab-results/P0003.png",
                 "CBC within normal limits. Fasting glucose 108 mg/dL (borderline, pre-diabetic range). "
                 "LDL cholesterol 142 mg/dL (borderline high). HDL 44 mg/dL (low normal). "
                 "Triglycerides 168 mg/dL (mildly elevated). Renal and hepatic panels normal. "
                 "TSH 2.1 mIU/L (normal). HbA1c 5.8% (upper normal).",
                 knowledge_bus={"supported_by": [],
                 "contradicted_by": [
                     {"agent": "clinical_notes", "finding": "Patient reports feeling well", "reason": "subclinical metabolic risk not yet symptomatic"},
                 ]})

def _c3_meds(cid):
    return _cite(cid, "medication", "P0003.csv", "/multimodal-data/medications/P0003.csv",
                 "Sertraline 50mg daily for GAD (uptitrated from 25mg in July 2021). "
                 "Ibuprofen 200mg PRN for occasional headaches. "
                 "Melatonin 3mg PRN at bedtime for insomnia. "
                 "Sulfa drug allergy documented (causes rash).",
                 knowledge_bus={"supported_by": [
                     {"agent": "clinical_notes", "finding": "GAD well-controlled", "reason": "current Sertraline dose is effective"},
                     {"agent": "lab_results", "finding": "Normal TSH", "reason": "no thyroid contribution to anxiety or insomnia"},
                 ], "contradicted_by": []},
                 review=_partial)


def _msg(role, content, citations=None, ts=None):
    """Build a history message. For assistant with citations, encodes as JSON."""
    if role == "assistant" and citations:
        return {"role": "assistant", "content": json.dumps({"response": content, "citations": citations}), "timestamp": ts}
    return {"role": role, "content": content, "timestamp": ts}


def _seed_demo_sessions():
    """Pre-populate demo conversations for P0001, P0002, and P0003."""

    # ── Conversation 1: Initial Clinical Assessment (3:3) ──────────
    sessions["demo-1"] = {
        "history": [
            _msg("user",
                 "James Carter is presenting today with chest discomfort. "
                 "Can you pull up his clinical notes and give me a summary of his presentation?",
                 ts="2026-03-03T09:15:00"),
            _msg("assistant",
                 "Based on the clinical notes [cite:1], James Carter presents with a 2\u20133 week history "
                 "of exertional, pressure-like chest discomfort rated 4/10, accompanied by mild dyspnea on "
                 "exertion and increased fatigue. The discomfort resolves within 5 minutes of rest, and he "
                 "denies rest pain, palpitations, or syncope.\n\n"
                 "He is currently on a multi-drug regimen [cite:2] including Lisinopril 20mg and Amlodipine 5mg "
                 "for hypertension, Metformin and Semaglutide for Type 2 Diabetes, Atorvastatin 20mg and "
                 "Aspirin 81mg for cardiovascular prevention, and Albuterol PRN for asthma. Note the documented "
                 "Penicillin allergy.\n\n"
                 "Would you like me to review his imaging and cardiac workup?",
                 [_c_notes(1), _c_meds(2)], ts="2026-03-03T09:15:32"),

            _msg("user",
                 "Yes, let\u2019s review his imaging. What does the chest X-ray show? "
                 "And pull up the ECG and echocardiogram.",
                 ts="2026-03-03T09:17:05"),
            _msg("assistant",
                 "The chest radiograph [cite:1] is clear \u2014 no acute cardiopulmonary abnormality and "
                 "normal cardiac silhouette, which rules out overt cardiomegaly or pulmonary congestion.\n\n"
                 "However, the ECG [cite:2] shows sinus rhythm with subtle but notable ST changes in the "
                 "lateral leads (V4\u2013V6). This is a concerning pattern in the context of his symptoms.\n\n"
                 "The echocardiogram [cite:3] corroborates the ECG findings: it reveals a mildly reduced left "
                 "ventricular ejection fraction with a regional wall motion abnormality specifically in the "
                 "lateral wall. The concordance between the ECG lateral ST changes and the echo lateral wall "
                 "abnormality raises suspicion for underlying coronary artery disease in that vascular territory.",
                 [_c_xray(1), _c_ecg(2), _c_echo(3)], ts="2026-03-03T09:17:48"),

            _msg("user",
                 "What about the heart sounds and lab work? Anything critical I should flag?",
                 ts="2026-03-03T09:19:20"),
            _msg("assistant",
                 "Auscultation [cite:1] reveals an S4 gallop and a grade 2/6 systolic murmur. The S4 is "
                 "consistent with impaired ventricular compliance, which aligns with the echocardiographic "
                 "findings of reduced ejection fraction and wall motion abnormality.\n\n"
                 "\u26a0\ufe0f **Potassium:** The lab results [cite:2] show potassium has exceeded 5.5 mmol/L "
                 "on three separate occasions over the past 4 months, with a peak of 6.2 mmol/L in late "
                 "December. This is a significant safety concern. Cross-referencing with his medications "
                 "[cite:3], the likely driver is Lisinopril 20mg \u2014 an ACE inhibitor known to elevate "
                 "potassium. Immediate review of ACE inhibitor dosing or switching to an alternative is "
                 "strongly recommended.\n\n"
                 "The clinical notes [cite:4] also document a Penicillin allergy, important to flag for any "
                 "future antibiotic prescriptions or procedural prophylaxis.",
                 [_c_heart(1), _c_labs(2), _c_meds(3), _c_notes(4)], ts="2026-03-03T09:20:05"),
        ],
        "processing": False,
        "uploaded_file": None,
        "patient_id": "P0001",
    }

    # ── Conversation 2: Medication Safety Review (2:2) ─────────────
    sessions["demo-2"] = {
        "history": [
            _msg("user",
                 "I need to review James Carter\u2019s medications against his recent lab work. "
                 "His potassium has been running high \u2014 can you cross-reference and flag any concerns?",
                 ts="2026-03-04T10:30:00"),
            _msg("assistant",
                 "His medication list [cite:1] includes Lisinopril 20mg, Amlodipine 5mg, Atorvastatin 20mg, "
                 "Aspirin 81mg, Metformin 1000mg twice daily, Semaglutide 0.5mg weekly, and Albuterol PRN.\n\n"
                 "Cross-referencing with labs [cite:2], potassium has been persistently elevated \u2014 5.9, "
                 "6.2, and 5.8 mmol/L across October to January. This is a clear pattern of ACE inhibitor\u2013"
                 "induced hyperkalemia, with Lisinopril as the primary suspect.\n\n"
                 "The clinical notes [cite:3] confirm a documented Penicillin allergy \u2014 important context "
                 "for any medication changes. Auscultation [cite:4] shows an S4 gallop suggesting diastolic "
                 "dysfunction, so whatever replacement we select must maintain adequate cardioprotection.",
                 [_c_meds(1), _c_labs(2), _c_notes(3), _c_heart(4)], ts="2026-03-04T10:30:45"),

            _msg("user",
                 "If we switch from Lisinopril to an ARB, what\u2019s the cardiac picture we\u2019re "
                 "working with? I want to make sure we\u2019re not compromising anything.",
                 ts="2026-03-04T10:32:10"),
            _msg("assistant",
                 "The ECG [cite:1] shows lateral ST changes in V4\u2013V6, indicating ischemia in that "
                 "territory. The echocardiogram [cite:2] confirms a lateral wall motion abnormality with "
                 "mildly reduced ejection fraction \u2014 these two findings are concordant and suggest "
                 "underlying coronary artery disease.\n\n"
                 "The chest X-ray [cite:3] is reassuringly normal \u2014 no cardiomegaly or pulmonary "
                 "congestion \u2014 so there\u2019s no evidence of overt heart failure at this stage.\n\n"
                 "Switching from Lisinopril to an ARB such as losartan would maintain RAAS blockade, which "
                 "is critical for blood pressure control, cardiac remodeling protection given the wall motion "
                 "abnormality, and renal protection with his Type 2 Diabetes. This switch should reduce the "
                 "hyperkalemia risk. Labs [cite:4] should be rechecked 1\u20132 weeks after the transition. "
                 "His diabetes medications [cite:5] (Metformin + Semaglutide) do not contribute to "
                 "hyperkalemia and can continue unchanged.",
                 [_c_ecg(1), _c_echo(2), _c_xray(3), _c_labs(4), _c_meds(5)], ts="2026-03-04T10:33:02"),
        ],
        "processing": False,
        "uploaded_file": None,
        "patient_id": "P0001",
    }

    # ── Conversation 3: Pre-Procedure Cardiac Risk (2:2) ───────────
    sessions["demo-3"] = {
        "history": [
            _msg("user",
                 "James Carter is being considered for an elective surgical procedure. "
                 "I need a cardiac risk assessment \u2014 walk me through his cardiac studies.",
                 ts="2026-03-05T14:00:00"),
            _msg("assistant",
                 "The echocardiogram [cite:1] is the most significant finding \u2014 mildly reduced LVEF "
                 "with a regional wall motion abnormality in the lateral wall, indicating structural cardiac "
                 "compromise. The ECG [cite:2] corroborates with ST changes in V4\u2013V6, suggesting "
                 "ischemia in the same distribution as the echo abnormality.\n\n"
                 "Auscultation [cite:3] reveals an S4 gallop and grade 2/6 systolic murmur, consistent "
                 "with impaired ventricular compliance. The chest X-ray [cite:4] is clear with a normal "
                 "cardiac silhouette \u2014 no pulmonary congestion, which is reassuring.\n\n"
                 "Clinical notes [cite:5] describe exertional symptoms that resolve at rest within 5 minutes, "
                 "with no rest pain or syncope. This pattern is consistent with stable demand ischemia rather "
                 "than acute coronary syndrome. Overall, this places him at intermediate cardiac risk \u2014 "
                 "the lateral wall abnormality and functional limitation warrant cardiology clearance before "
                 "proceeding with the procedure.",
                 [_c_echo(1), _c_ecg(2), _c_heart(3), _c_xray(4), _c_notes(5)], ts="2026-03-05T14:00:52"),

            _msg("user",
                 "What about his labs and medications \u2014 anything to adjust before surgery?",
                 ts="2026-03-05T14:02:30"),
            _msg("assistant",
                 "\u26a0\ufe0f **Potassium:** Recurrent hyperkalemia [cite:1] with a peak of 6.2 mmol/L. "
                 "This must be normalized before surgery \u2014 hyperkalemia significantly increases the "
                 "risk of intraoperative arrhythmias, especially under general anesthesia.\n\n"
                 "For medication adjustments [cite:2]: hold Lisinopril 20mg the morning of surgery (also "
                 "helps with potassium), discuss Aspirin 81mg hold timing with the surgical team (typically "
                 "7 days prior for major procedures), hold Metformin the day of and day after to avoid lactic "
                 "acidosis risk, and consider holding Semaglutide the week of surgery due to gastroparesis "
                 "risk affecting anesthesia. Amlodipine and Atorvastatin may continue as usual. Albuterol "
                 "PRN must be available peri-operatively for asthma management.\n\n"
                 "\u26a0\ufe0f **Allergy alert:** Penicillin allergy is documented [cite:3]. This directly "
                 "impacts surgical antibiotic prophylaxis \u2014 standard cefazolin may pose cross-reactivity "
                 "risk. Use clindamycin or vancomycin as alternatives.",
                 [_c_labs(1), _c_meds(2), _c_notes(3)], ts="2026-03-05T14:03:18"),
        ],
        "processing": False,
        "uploaded_file": None,
        "patient_id": "P0001",
    }

    # ── Conversation 4: Acute Pneumonia Workup — P0002 (3:3) ──────
    sessions["demo-4"] = {
        "history": [
            _msg("user",
                 "Maria Lopez came in with fever and a productive cough. "
                 "Can you review her clinical notes and tell me what we\u2019re dealing with?",
                 ts="2026-03-06T08:45:00"),
            _msg("assistant",
                 "Based on the clinical notes [cite:1], Maria Lopez is a 34-year-old female presenting "
                 "with a 5-day history of fever (peak 39.2\u00b0C), productive cough with yellowish-green "
                 "sputum, and right-sided pleuritic chest pain that worsens on deep inspiration. She reports "
                 "progressive dyspnea over the past 3 days, now occurring with minimal exertion.\n\n"
                 "Vitals on presentation: tachycardic at 112 bpm, tachypneic at 22/min, SpO2 94% on room "
                 "air. Lung exam reveals crackles and bronchial breath sounds over the right lower lobe with "
                 "dullness to percussion. She has a known history of mild intermittent asthma, well-controlled.\n\n"
                 "She has been started on dual antibiotic therapy [cite:2]: Amoxicillin 875mg twice daily and "
                 "Azithromycin 500mg, along with Acetaminophen PRN for fever. Her baseline medication is "
                 "Albuterol inhaler PRN for asthma. No known drug allergies.",
                 [_c2_notes(1), _c2_meds(2)], ts="2026-03-06T08:45:40"),

            _msg("user",
                 "What does her chest X-ray show? And let\u2019s check the ECG \u2014 "
                 "she\u2019s tachycardic at 112.",
                 ts="2026-03-06T08:47:15"),
            _msg("assistant",
                 "The chest X-ray [cite:1] shows right lower lobe consolidation, which is consistent "
                 "with the clinical exam findings of crackles, bronchial breath sounds, and dullness to "
                 "percussion in that region. This confirms community-acquired pneumonia.\n\n"
                 "The ECG [cite:2] shows sinus tachycardia at 112 bpm, which is expected given the fever "
                 "and active infection. Importantly, there are no ST changes, no arrhythmia, and no right "
                 "heart strain pattern \u2014 the tachycardia is likely infectious/febrile in origin rather "
                 "than cardiac.\n\n"
                 "The echocardiogram [cite:3] confirms normal left ventricular function with no valvular "
                 "abnormality and no pericardial effusion \u2014 the heart is structurally normal. This "
                 "effectively rules out any cardiac complication from the infection.",
                 [_c2_xray(1), _c2_ecg(2), _c2_echo(3)], ts="2026-03-06T08:48:03"),

            _msg("user",
                 "How about her labs and heart sounds? Anything unexpected, "
                 "or is this straightforward pneumonia?",
                 ts="2026-03-06T08:49:30"),
            _msg("assistant",
                 "Heart sounds [cite:1] are normal \u2014 no murmurs, no gallops, no rubs. This is "
                 "reassuring and consistent with a non-cardiac cause for her symptoms.\n\n"
                 "Lab results [cite:2] show elevated WBC and inflammatory markers consistent with acute "
                 "bacterial infection. CRP is elevated. Renal and hepatic panels are within normal limits, "
                 "which is important for antibiotic dosing.\n\n"
                 "The clinical notes [cite:3] document no drug allergies, so her current antibiotic regimen "
                 "(amoxicillin + azithromycin) [cite:4] is appropriate for community-acquired pneumonia.\n\n"
                 "Overall, this is a straightforward community-acquired pneumonia in an otherwise healthy "
                 "34-year-old. Key monitoring points: repeat SpO2 trending, fever curve, and ensure adequate "
                 "hydration given her reduced oral intake over the past 2 days.",
                 [_c2_heart(1), _c2_labs(2), _c2_notes(3), _c2_meds(4)], ts="2026-03-06T08:50:12"),
        ],
        "processing": False,
        "uploaded_file": None,
        "patient_id": "P0002",
    }

    # ── Demo pins for P0001 ─────────────────────────────────────
    patient_pins["P0001"] = [
        {"pin_id": str(uuid4()), "type": "citation", "citation": _c_ecg(1), "source": "demo-1",
         "created_at": datetime.now(timezone.utc).isoformat(), "query": "",
         "annotations": {"text": "", "tags": []}},
        {"pin_id": str(uuid4()), "type": "citation", "citation": _c_labs(2), "source": "demo-2",
         "created_at": datetime.now(timezone.utc).isoformat(), "query": "",
         "annotations": {"text": "", "tags": []}},
        {"pin_id": str(uuid4()), "type": "citation", "citation": _c_meds(3), "source": "demo-1",
         "created_at": datetime.now(timezone.utc).isoformat(), "query": "",
         "annotations": {"text": "", "tags": []}},
    ]

    # ── Demo pins for P0002 ─────────────────────────────────────
    patient_pins["P0002"] = [
        {"pin_id": str(uuid4()), "type": "citation", "citation": _c2_xray(1), "source": "demo-4",
         "created_at": datetime.now(timezone.utc).isoformat(), "query": "",
         "annotations": {"text": "", "tags": []}},
        {"pin_id": str(uuid4()), "type": "citation", "citation": _c2_labs(2), "source": "demo-4",
         "created_at": datetime.now(timezone.utc).isoformat(), "query": "",
         "annotations": {"text": "", "tags": []}},
        {"pin_id": str(uuid4()), "type": "citation", "citation": _c2_meds(3), "source": "demo-5",
         "created_at": datetime.now(timezone.utc).isoformat(), "query": "",
         "annotations": {"text": "", "tags": []}},
    ]

    # ── Demo pins for P0003 ─────────────────────────────────────
    patient_pins["P0003"] = [
        {"pin_id": str(uuid4()), "type": "citation", "citation": _c3_labs(1), "source": "demo-7",
         "created_at": datetime.now(timezone.utc).isoformat(), "query": "",
         "annotations": {"text": "", "tags": []}},
        {"pin_id": str(uuid4()), "type": "citation", "citation": _c3_meds(2), "source": "demo-9",
         "created_at": datetime.now(timezone.utc).isoformat(), "query": "",
         "annotations": {"text": "", "tags": []}},
        {"pin_id": str(uuid4()), "type": "citation", "citation": _c3_echo(3), "source": "demo-7",
         "created_at": datetime.now(timezone.utc).isoformat(), "query": "",
         "annotations": {"text": "", "tags": []}},
    ]

    # ── Demo review states (decoupled from pins) ────────────────
    patient_reviews["P0001"] = {
        f"{c['agent']}|{c['web_path']}": c["review"]
        for c in [_c_notes(0), _c_xray(0), _c_ecg(0), _c_heart(0), _c_echo(0), _c_labs(0), _c_meds(0)]
        if c.get("review")
    }
    patient_reviews["P0002"] = {
        f"{c['agent']}|{c['web_path']}": c["review"]
        for c in [_c2_notes(0), _c2_xray(0), _c2_ecg(0), _c2_heart(0), _c2_echo(0), _c2_labs(0), _c2_meds(0)]
        if c.get("review")
    }
    patient_reviews["P0003"] = {
        f"{c['agent']}|{c['web_path']}": c["review"]
        for c in [_c3_notes(0), _c3_xray(0), _c3_ecg(0), _c3_heart(0), _c3_echo(0), _c3_labs(0), _c3_meds(0)]
        if c.get("review")
    }

    # ── Conversation 5: Asthma + Pneumonia Management — P0002 (2:2)
    sessions["demo-5"] = {
        "history": [
            _msg("user",
                 "I\u2019m concerned about Maria Lopez\u2019s asthma complicating her pneumonia. "
                 "Can you cross-reference her respiratory data and medications?",
                 ts="2026-03-07T11:00:00"),
            _msg("assistant",
                 "Clinical notes [cite:1] confirm mild intermittent asthma, well-controlled with PRN "
                 "albuterol, and no recent exacerbations. However, the pneumonia introduces significant "
                 "risk \u2014 the infection and inflammation can trigger bronchospasm. Her SpO2 is already "
                 "at 94% on room air, leaving limited respiratory reserve.\n\n"
                 "The chest X-ray [cite:2] shows right lower lobe consolidation, which is reducing her "
                 "functional lung capacity on top of any potential asthma-related airway narrowing.\n\n"
                 "Current medications [cite:3]: Albuterol inhaler PRN (should be used more aggressively "
                 "during the acute illness), Amoxicillin + Azithromycin for the pneumonia, Acetaminophen "
                 "for fever. Notably, she was previously on Montelukast 10mg daily for asthma but it was "
                 "discontinued in January 2023.\n\n"
                 "Heart sounds [cite:4] are clear \u2014 no wheezing transmitted to the cardiac exam, "
                 "which is a good sign for current asthma control.",
                 [_c2_notes(1), _c2_xray(2), _c2_meds(3), _c2_heart(4)], ts="2026-03-07T11:00:48"),

            _msg("user",
                 "Should we restart the Montelukast? And give me the full cardiac picture "
                 "to rule out any complications.",
                 ts="2026-03-07T11:02:15"),
            _msg("assistant",
                 "Restarting Montelukast 10mg daily is reasonable as a temporary measure during the acute "
                 "illness to provide additional bronchodilation and anti-inflammatory coverage alongside "
                 "the Albuterol [cite:1].\n\n"
                 "The ECG [cite:2] shows sinus tachycardia but no right heart strain pattern \u2014 no "
                 "S1Q3T3, no right axis deviation. This effectively rules out a PE mimic, which is worth "
                 "checking given her pleuritic pain and tachycardia.\n\n"
                 "The echocardiogram [cite:3] shows normal biventricular function and no pericardial "
                 "effusion \u2014 no cardiac complications from the infection.\n\n"
                 "Labs [cite:4] support a purely infectious process with no end-organ involvement.\n\n"
                 "I\u2019d recommend continuing the Amoxicillin + Azithromycin as prescribed, increasing "
                 "Albuterol to scheduled q4\u20136h rather than just PRN, and restarting Montelukast 10mg "
                 "daily. Monitor SpO2 closely and supplement O2 if she drops below 92%. Reassess in "
                 "48\u201372 hours for clinical improvement.",
                 [_c2_meds(1), _c2_ecg(2), _c2_echo(3), _c2_labs(4)], ts="2026-03-07T11:03:05"),
        ],
        "processing": False,
        "uploaded_file": None,
        "patient_id": "P0002",
    }

    # ── Conversation 6: Treatment Response & Discharge — P0002 (2:2)
    sessions["demo-6"] = {
        "history": [
            _msg("user",
                 "Maria Lopez is on day 3 of antibiotics. Can you pull up her latest labs "
                 "and vitals \u2014 is she improving enough to consider discharge?",
                 ts="2026-03-09T09:30:00"),
            _msg("assistant",
                 "Lab results [cite:1] show WBC and CRP are trending down from admission, "
                 "indicating the infection is responding to the Amoxicillin + Azithromycin "
                 "regimen [cite:2]. Renal and hepatic panels remain normal.\n\n"
                 "Clinical notes [cite:3] document progressive improvement: temperature has been "
                 "below 37.5\u00b0C for the past 24 hours, respiratory rate has normalized to 16/min, "
                 "and SpO2 has improved to 97% on room air. She reports reduced cough frequency "
                 "and improved oral intake.\n\n"
                 "Heart sounds [cite:4] remain normal with no new findings. Heart rate has come "
                 "down from 112 to 82 bpm, consistent with defervescence.\n\n"
                 "She meets standard criteria for transition to oral-only outpatient therapy \u2014 "
                 "afebrile >24h, tolerating PO, and trending toward resolution. She should complete the "
                 "full antibiotic course at home.",
                 [_c2_labs(1), _c2_meds(2), _c2_notes(3), _c2_heart(4)], ts="2026-03-09T09:30:52"),

            _msg("user",
                 "What follow-up plan should I set for her? And any concerns about "
                 "the asthma going forward?",
                 ts="2026-03-09T09:32:30"),
            _msg("assistant",
                 "The chest X-ray [cite:1] at admission showed RLL consolidation. A follow-up "
                 "radiograph in 4\u20136 weeks is recommended to confirm complete resolution, "
                 "especially given her age \u2014 persistent opacities would warrant further workup.\n\n"
                 "Regarding asthma [cite:2]: her baseline is mild and intermittent, well-controlled "
                 "on Albuterol PRN. The pneumonia did not trigger a significant exacerbation, which "
                 "is reassuring. However, post-infectious airway hyperreactivity can persist for "
                 "several weeks.\n\n"
                 "The ECG [cite:3] and echocardiogram [cite:4] were both normal on admission, "
                 "so no cardiac follow-up is needed.\n\n"
                 "For discharge: complete the Amoxicillin + Azithromycin course (total 7 days), continue "
                 "Montelukast for 2\u20134 weeks if it was restarted then reassess, and keep Albuterol PRN "
                 "for any post-infectious cough or wheeze. Schedule a follow-up chest X-ray in 4\u20136 "
                 "weeks. Return precautions include worsening dyspnea, recurrent fever, or hemoptysis.",
                 [_c2_xray(1), _c2_notes(2), _c2_ecg(3), _c2_echo(4)], ts="2026-03-09T09:33:18"),
        ],
        "processing": False,
        "uploaded_file": None,
        "patient_id": "P0002",
    }

    # ── Conversation 7: Annual Wellness Review — P0003 (3:3) ──────
    sessions["demo-7"] = {
        "history": [
            _msg("user",
                 "Robert Chen is here for his annual wellness visit. "
                 "Pull up his clinical notes and give me the overview.",
                 ts="2026-03-10T10:00:00"),
            _msg("assistant",
                 "Based on the clinical notes [cite:1], Robert Chen is a 45-year-old male with "
                 "no acute complaints. He reports generally feeling well but acknowledges decreased "
                 "physical activity over the past year due to a demanding work schedule. He describes "
                 "mild exercise intolerance when climbing more than two flights of stairs, which he "
                 "attributes to deconditioning.\n\n"
                 "He denies chest pain, palpitations, syncope, dyspnea at rest, or edema. His GAD "
                 "is well-controlled on Sertraline 50mg daily [cite:2]. He is a non-smoker with "
                 "moderate alcohol use (2\u20133 drinks/week).\n\n"
                 "Notable family history: father with hypertension (age 68), maternal uncle with "
                 "colorectal cancer (age 62). Sulfa drug allergy is documented.\n\n"
                 "Vitals are normal: HR 74, BP 122/78, SpO2 99%, BMI 26.1.",
                 [_c3_notes(1), _c3_meds(2)], ts="2026-03-10T10:00:38"),

            _msg("user",
                 "Let\u2019s check his cardiac workup \u2014 ECG, echo, and chest X-ray. "
                 "I want to make sure the exercise intolerance is really just deconditioning.",
                 ts="2026-03-10T10:02:00"),
            _msg("assistant",
                 "The ECG [cite:1] is completely normal \u2014 sinus rhythm at 74 bpm with normal "
                 "intervals, no ST changes, no axis deviation, and no evidence of chamber "
                 "enlargement. This effectively rules out a cardiac conduction or ischemic cause "
                 "for his exercise intolerance.\n\n"
                 "The echocardiogram [cite:2] confirms a normal ejection fraction of 60\u201365% with "
                 "no wall motion abnormalities and normal valve function. The heart is structurally "
                 "and functionally normal.\n\n"
                 "The chest X-ray [cite:3] shows clear lung fields with a normal cardiac silhouette "
                 "\u2014 no cardiopulmonary pathology.\n\n"
                 "Heart sounds [cite:4] are normal with no murmurs, gallops, or rubs.\n\n"
                 "**Assessment:** The cardiac workup is entirely reassuring. His exercise intolerance "
                 "is consistent with physical deconditioning rather than an underlying cardiac "
                 "or pulmonary process.",
                 [_c3_ecg(1), _c3_echo(2), _c3_xray(3), _c3_heart(4)], ts="2026-03-10T10:02:45"),

            _msg("user",
                 "What about his labs? Anything to flag for a 45-year-old with his family history?",
                 ts="2026-03-10T10:04:10"),
            _msg("assistant",
                 "The lab results [cite:1] reveal several findings that deserve attention in the "
                 "context of his family history and lifestyle. Fasting glucose is 108 mg/dL with "
                 "HbA1c 5.8%, placing him in the pre-diabetic range \u2014 not yet actionable with "
                 "medication, but lifestyle modification is critical. His lipid profile shows LDL "
                 "142 mg/dL (borderline high), HDL 44 mg/dL (low normal), and triglycerides 168 mg/dL "
                 "(mildly elevated), trending toward metabolic syndrome in the setting of his sedentary "
                 "year and BMI of 26.1.\n\n"
                 "His current medications [cite:2] (Sertraline, PRN Ibuprofen, PRN Melatonin) do not "
                 "include any lipid or glucose management.\n\n"
                 "Clinical notes [cite:3] document a family history of hypertension \u2014 with his "
                 "current BP at 122/78 and these metabolic trends, he is at risk for developing "
                 "hypertension within the next 5\u201310 years if not addressed. This is the window for "
                 "aggressive lifestyle intervention before pharmacotherapy becomes necessary \u2014 a "
                 "structured exercise program and dietary counseling should be the primary prescription "
                 "at this visit.",
                 [_c3_labs(1), _c3_meds(2), _c3_notes(3)], ts="2026-03-10T10:04:55"),
        ],
        "processing": False,
        "uploaded_file": None,
        "patient_id": "P0003",
    }

    # ── Conversation 8: Cardiovascular Risk & Exercise Plan — P0003 (2:2)
    sessions["demo-8"] = {
        "history": [
            _msg("user",
                 "I want to calculate Robert Chen\u2019s cardiovascular risk and put together "
                 "an exercise prescription. What data do we have to work with?",
                 ts="2026-03-11T14:00:00"),
            _msg("assistant",
                 "Labs [cite:1] show LDL 142, HDL 44, triglycerides 168, fasting glucose 108, and "
                 "HbA1c 5.8% \u2014 placing him in a borderline metabolic risk category. Clinical "
                 "notes [cite:2] confirm a 45-year-old male, BMI 26.1, BP 122/78, non-smoker with "
                 "2\u20133 drinks/week and a family history positive for hypertension.\n\n"
                 "The ECG [cite:3] is normal with no ischemic changes, and the echocardiogram [cite:4] "
                 "shows normal EF at 60\u201365% with no structural abnormalities.\n\n"
                 "Using the pooled cohort equations, his age, sex, BP, lipids, and absence of diabetes "
                 "or smoking place his 10-year ASCVD risk at approximately 4\u20136% (low-to-borderline). "
                 "This does not yet meet the threshold for statin therapy, but the trajectory is "
                 "concerning given his sedentary trend. His cardiac workup confirms he is safe for a "
                 "graded exercise program with no restrictions.",
                 [_c3_labs(1), _c3_notes(2), _c3_ecg(3), _c3_echo(4)], ts="2026-03-11T14:00:48"),

            _msg("user",
                 "Good. Draft an exercise and lifestyle plan I can give him. "
                 "Also, should we start a statin or wait?",
                 ts="2026-03-11T14:02:30"),
            _msg("assistant",
                 "Given his ASCVD risk is below 7.5% and he has no diabetes, current guidelines "
                 "favor lifestyle modification before initiating statin therapy. However, if his "
                 "lipid profile does not improve after 3\u20136 months of sustained lifestyle changes, "
                 "a moderate-intensity statin would be appropriate.\n\n"
                 "His medications [cite:1] are minimal \u2014 Sertraline 50mg, PRN Ibuprofen, PRN "
                 "Melatonin. The Ibuprofen should be used sparingly given his borderline BP, as "
                 "chronic NSAID use can elevate blood pressure.\n\n"
                 "Labs [cite:2] confirm no contraindications to exercise (normal renal function, "
                 "no anemia, normal TSH). The chest X-ray [cite:3] and heart sounds [cite:4] are "
                 "both normal.\n\n"
                 "For exercise, aim for 150 min/week of moderate aerobic activity (brisk walking, cycling) "
                 "\u2014 start with 20 min \u00d7 3 days/week, increase by 10% weekly, and add resistance "
                 "training 2 days/week. Diet should shift toward a Mediterranean-style pattern with less "
                 "refined carbohydrates and more fiber, omega-3s, and plant-based proteins. Target BMI <25, "
                 "roughly 5\u20137 kg over 6 months. Current alcohol intake (2\u20133/week) is acceptable.\n\n"
                 "Recheck fasting lipid panel and glucose in 3 months to assess response. If LDL remains "
                 ">130 after 3\u20136 months of adherence, initiate atorvastatin 10\u201320mg.",
                 [_c3_meds(1), _c3_labs(2), _c3_xray(3), _c3_heart(4)], ts="2026-03-11T14:03:22"),
        ],
        "processing": False,
        "uploaded_file": None,
        "patient_id": "P0003",
    }

    # ── Conversation 9: Anxiety & Medication Review — P0003 (2:2) ─
    sessions["demo-9"] = {
        "history": [
            _msg("user",
                 "Robert Chen mentioned his sleep has been worse lately and he\u2019s been "
                 "using Melatonin more often. Can you review his medications and see if "
                 "anything needs adjusting?",
                 ts="2026-03-12T11:15:00"),
            _msg("assistant",
                 "His medication list [cite:1] is straightforward: Sertraline 50mg daily for GAD, "
                 "Ibuprofen 200mg PRN for occasional headaches, and Melatonin 3mg PRN for insomnia "
                 "with reportedly increased use.\n\n"
                 "Clinical notes [cite:2] describe his GAD as well-controlled, but the increased "
                 "insomnia and demanding work schedule suggest his anxiety may be subtly worsening "
                 "even if he does not report overt symptoms. Poor sleep and work stress are common "
                 "triggers for GAD relapse. The Melatonin at 3mg is a reasonable first-line sleep aid, "
                 "but escalating use warrants evaluation of the underlying cause rather than simply "
                 "increasing the dose.\n\n"
                 "His labs [cite:3] show a normal TSH (2.1 mIU/L), which rules out thyroid "
                 "dysfunction as a contributor to insomnia or fatigue.",
                 [_c3_meds(1), _c3_notes(2), _c3_labs(3)], ts="2026-03-12T11:15:42"),

            _msg("user",
                 "Should we uptitrate the Sertraline? And is the Ibuprofen use a concern "
                 "given his labs?",
                 ts="2026-03-12T11:17:15"),
            _msg("assistant",
                 "On Sertraline: the current 50mg dose was uptitrated from 25mg in 2021 [cite:1] "
                 "and has maintained good control. Before uptitrating to 75\u2013100mg, I would "
                 "recommend first addressing the modifiable factors \u2014 specifically his exercise "
                 "deficit and sleep hygiene. Physical activity has strong evidence for reducing "
                 "anxiety symptoms and improving sleep quality, and his cardiac workup [cite:2] "
                 "confirms he is safe to exercise.\n\n"
                 "If sleep does not improve after 4\u20136 weeks of regular exercise and sleep hygiene "
                 "optimization, then uptitrating Sertraline is reasonable as the next step.\n\n"
                 "On Ibuprofen: his labs [cite:3] show borderline BP (122/78) and normal renal "
                 "function. Occasional PRN use at 200mg is low-risk, but he should avoid daily or "
                 "near-daily use \u2014 chronic NSAIDs can raise blood pressure by 3\u20135 mmHg and "
                 "impair renal function over time. If headaches are frequent, Acetaminophen is a "
                 "safer alternative given his cardiovascular risk profile.\n\n"
                 "His clinical notes [cite:4] document the sulfa drug allergy, which is relevant "
                 "if we ever consider thiazide diuretics for blood pressure management down the line.",
                 [_c3_meds(1), _c3_ecg(2), _c3_labs(3), _c3_notes(4)], ts="2026-03-12T11:18:05"),
        ],
        "processing": False,
        "uploaded_file": None,
        "patient_id": "P0003",
    }


_seed_demo_sessions()

# Serve static files (frontend)
frontend_path = Path(__file__).parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")

# Serve multimodal data files
multimodal_path = project_root / "multimodal-data"
app.mount("/multimodal-data", StaticFiles(directory=str(multimodal_path)), name="multimodal-data")


@app.get("/")
async def read_root():
    """Serve the main HTML page."""
    return FileResponse(str(frontend_path / "index.html"))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time log streaming."""
    await websocket.accept()
    ws_manager.connect(websocket)

    # Set the event loop for the WebSocket manager
    if ws_manager.loop is None:
        ws_manager.set_loop(asyncio.get_event_loop())

    try:
        # Send initial connection message
        await websocket.send_json({
            "type": "system",
            "message": "Connected to MedDataOS Monitor",
            "timestamp": ""
        })

        # Keep connection alive and handle incoming messages
        while True:
            data = await websocket.receive_text()
            # Echo back for keep-alive (optional)

    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        ws_manager.disconnect(websocket)


@app.post("/api/start")
async def start_pipeline(request: dict):
    """Start the agent pipeline with user query in a background thread."""
    user_query = request.get("query", "")

    if not user_query:
        return {"status": "error", "message": "Query is required"}

    def run_in_thread():
        run_pipeline(user_query)

    thread = threading.Thread(target=run_in_thread, daemon=True)
    thread.start()

    return {"status": "started", "query": user_query}


@app.post("/api/chat")
async def chat(
    message: str = Form(...),
    session_id: Optional[str] = Form(None),
    patient_id: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    """
    Chat endpoint with session management and file upload support.

    Form Data:
        message: str - User message
        session_id: str (optional) - Session UUID
        file: UploadFile (optional) - .xlsx file upload

    Response:
        {
            "session_id": "uuid",
            "status": "processing" | "error"
        }
    """
    message = message.strip()

    if not message:
        return {"status": "error", "message": "Message is required"}

    # Create new session if not provided or doesn't exist
    if not session_id or session_id not in sessions:
        session_id = str(uuid4())
        sessions[session_id] = {
            "history": [],
            "processing": False,
            "uploaded_file": None,
            "patient_id": patient_id,
        }

    # Set patient_id only if not already assigned to a different patient
    current_patient = sessions[session_id].get("patient_id")
    if patient_id and (not current_patient or current_patient == patient_id):
        sessions[session_id]["patient_id"] = patient_id

    session = sessions[session_id]

    # Reject concurrent requests for the same session
    if session["processing"]:
        return {
            "status": "error",
            "message": "Session is currently processing a request. Please wait.",
            "session_id": session_id
        }

    # Handle file upload
    file_path = None
    if file:
        # Validate file extension
        if not file.filename.lower().endswith(('.csv', '.xlsx')):
            return {
                "status": "error",
                "message": "Only CSV and Excel (.csv, .xlsx) files are allowed",
                "session_id": session_id
            }

        # Create upload directory if it doesn't exist
        upload_dir = project_root / "data" / "input"
        upload_dir.mkdir(parents=True, exist_ok=True)

        # Save to standard location (overwrite if exists)
        extension = '.csv' if file.filename.lower().endswith('.csv') else '.xlsx'
        file_path = upload_dir / f"dataset{extension}"
        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            # Store file path in session
            session["uploaded_file"] = str(file_path)
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to save file: {str(e)}",
                "session_id": session_id
            }
        finally:
            file.file.close()

    # Mark session as processing
    session["processing"] = True

    # Add user message to history
    user_message = message
    if file:
        user_message += f"\n[Uploaded file: {file.filename}]"
    session["history"].append({"role": "user", "content": user_message, "timestamp": datetime.now(timezone.utc).isoformat()})

    # Load patient info if patient_id is set
    p_id = session.get("patient_id")
    patient_info = None
    if p_id:
        patient_file = project_root / "multimodal-data" / "patient-info" / f"{p_id}.json"
        if patient_file.exists():
            try:
                patient_info = json.loads(patient_file.read_text())
            except Exception:
                pass

    # Run pipeline in background with conversation history and file
    def run_in_thread():
        try:
            def _patch_session_kb(kb_by_agent):
                """Patch the last assistant message in session history with knowledge bus data."""
                for msg in reversed(session["history"]):
                    if msg["role"] == "assistant":
                        try:
                            parsed = json.loads(msg["content"])
                            if "citations" in parsed:
                                for c in parsed["citations"]:
                                    if c["agent"] in kb_by_agent:
                                        c["knowledge_bus"] = kb_by_agent[c["agent"]]
                                msg["content"] = json.dumps(parsed, ensure_ascii=False)
                        except (json.JSONDecodeError, KeyError):
                            pass
                        break

            assistant_response = run_pipeline(
                message,
                conversation_history=session["history"],
                patient_id=p_id,
                patient_info=patient_info,
                on_knowledge_bus_complete=_patch_session_kb,
            )
            # Add assistant response to history
            if assistant_response:
                session["history"].append({"role": "assistant", "content": assistant_response, "timestamp": datetime.now(timezone.utc).isoformat()})
        finally:
            # Mark session as no longer processing
            session["processing"] = False

    thread = threading.Thread(target=run_in_thread, daemon=True)
    thread.start()

    return {
        "status": "processing",
        "session_id": session_id
    }


@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    """Get session information for debugging."""
    if session_id not in sessions:
        return {"status": "error", "message": "Session not found"}

    session = sessions[session_id]
    return {
        "status": "success",
        "session_id": session_id,
        "message_count": len(session["history"]),
        "processing": session["processing"],
        "history": session["history"]
    }


@app.get("/api/patients")
async def get_patients():
    """Return list of patients from multimodal-data/patient-info/*.json."""
    patient_dir = project_root / "multimodal-data" / "patient-info"
    multimodal_dir = project_root / "multimodal-data"
    modality_dirs = [
        ("chest-xray", "CXR"),
        ("ecg", "ECG"),
        ("clinical-notes", "Notes"),
        ("echo", "Echo"),
        ("heart-sounds", "Heart"),
        ("lab-results", "Labs"),
        ("medications", "Meds"),
    ]
    patients = []
    for f in sorted(patient_dir.glob("P*.json")):
        try:
            data = json.loads(f.read_text())
            pid = f.stem
            data_dates = data.get("data_dates", {})
            modalities = []
            for dir_name, label in modality_dirs:
                if any((multimodal_dir / dir_name).glob(f"{pid}.*")):
                    modalities.append({
                        "label": label,
                        "date": data_dates.get(dir_name)
                    })
            patients.append({
                "id": pid,
                "name": data.get("name", f.stem),
                "age": data.get("age"),
                "sex": data.get("sex"),
                "blood_type": data.get("blood_type"),
                "allergies": data.get("allergies", []),
                "conditions": data.get("conditions", []),
                "modalities": modalities,
                "data_dates": data.get("data_dates", {}),
            })
        except Exception:
            continue
    return patients


@app.get("/api/patients/{patient_id}/conversations")
async def get_patient_conversations(patient_id: str):
    """Return conversations (sessions) linked to a patient."""
    convos = []
    for sid, session in sessions.items():
        if session.get("patient_id") != patient_id:
            continue
        # Build a preview from the first user message
        preview = ""
        for msg in session["history"]:
            if msg["role"] == "user":
                preview = msg["content"][:60]
                break
        convos.append({
            "session_id": sid,
            "preview": preview or "New conversation",
        })
    return convos


@app.post("/api/patients/{patient_id}/conversations")
async def create_patient_conversation(patient_id: str):
    """Create a new empty session linked to a patient."""
    sid = str(uuid4())
    sessions[sid] = {
        "history": [],
        "processing": False,
        "uploaded_file": None,
        "patient_id": patient_id,
    }
    return {"session_id": sid, "patient_id": patient_id}


@app.get("/api/patients/{patient_id}/pins")
async def get_patient_pins(patient_id: str):
    """Return pinned citations for a patient."""
    return patient_pins.get(patient_id, [])


@app.post("/api/patients/{patient_id}/pins")
async def add_patient_pin(patient_id: str, request: dict):
    """Pin a citation or text selection. Deduplicates accordingly."""
    pin_type = request.get("type", "citation")
    pins = patient_pins.setdefault(patient_id, [])

    if pin_type == "citation":
        citation = request.get("citation")
        if not citation:
            return {"status": "error", "message": "citation is required"}
        # Deduplicate by agent + web_path
        for p in pins:
            if p.get("type", "citation") == "citation" and \
               p.get("citation", {}).get("agent") == citation.get("agent") and \
               p.get("citation", {}).get("web_path") == citation.get("web_path"):
                return {"status": "duplicate", "pin_id": p["pin_id"]}
        pin_id = str(uuid4())
        pins.append({"pin_id": pin_id, "type": "citation", "citation": citation,
                      "source": request.get("source", ""),
                      "created_at": datetime.now(timezone.utc).isoformat(),
                      "query": request.get("query", ""),
                      "annotations": {"text": "", "tags": []}})
    elif pin_type == "text":
        text = request.get("text", "").strip()
        if not text:
            return {"status": "error", "message": "text is required"}
        # Deduplicate by exact text
        for p in pins:
            if p.get("type") == "text" and p.get("text") == text:
                return {"status": "duplicate", "pin_id": p["pin_id"]}
        pin_id = str(uuid4())
        pins.append({
            "pin_id": pin_id,
            "type": "text",
            "text": text,
            "source": request.get("source", ""),
        })
    else:
        return {"status": "error", "message": "invalid pin type"}

    return {"status": "ok", "pin_id": pin_id}


@app.delete("/api/patients/{patient_id}/pins/{pin_id}")
async def remove_patient_pin(patient_id: str, pin_id: str):
    """Unpin a citation for a patient."""
    pins = patient_pins.get(patient_id, [])
    patient_pins[patient_id] = [p for p in pins if p["pin_id"] != pin_id]
    return {"status": "ok"}


@app.patch("/api/patients/{patient_id}/pins/{pin_id}")
async def patch_patient_pin(patient_id: str, pin_id: str, request: dict):
    """Update annotations for a pin."""
    pins = patient_pins.get(patient_id, [])
    for p in pins:
        if p["pin_id"] == pin_id:
            if "annotations" in request:
                p["annotations"] = request["annotations"]
            return p
    return {"status": "error", "message": "Pin not found"}


@app.get("/api/patients/{patient_id}/reviews")
async def get_patient_reviews(patient_id: str):
    """Return review checklist states for a patient."""
    return patient_reviews.get(patient_id, {})


@app.patch("/api/patients/{patient_id}/reviews")
async def patch_patient_review(patient_id: str, request: dict):
    """Upsert review checklist state for a citation."""
    key = request.get("key", "").strip()
    if not key:
        return JSONResponse(status_code=422, content={"error": "key is required"})
    checklist_state = request.get("checklist_state", {})
    patient_reviews.setdefault(patient_id, {})[key] = checklist_state
    return {"key": key, "checklist_state": checklist_state}


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "active_connections": len(ws_manager.active_connections),
        "active_sessions": len(sessions)
    }


def check_port_available(host: str, port: int) -> bool:
    """Check if a port is available."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sock.bind((host, port))
        sock.close()
        return True
    except OSError:
        sock.close()
        return False


def kill_process_on_port(port: int):
    """Kill any process using the specified port."""
    import subprocess
    try:
        # Find process using the port
        result = subprocess.run(
            ['lsof', '-ti', f':{port}'],
            capture_output=True,
            text=True
        )
        if result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                print(f"⚠️  Killing existing process on port {port} (PID: {pid})")
                subprocess.run(['kill', '-9', pid])
            import time
            time.sleep(0.5)  # Give it time to release
    except Exception as e:
        print(f"Could not kill process on port {port}: {e}")


def start_server(host: str = None, port: int = None):
    """Start the FastAPI server with proper cleanup."""
    global _server

    # Use environment variables with sensible defaults
    host = host or os.getenv("HOST", "0.0.0.0")
    port = port or int(os.getenv("PORT", "8080"))

    print(f"\n🌐 Starting MedDataOS Web Monitor...")

    # Check if port is available, kill existing process if needed
    if not check_port_available(host, port):
        print(f"⚠️  Port {port} is already in use")
        kill_process_on_port(port)

        # Verify port is now available
        if not check_port_available(host, port):
            print(f"❌ Failed to free port {port}. Please manually kill the process.")
            print(f"   Run: lsof -ti :{port} | xargs kill -9")
            sys.exit(1)

    print(f"📡 WebSocket server: ws://{host}:{port}/ws")
    print(f"🖥️  Web UI: http://{host}:{port}")
    print(f"💡 Press Ctrl+C to stop (NOT Ctrl+Z)\n")

    # Configure uvicorn with proper socket reuse
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=False,  # Reduce noise
        server_header=False,
        date_header=False
    )

    _server = uvicorn.Server(config)

    # Handle graceful shutdown — let uvicorn close the socket properly
    def signal_handler(sig, frame):
        print("\n\n🛑 Shutting down MedDataOS Monitor...")
        if _server:
            _server.should_exit = True

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        _server.run()
    except Exception as e:
        print(f"❌ Server error: {e}")
    finally:
        print("✓ Server stopped cleanly")


if __name__ == "__main__":
    start_server()
