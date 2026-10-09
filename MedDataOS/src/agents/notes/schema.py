"""Fact ledger schema and deterministic normalization of attributes and values."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from pydantic import BaseModel, Field

CATEGORIES = (
    "demographics", "diagnosis", "staging", "tumor", "pathology", "imaging_finding",
    "lab", "vital", "medication", "allergy", "procedure", "treatment", "response",
    "toxicity", "symptom", "exam", "performance_status", "social_history",
    "family_history", "comorbidity", "outcome", "plan",
)
ASSERTIONS = ("present", "negated", "possible", "historical", "family", "planned")
SOURCE_TYPES = (
    "pathology", "radiology_final", "radiology_prelim", "operative", "physician",
    "nursing_phone", "research_form", "unknown",
)

# Reliability rank used to pick a preferred value when sources disagree (lower wins).
SOURCE_RELIABILITY = {
    "pathology": 0,
    "radiology_final": 1,
    "operative": 2,
    "radiology_prelim": 3,
    "physician": 3,
    "nursing_phone": 5,
    "research_form": 6,
    "unknown": 7,
}


class ExtractedFact(BaseModel):
    """One fact as returned by the extraction model (pre-grounding)."""

    category: str = Field(description="One of: " + ", ".join(CATEGORIES))
    attribute: str = Field(
        description=(
            "Canonical dotted key, e.g. demographics.age, tumor.location, "
            "tumor.size_cm, tumor.histology, tumor.grade, tumor.stage, lab.afp, "
            "rt.dose_gy, rt.fractions, performance.ecog, outcome.vital_status, "
            "outcome.recurrence, recurrence.location, metastasis.site, "
            "imaging.ascites, comorbidity.<name>, medication.<name>, "
            "symptom.<name>, procedure.<name>. tumor.* is only for the primary tumor."
        )
    )
    value: str = Field(description="Value exactly as written in the note")
    unit: str | None = Field(default=None, description="Unit as written, if any")
    event_date: str | None = Field(
        default=None,
        description="Date the event happened if stated (YYYY-MM-DD, YYYY-MM or YYYY)",
    )
    relative_time: str | None = Field(
        default=None,
        description="Relative timing as written, e.g. '~22 mo post-op', '656 days from index'",
    )
    assertion: str = Field(description="One of: " + ", ".join(ASSERTIONS))
    source_type: str = Field(description="One of: " + ", ".join(SOURCE_TYPES))
    quote: str = Field(
        description="Verbatim substring of the note (<= 200 chars) supporting the fact"
    )


class ExtractionResult(BaseModel):
    """Structured-output envelope for per-note extraction."""

    facts: list[ExtractedFact]


@dataclass
class Fact:
    """A grounded fact in the patient ledger."""

    fact_id: str
    category: str
    attribute: str
    value: str
    unit: str | None
    normalized_value: float | str | None
    event_date: str | None
    date_kind: str  # explicit | relative | inferred | note_date | unknown
    relative_time: str | None
    assertion: str
    source_type: str
    note_id: str  # resource id, or "<note_id>#ocr<page>" for OCR-derived text
    note_label: str
    note_date: str | None
    section: str | None
    char_start: int
    char_end: int
    quote: str
    derived: str = "native"  # native | ocr
    grounding: str = "exact"  # exact | normalized | fuzzy
    grounding_score: float = 1.0
    anchor_fact_id: str | None = None
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the fact for JSON persistence."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Fact:
        """Rebuild a fact from its JSON form, ignoring unknown keys."""
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    @property
    def source_note_id(self) -> str:
        """The underlying datactrl note id (strips the OCR suffix)."""
        return self.note_id.split("#", 1)[0]


def make_fact_id(note_id: str, start: int, end: int, attribute: str, value: str) -> str:
    """Stable fact id from its source span and content."""
    raw = f"{note_id}|{start}|{end}|{attribute}|{value}".encode()
    return "F" + hashlib.sha1(raw).hexdigest()[:10]


# --- normalization ---------------------------------------------------------

_ATTRIBUTE_ALIASES = {
    "age": "demographics.age",
    "patient.age": "demographics.age",
    "sex": "demographics.sex",
    "gender": "demographics.sex",
    "demographics.gender": "demographics.sex",
    "weight": "vital.weight",
    "height": "vital.height",
    "bmi": "vital.bmi",
    "tumor.size": "tumor.size_cm",
    "tumor.size_mm": "tumor.size_cm",
    "tumor.diameter": "tumor.size_cm",
    "lesion.size": "tumor.size_cm",
    "lesion.size_cm": "tumor.size_cm",
    "mass.size": "tumor.size_cm",
    "mass.size_cm": "tumor.size_cm",
    "tumor.segment": "tumor.location",
    "lesion.location": "tumor.location",
    "lesion.segment": "tumor.location",
    "mass.location": "tumor.location",
    "tumor.site": "tumor.location",
    "tumor.differentiation": "tumor.grade",
    "tumor.histologic_grade": "tumor.grade",
    "tumor.type": "tumor.histology",
    "diagnosis.histology": "tumor.histology",
    "tumor.tnm": "tumor.stage",
    "tumor.stage_tnm": "tumor.stage",
    "staging.tnm": "tumor.stage",
    "staging.clinical": "tumor.stage",
    "tumor.lirads": "tumor.li_rads",
    "imaging.li_rads": "tumor.li_rads",
    "imaging.lirads": "tumor.li_rads",
    "lirads": "tumor.li_rads",
    "pathology.vascular_invasion": "tumor.vascular_invasion",
    "pathology.margins": "tumor.margin_status",
    "pathology.margin_status": "tumor.margin_status",
    "tumor.margins": "tumor.margin_status",
    "ecog": "performance.ecog",
    "performance_status.ecog": "performance.ecog",
    "performance.status": "performance.ecog",
    "radiation.dose": "rt.dose_gy",
    "radiation.dose_gy": "rt.dose_gy",
    "rt.dose": "rt.dose_gy",
    "treatment.rt_dose": "rt.dose_gy",
    "radiation.fractions": "rt.fractions",
    "imaging.pvt": "imaging.portal_vein_thrombosis",
    "outcome.status": "outcome.vital_status",
    "outcome.death": "outcome.vital_status",
    "vital_status": "outcome.vital_status",
    "lab.alpha_fetoprotein": "lab.afp",
    "lab.platelet": "lab.platelets",
    "lab.plt": "lab.platelets",
}

_ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7, "viii": 8}


def normalize_attribute(attribute: str, category: str | None = None) -> str:
    """Canonicalize an attribute key so the same concept groups across notes."""
    key = attribute.strip().lower()
    key = re.sub(r"[\s\-]+", "_", key)
    key = re.sub(r"_*\._*", ".", key).strip("._")
    key = re.sub(r"\btumour", "tumor", key)
    if key in _ATTRIBUTE_ALIASES:
        return _ATTRIBUTE_ALIASES[key]
    if "." not in key and category:
        prefixed = f"{category}.{key}"
        return _ATTRIBUTE_ALIASES.get(prefixed, prefixed)
    return key


def normalize_choice(value: str | None, allowed: tuple[str, ...], default: str) -> str:
    """Map a free-text enum value onto ``allowed`` (loose matching)."""
    if not value:
        return default
    cleaned = re.sub(r"[\s\-]+", "_", value.strip().lower())
    if cleaned in allowed:
        return cleaned
    for option in allowed:
        if option.startswith(cleaned) or cleaned.startswith(option):
            return option
    return default


def _first_number(text: str) -> float | None:
    match = re.search(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return float(match.group(0)) if match else None


# Spoken numbers as transcribed by voice recognition ("segment for a" = 4a).
_SPOKEN = {"one": 1, "two": 2, "to": 2, "too": 2, "three": 3, "four": 4, "for": 4,
           "five": 5, "six": 6, "seven": 7, "eight": 8}


def _liver_segment(text: str) -> str | None:
    match = re.search(
        r"\b(?:seg(?:ment)?\.?|s)\s*([ivx]+|\d|one|two|to|too|three|four|for|five|six|"
        r"seven|eight)\s*([ab])?\b",
        text.lower(),
    )
    if not match:
        return None
    raw, sub = match.group(1), match.group(2) or ""
    if raw.isdigit():
        num = int(raw)
    else:
        num = _ROMAN.get(raw) or _SPOKEN.get(raw)
    return f"segment {num}{sub}" if num else None


def _margin(text: str) -> str | None:
    if re.search(r"\br0\b|negative|free of|clear|uninvolved|not involved", text):
        return "negative"
    if re.search(r"\br[12]\b|positive|involved|close", text):
        return "positive"
    return None


def normalize_value(attribute: str, value: str, unit: str | None) -> float | str | None:
    """Deterministic value normalization (units, synonyms) for comparison."""
    text = f"{value} {unit or ''}".strip().lower()
    if attribute == "tumor.size_cm":
        number = _first_number(text)
        if number is None:
            return None
        if re.search(r"\bmm\b|millimet", text):
            number /= 10.0
        return round(number, 2)
    if attribute.startswith("rt.dose"):
        number = _first_number(text)
        if number is None:
            return None
        if "cgy" in text:
            number /= 100.0
        return round(number, 2)
    if attribute == "tumor.location":
        segment = _liver_segment(text)
        if segment:
            return segment
    if attribute == "tumor.margin_status":
        margin = _margin(text)
        if margin:
            return margin
    if attribute in {"demographics.age", "performance.ecog", "rt.fractions"} or (
        attribute.startswith(("lab.", "vital."))
    ):
        number = _first_number(text)
        if number is not None:
            return number
    if attribute == "demographics.sex":
        if re.search(r"\b(m|male|man|gentleman|mr)\b", text):
            return "male"
        if re.search(r"\b(f|female|woman|mrs|ms)\b", text):
            return "female"
    if attribute == "tumor.li_rads":
        match = re.search(r"lr[\s\-]?(\d|m|tiv)", text.replace("li-rads", "lr"))
        if match:
            return f"LR-{match.group(1).upper()}"
    if attribute == "tumor.grade":
        if match := re.search(r"\bg\s?([1-4])\b|grade\s?([1-4])\b", text):
            return f"G{match.group(1) or match.group(2)}"
        for phrase, grade in _GRADE_WORDS:
            if phrase in text:
                return grade
    cleaned = re.sub(r"[^a-z0-9.+/ ]+", " ", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    words = [_SYNONYMS.get(word, word) for word in cleaned.split(" ")]
    return " ".join(words).strip() or None


_GRADE_WORDS = (
    ("poorly", "G3"),
    ("moderately", "G2"),
    ("well", "G1"),
    ("undifferentiated", "G4"),
)

# Word-level synonym consolidation so aliases compare equal across notes.
_SYNONYMS = {
    "hcc": "hepatocellular carcinoma",
    "scc": "squamous cell carcinoma",
    "htn": "hypertension",
    "dm": "diabetes",
    "t2dm": "diabetes",
    "cad": "coronary artery disease",
    "copd": "chronic obstructive pulmonary disease",
    "hcv": "hepatitis c",
    "hcv+": "hepatitis c",
    "hbv": "hepatitis b",
    "pvt": "portal vein thrombosis",
    "mets": "metastases",
    "neg": "negative",
    "pos": "positive",
    "m": "male",
    "f": "female",
    "yes": "present",
    "y": "present",
}
