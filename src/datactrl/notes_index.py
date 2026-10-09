"""Deterministic clinical note indexing: sections, dates, note type, quality, BM25.

No LLM calls. Every function works on plain text and returns character offsets
into that text, so callers can cite exact spans.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from typing import Any

# --- Sections ---------------------------------------------------------------

_KNOWN_SECTIONS: dict[str, str] = {
    "cc": "chief_complaint",
    "chief complaint": "chief_complaint",
    "reason": "reason",
    "reason for consultation": "reason",
    "reason for visit": "reason",
    "indication": "indication",
    "hpi": "hpi",
    "history of present illness": "hpi",
    "interval hx": "interval_history",
    "interval history": "interval_history",
    "pmh": "past_medical_history",
    "past medical history": "past_medical_history",
    "psh": "past_surgical_history",
    "past surgical history": "past_surgical_history",
    "meds": "medications",
    "medications": "medications",
    "current medications": "medications",
    "all": "allergies",
    "allergies": "allergies",
    "social history": "social_history",
    "smoking history": "social_history",
    "family history": "family_history",
    "ros": "review_of_systems",
    "review of systems": "review_of_systems",
    "pe": "physical_exam",
    "physical exam": "physical_exam",
    "exam": "physical_exam",
    "labs": "labs",
    "laboratory": "labs",
    "technique": "technique",
    "findings": "findings",
    "impression": "impression",
    "addendum": "addendum",
    "diagnosis": "diagnosis",
    "dx": "diagnosis",
    "cancer diagnosis": "diagnosis",
    "preop dx": "preop_diagnosis",
    "postop dx": "postop_diagnosis",
    "pathology": "pathology",
    "path prelim": "pathology",
    "specimen": "specimen",
    "proc": "procedure",
    "procedure": "procedure",
    "complications": "complications",
    "assessment": "assessment",
    "plan": "plan",
    "assessment/plan": "assessment_plan",
    "assessment and plan": "assessment_plan",
    "a/p": "assessment_plan",
    "treatment plan": "treatment_plan",
    "treatment intent": "treatment_plan",
    "treatment options discussed": "treatment_plan",
    "treatment summary": "treatment_summary",
    "current treatment": "current_treatment",
    "systemic therapy": "systemic_therapy",
    "toxicity assessment": "toxicity",
    "tolerance": "toxicity",
    "status at start of rt": "status",
    "disease status": "disease_status",
    "outcome": "outcome",
    "disposition": "disposition",
    "follow-up duration": "follow_up",
    "nurse note": "nursing_note",
    "note type": "note_type",
}

_HEADER_RE = re.compile(
    r"^[ \t]*(?P<title>[A-Za-z][A-Za-z0-9 /&()'\-]{0,48}?)"
    r"(?:[ \t]*\([^)\n]{0,30}\))?[ \t]*:(?P<rest>[^\n]*)$",
    re.MULTILINE,
)


@dataclass(frozen=True)
class Section:
    """One contiguous section of a note with character offsets."""

    name: str
    title: str
    start: int
    end: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize the section for API responses."""
        return {
            "name": self.name,
            "title": self.title,
            "start": self.start,
            "end": self.end,
            "chars": self.end - self.start,
        }


def _canonical_section(title: str, rest: str) -> str | None:
    """Map a header title to a canonical section name, or None if not a header."""
    key = re.sub(r"\s+", " ", title.strip().lower())
    key = re.sub(r"\s*\([^)]*\)$", "", key)
    if key in _KNOWN_SECTIONS:
        return _KNOWN_SECTIONS[key]
    # Unknown title: only a header when nothing follows the colon (a heading line).
    if rest.strip() or len(key.split()) > 6:
        return None
    return re.sub(r"[^a-z0-9]+", "_", key).strip("_") or None


def sectionize(text: str) -> list[Section]:
    """Split note text into sections covering the full text.

    Text before the first recognised header becomes a ``header`` section.
    """
    starts: list[tuple[int, str, str]] = []
    for match in _HEADER_RE.finditer(text):
        name = _canonical_section(match.group("title"), match.group("rest"))
        if name is None:
            continue
        starts.append((match.start(), name, match.group("title").strip()))

    sections: list[Section] = []
    if not starts or starts[0][0] > 0:
        first = starts[0][0] if starts else len(text)
        if text[:first].strip():
            sections.append(Section("header", "", 0, first))
    for idx, (start, name, title) in enumerate(starts):
        end = starts[idx + 1][0] if idx + 1 < len(starts) else len(text)
        sections.append(Section(name, title, start, end))
    if not sections and text:
        sections.append(Section("body", "", 0, len(text)))
    return sections


def section_at(sections: list[Section], offset: int) -> Section | None:
    """Return the section containing ``offset``."""
    for section in sections:
        if section.start <= offset < section.end:
            return section
    return sections[-1] if sections else None


# --- Dates ------------------------------------------------------------------

_DATE_TOKEN = (
    r"(?P<date>\d{4}-\d{1,2}-\d{1,2}|\d{1,2}/\d{1,2}/\d{2,4}|"
    r"[A-Za-z]{3,9}\.? \d{1,2},? \d{4})"
)
_HEADER_DATE_RE = re.compile(
    r"^[ \t]*(?P<label>Date of Service|Visit Date|Encounter Date|Study Date|"
    r"Date|DOS)[ \t]*:[ \t]*" + _DATE_TOKEN + r"(?P<rest>[^\n]*)$",
    re.MULTILINE | re.IGNORECASE,
)
_FILENAME_DATE_RE = re.compile(r"(?<!\d)(\d{4})[-_](\d{2})[-_](\d{2})(?!\d)")
_FILENAME_YEAR_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
_DOUBT_RE = re.compile(r"\?|pending|correlat|unverified|approx|illegible", re.I)
_MONTHS = {
    name: idx
    for idx, names in enumerate(
        (
            (),
            ("jan", "january"),
            ("feb", "february"),
            ("mar", "march"),
            ("apr", "april"),
            ("may",),
            ("jun", "june"),
            ("jul", "july"),
            ("aug", "august"),
            ("sep", "sept", "september"),
            ("oct", "october"),
            ("nov", "november"),
            ("dec", "december"),
        )
    )
    for name in names
}


def parse_date(token: str) -> str | None:
    """Parse a date token into ISO ``YYYY-MM-DD``; None when invalid."""
    token = token.strip().rstrip(".,")
    year = month = day = None
    if m := re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", token):
        year, month, day = (int(x) for x in m.groups())
    elif m := re.fullmatch(r"(\d{1,2})/(\d{1,2})/(\d{2,4})", token):
        month, day, year = (int(x) for x in m.groups())
        if year < 100:
            year += 2000 if year < 50 else 1900
    elif m := re.fullmatch(r"([A-Za-z]{3,9})\.? (\d{1,2}),? (\d{4})", token):
        month = _MONTHS.get(m.group(1).lower())
        day, year = int(m.group(2)), int(m.group(3))
    if not (year and month and day):
        return None
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return None


def resolve_note_date(text: str, filename: str) -> dict[str, Any]:
    """Resolve the note's service date from headers and the filename.

    Returns ``{"date", "date_source", "date_confidence", "date_flags"}``.
    ``date`` is ISO ``YYYY-MM-DD``, a bare year, or None.
    """
    flags: list[str] = []
    header_date = header_label = None
    header_doubt = False
    head = text[:4_000]
    for match in _HEADER_DATE_RE.finditer(head):
        parsed = parse_date(match.group("date"))
        if parsed:
            header_date = parsed
            header_label = match.group("label")
            header_doubt = bool(_DOUBT_RE.search(match.group("rest")))
            break

    file_date = None
    if m := _FILENAME_DATE_RE.search(filename):
        file_date = parse_date("-".join(m.groups()))
    file_year = None
    if file_date is None and (m := _FILENAME_YEAR_RE.search(filename)):
        file_year = m.group(1)

    if header_date:
        confidence = "medium" if header_doubt else "high"
        if header_doubt:
            flags.append("header_date_flagged_uncertain")
        if file_date and file_date == header_date:
            confidence = "high"
        elif file_date and file_date != header_date:
            flags.append(f"filename_date_disagrees:{file_date}")
            confidence = "low"
        return {
            "date": header_date,
            "date_source": f"header:{header_label}",
            "date_confidence": confidence,
            "date_flags": flags,
        }
    if file_date:
        return {
            "date": file_date,
            "date_source": "filename",
            "date_confidence": "high",
            "date_flags": flags,
        }
    if file_year:
        return {
            "date": file_year,
            "date_source": "filename_year",
            "date_confidence": "low",
            "date_flags": flags,
        }
    return {
        "date": None,
        "date_source": None,
        "date_confidence": "none",
        "date_flags": flags,
    }


# --- Note type --------------------------------------------------------------

_NOTE_TYPES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("pathology", ("pathology", "surgical pathology", "path report", "-pr")),
    ("radiology", ("radiology", "mri ", "ct ", "ultrasound", "addendum to mr")),
    ("operative", ("op note", "operative", "procedure note")),
    ("nursing", ("nursing", "phone encounter", "triage", "rn:")),
    ("consult", ("consult",)),
    ("treatment", ("treatment start", "treatment completion", "on-treatment")),
    ("follow_up", ("follow-up", "follow up", "final status", "status update")),
    ("progress", ("progress", "onc follow")),
    ("clinic", ("clinic note", "clinic visit")),
)


def guess_note_type(text: str, filename: str) -> str:
    """Guess a coarse note type from ``Note Type:``, filename, and first lines."""
    explicit = re.search(
        r"^[ \t]*Note Type[ \t]*:[ \t]*(.+)$", text[:2_000], re.M | re.I
    )
    haystacks = []
    if explicit:
        haystacks.append(explicit.group(1).lower())
    haystacks.append(filename.lower().replace("_", " "))
    haystacks.append(text[:300].lower())
    for hay in haystacks:
        for label, keys in _NOTE_TYPES:
            if any(key in hay for key in keys):
                return label
    return "unknown"


# --- Text quality -----------------------------------------------------------

_GOOD_TOKEN = re.compile(
    r"[(\[\"']?(?:[A-Za-z]{2,}(?:[-'/][A-Za-z]+)*|[A-Za-z]|"
    r"\d+(?:[.,:/-]\d+)*(?:[a-zA-Z%]{1,4})?|[A-Za-z]\d{1,3}[A-Za-z]?)"
    r"[)\]\"'.,;:!?%]*"
)


def _good_token_count(tokens: list[str]) -> int:
    good = 0
    for tok in tokens:
        if not _GOOD_TOKEN.fullmatch(tok):
            continue
        letters = re.sub(r"[^A-Za-z]", "", tok)
        # Barcode-like runs of I/l/1 are OCR noise, not words.
        if len(tok) >= 6 and re.fullmatch(r"[Il1|!]+", re.sub(r"\W", "", tok)):
            continue
        if letters.isupper() and len(letters) > 12:
            continue
        good += 1
    return good


def _alnum_tokens(text: str) -> list[str]:
    # Punctuation-only tokens (dividers, checkboxes, dashes) carry no signal.
    return [tok for tok in text.split() if re.search(r"[A-Za-z0-9]", tok)]


def text_quality(text: str) -> dict[str, Any]:
    """Score how word-like the text is; low scores suggest OCR garbage.

    ``score`` is the fraction of word-like tokens. ``garbage_lines`` is the
    fraction of non-empty lines that are mostly non-words or short fragments
    (typical of OCR'd handwriting and barcodes).
    """
    tokens = _alnum_tokens(text)
    if not tokens:
        return {"score": 0.0, "level": "empty", "tokens": 0, "garbage_lines": 0.0}
    score = round(_good_token_count(tokens) / len(tokens), 3)

    lines = [_alnum_tokens(line) for line in text.splitlines()]
    lines = [toks for toks in lines if toks]
    bad = 0
    for toks in lines:
        line_score = _good_token_count(toks) / len(toks)
        avg_len = sum(len(t) for t in toks) / len(toks)
        if line_score < 0.75 or (len(toks) >= 3 and avg_len < 2.6):
            bad += 1
    garbage = round(bad / len(lines), 3) if lines else 0.0

    if score >= 0.85 and garbage < 0.2:
        level = "good"
    elif score >= 0.6 and garbage < 0.5:
        level = "mixed"
    else:
        level = "poor"
    return {
        "score": score,
        "level": level,
        "tokens": len(tokens),
        "garbage_lines": garbage,
    }


# --- BM25 -------------------------------------------------------------------

_STOP = frozenset(
    "a an and are as at be by for from has have in is it of on or the to was were "
    "with w/ per pt patient this that no not".split()
)

# Clinical abbreviation expansions applied to both documents and queries.
_ABBREV: dict[str, tuple[str, ...]] = {
    "hcc": ("hepatocellular", "carcinoma"),
    "hcv": ("hepatitis", "c"),
    "pvt": ("portal", "vein", "thrombosis"),
    "htn": ("hypertension",),
    "rt": ("radiation", "radiotherapy"),
    "ebrt": ("external", "beam", "radiation"),
    "afp": ("alpha", "fetoprotein"),
    "ecog": ("performance", "status"),
    "ps": ("performance", "status"),
    "nkda": ("allergies",),
    "mets": ("metastasis", "metastases"),
    "met": ("metastasis",),
    "scc": ("squamous", "cell", "carcinoma"),
    "dx": ("diagnosis",),
    "hx": ("history",),
    "tx": ("treatment",),
    "fu": ("follow", "up"),
    "wt": ("weight",),
    "etoh": ("alcohol",),
    "aphe": ("arterial", "hyperenhancement"),
    "lr": ("li", "rads"),
    "plt": ("platelets", "thrombocytopenia"),
    "bili": ("bilirubin",),
    "alb": ("albumin",),
    "cr": ("creatinine",),
    "pmh": ("history",),
    "sob": ("shortness", "breath"),
    "mi": ("myocardial", "infarction"),
    "cad": ("coronary", "artery", "disease"),
    "gy": ("dose",),
    "cgy": ("dose",),
    "ruq": ("right", "upper", "quadrant"),
    "otv": ("on", "treatment", "visit"),
    "s/p": ("status", "post"),
}

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[./][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lowercase tokens with clinical abbreviation expansion; stopwords removed."""
    out: list[str] = []
    for tok in _TOKEN_RE.findall(text.lower()):
        if tok in _STOP:
            continue
        out.append(tok)
        out.extend(_ABBREV.get(tok, ()))
    return out


@dataclass
class Chunk:
    """A searchable window of one note section."""

    note_id: str
    section: str
    start: int
    end: int
    tokens: list[str] = field(default_factory=list)


@dataclass
class BM25Index:
    """Small in-memory Okapi BM25 over note section chunks."""

    chunks: list[Chunk]
    k1: float = 1.4
    b: float = 0.75
    _df: Counter[str] = field(default_factory=Counter, init=False)
    _avgdl: float = field(default=1.0, init=False)

    def __post_init__(self) -> None:
        """Precompute document frequencies and average chunk length."""
        for chunk in self.chunks:
            self._df.update(set(chunk.tokens))
        total = sum(len(chunk.tokens) for chunk in self.chunks)
        self._avgdl = total / len(self.chunks) if self.chunks else 1.0

    def search(self, query: str, limit: int) -> list[tuple[Chunk, float, list[str]]]:
        """Return ``(chunk, score, matched_terms)`` sorted by descending score."""
        terms = list(dict.fromkeys(tokenize(query)))
        n_docs = len(self.chunks)
        scored: list[tuple[Chunk, float, list[str]]] = []
        for chunk in self.chunks:
            counts = Counter(chunk.tokens)
            dl = len(chunk.tokens) or 1
            score = 0.0
            matched: list[str] = []
            for term in terms:
                tf = counts.get(term, 0)
                if not tf:
                    continue
                matched.append(term)
                df = self._df[term]
                idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
                norm = tf + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
                score += idf * tf * (self.k1 + 1) / norm
            if score > 0:
                scored.append((chunk, round(score, 4), matched))
        scored.sort(key=lambda item: (-item[1], item[0].note_id, item[0].start))
        return scored[:limit]


def build_chunks(
    note_id: str,
    text: str,
    sections: list[Section],
    *,
    window: int = 1_200,
    overlap: int = 200,
) -> list[Chunk]:
    """Chunk each section into overlapping windows that keep section labels."""
    chunks: list[Chunk] = []
    for section in sections:
        start = section.start
        while start < section.end:
            end = min(section.end, start + window)
            if end < section.end:
                # Prefer to break on a newline inside the last 200 chars.
                cut = text.rfind("\n", start + window - 200, end)
                if cut > start:
                    end = cut
            body = text[start:end]
            if body.strip():
                chunks.append(Chunk(note_id, section.name, start, end, tokenize(body)))
            if end >= section.end:
                break
            start = max(end - overlap, start + 1)
    return chunks


def snippet(
    text: str, start: int, end: int, terms: list[str], width: int = 280
) -> dict:
    """Return a snippet window inside ``[start, end)`` centred on the first match."""
    lowered = text[start:end].lower()
    hit = -1
    for term in terms:
        idx = lowered.find(term)
        if idx >= 0 and (hit < 0 or idx < hit):
            hit = idx
    center = start + (hit if hit >= 0 else 0)
    s = max(start, center - width // 3)
    e = min(end, s + width)
    return {"text": text[s:e], "start": s, "end": e}
