"""Align model-quoted evidence to exact character offsets in the source text.

A fact or claim is only kept when its quote can be found in the note. This is
the main guard against hallucinated evidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

FUZZY_THRESHOLD = 0.85
_MAX_CANDIDATES = 60


@dataclass(frozen=True)
class Span:
    """A located quote: offsets into the source, method and match score."""

    start: int
    end: int
    method: str  # exact | normalized | fuzzy
    score: float


def _normalize_with_map(text: str) -> tuple[str, list[int]]:
    """Lowercase + collapse whitespace; keep a map from normalized to source index."""
    out: list[str] = []
    index: list[int] = []
    prev_space = True
    for i, ch in enumerate(text):
        if ch.isspace():
            if prev_space:
                continue
            out.append(" ")
            index.append(i)
            prev_space = True
            continue
        out.append(ch.lower())
        index.append(i)
        prev_space = False
    return "".join(out), index


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def locate(quote: str, text: str, *, hint: int | None = None) -> Span | None:
    """Find ``quote`` in ``text``: exact, then whitespace/case-normalized, then fuzzy."""
    quote = (quote or "").strip().strip('"').strip()
    if len(quote) < 2 or not text:
        return None

    exact = [m.start() for m in re.finditer(re.escape(quote), text)]
    if exact:
        start = min(exact, key=lambda s: abs(s - hint)) if hint is not None else exact[0]
        return Span(start, start + len(quote), "exact", 1.0)

    norm_text, index = _normalize_with_map(text)
    norm_quote = _norm(quote)
    pos = norm_text.find(norm_quote)
    if pos >= 0:
        start = index[pos]
        end = index[pos + len(norm_quote) - 1] + 1
        return Span(start, end, "normalized", 1.0)

    return _fuzzy(norm_quote, norm_text, index)


def _fuzzy(norm_quote: str, norm_text: str, index: list[int]) -> Span | None:
    """Best approximate window around occurrences of the quote's rarest long token."""
    tokens = sorted(set(re.findall(r"[a-z0-9]{3,}", norm_quote)), key=len, reverse=True)
    if not tokens:
        return None
    qlen = len(norm_quote)
    best: tuple[float, int, int] | None = None
    candidates: list[int] = []
    for token in tokens[:3]:
        candidates.extend(m.start() for m in re.finditer(re.escape(token), norm_text))
        if len(candidates) >= _MAX_CANDIDATES:
            break
    for pos in candidates[:_MAX_CANDIDATES]:
        lo = max(0, pos - qlen)
        window = norm_text[lo : pos + qlen + qlen // 2]
        matcher = SequenceMatcher(None, window, norm_quote, autojunk=False)
        blocks = [b for b in matcher.get_matching_blocks() if b.size]
        if not blocks:
            continue
        matched = sum(b.size for b in blocks)
        score = matched / qlen
        w_start = lo + blocks[0].a
        w_end = lo + blocks[-1].a + blocks[-1].size
        # Penalize matches that are spread far wider than the quote itself.
        if (w_end - w_start) > 1.5 * qlen:
            score *= qlen / (w_end - w_start)
        if best is None or score > best[0]:
            best = (score, w_start, w_end)
    if best is None or best[0] < FUZZY_THRESHOLD:
        return None
    score, s, e = best
    return Span(index[s], index[e - 1] + 1, "fuzzy", round(score, 3))


_NUM = re.compile(r"\d+(?:\.\d+)?")


def value_supported(value: str, quote: str) -> bool:
    """Every number in ``value`` must appear in ``quote`` (unit-free check)."""
    numbers = _NUM.findall(value or "")
    if not numbers:
        return True
    quote_numbers = set(_NUM.findall(quote or ""))
    for number in numbers:
        stripped = number.rstrip("0").rstrip(".") if "." in number else number
        if number not in quote_numbers and stripped not in {
            q.rstrip("0").rstrip(".") if "." in q else q for q in quote_numbers
        }:
            return False
    return True
