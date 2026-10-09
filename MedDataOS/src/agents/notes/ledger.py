"""Per-patient fact ledger: grounded facts + timeline + conflicts, JSON-persisted."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.agents.notes.schema import Fact

LEDGER_VERSION = "1"


@dataclass
class Ledger:
    """Compiled, source-grounded view of one patient's notes."""

    patient_id: str
    model: str
    built_at: str
    notes: list[dict[str, Any]] = field(default_factory=list)
    docs: dict[str, dict[str, Any]] = field(default_factory=dict)
    facts: list[Fact] = field(default_factory=list)
    dropped: list[dict[str, Any]] = field(default_factory=list)
    timeline: list[dict[str, Any]] = field(default_factory=list)
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    version: str = LEDGER_VERSION

    # --- lookup ---

    def fact(self, fact_id: str) -> Fact | None:
        """Return one fact by id."""
        for item in self.facts:
            if item.fact_id == fact_id:
                return item
        return None

    def note_meta(self, note_id: str) -> dict[str, Any] | None:
        """Return ledger metadata for one source note."""
        base = note_id.split("#", 1)[0]
        return next((n for n in self.notes if n["note_id"] == base), None)

    def query(
        self,
        *,
        category: str | None = None,
        attribute: str | None = None,
        text: str | None = None,
        note_id: str | None = None,
        limit: int = 40,
    ) -> list[Fact]:
        """Filter facts by category, attribute prefix, note, and free-text terms."""
        terms = [t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(t) > 1]
        scored: list[tuple[int, Fact]] = []
        for item in self.facts:
            if category and item.category != category:
                continue
            if attribute and not item.attribute.startswith(attribute.lower()):
                continue
            if note_id and item.source_note_id != note_id.split("#", 1)[0]:
                continue
            score = 1
            if terms:
                hay = f"{item.attribute} {item.value} {item.quote} {item.category}".lower()
                score = sum(1 for t in terms if t in hay)
                if score == 0:
                    continue
            scored.append((score, item))
        scored.sort(key=lambda pair: -pair[0])
        return [item for _, item in scored[:limit]]

    # --- rendering ---

    def digest(self, max_facts: int = 300) -> str:
        """Compact one-line-per-fact rendering for prompts."""
        lines = []
        for item in self.facts[:max_facts]:
            when = item.event_date or (
                f"{item.relative_time} (rel)" if item.relative_time else item.note_date
            )
            unit = f" {item.unit}" if item.unit and item.unit not in item.value else ""
            flags = f" flags={','.join(item.flags)}" if item.flags else ""
            lines.append(
                f"[{item.fact_id}] {when or 'undated'} | {item.attribute} = "
                f"{item.value}{unit} ({item.assertion}; {item.source_type}"
                f"{'; OCR' if item.derived == 'ocr' else ''}) | {item.note_label}{flags}"
            )
        if len(self.facts) > max_facts:
            lines.append(f"... {len(self.facts) - max_facts} more facts (use ledger_query)")
        return "\n".join(lines)

    def timeline_digest(self, max_events: int = 80) -> str:
        """Compact timeline rendering for prompts."""
        lines = []
        for event in self.timeline[:max_events]:
            kind = event.get("date_kind")
            tag = "" if kind in {"explicit", "note"} else f" [{kind}]"
            ref = f"[{event['fact_id']}] " if event.get("fact_id") else ""
            lines.append(f"{event.get('date') or 'undated'}{tag}: {ref}{event['label']}")
        return "\n".join(lines)

    def conflicts_digest(self) -> str:
        """Compact conflict rendering for prompts."""
        if not self.conflicts:
            return "(none detected)"
        lines = []
        for c in self.conflicts:
            values = "; ".join(
                f"{v['value']} [{v['fact_id'] or v['note_label']}] ({v['source_type']}, "
                f"{v['note_label']}, {v.get('note_date') or 'undated'})"
                for v in c["values"]
            )
            lines.append(
                f"{c['conflict_id']} {c['kind']} ({c['severity']}) {c['attribute']}: "
                f"{values}. {c['rationale']}"
            )
        return "\n".join(lines)

    # --- persistence ---

    def to_dict(self) -> dict[str, Any]:
        """Serialize the ledger for JSON persistence and the ledger endpoint."""
        return {
            "version": self.version,
            "patient_id": self.patient_id,
            "model": self.model,
            "built_at": self.built_at,
            "notes": self.notes,
            "docs": self.docs,
            "facts": [item.to_dict() for item in self.facts],
            "dropped": self.dropped,
            "timeline": self.timeline,
            "conflicts": self.conflicts,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Ledger:
        """Rebuild a ledger from its JSON form."""
        return cls(
            patient_id=data["patient_id"],
            model=data.get("model", ""),
            built_at=data.get("built_at", ""),
            notes=list(data.get("notes") or []),
            docs=dict(data.get("docs") or {}),
            facts=[Fact.from_dict(item) for item in data.get("facts") or []],
            dropped=list(data.get("dropped") or []),
            timeline=list(data.get("timeline") or []),
            conflicts=list(data.get("conflicts") or []),
            version=data.get("version", LEDGER_VERSION),
        )

    def save(self, path: Path) -> None:
        """Write the ledger JSON atomically."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=1, ensure_ascii=False))
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> Ledger:
        """Read a ledger JSON file."""
        return cls.from_dict(json.loads(path.read_text()))
