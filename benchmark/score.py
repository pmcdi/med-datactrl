"""Score agent responses against clinical benchmark questions."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

QUESTIONS_PATH = Path(__file__).parent / "clinical_questions.json"


@lru_cache(maxsize=1)
def _questions_by_id() -> dict[str, dict[str, Any]]:
    questions = json.loads(QUESTIONS_PATH.read_text())
    return {question["id"]: question for question in questions}


def _normalize(value: Any) -> Any:
    if isinstance(value, list):
        return sorted(str(item) for item in value)
    if isinstance(value, bool):
        return value
    if value is None:
        return ""
    return str(value)


def score_answer(question_id: str, answer: dict[str, Any]) -> dict[str, Any]:
    """Score a structured answer against the expected ground truth."""

    questions = _questions_by_id()
    question = questions.get(question_id)
    if question is None:
        raise ValueError(f"Unknown question id: {question_id}")
    expected = question["expected"]

    field_scores = {
        key: _normalize(answer.get(key)) == _normalize(expected_value)
        for key, expected_value in expected.items()
    }

    correct_fields = sum(field_scores.values())
    total_fields = len(field_scores)

    return {
        "question_id": question_id,
        "outcome_score": correct_fields / total_fields if total_fields else 0.0,
        "field_scores": field_scores,
        "passed": all(field_scores.values()),
    }


def score_run(responses: list[dict[str, Any]]) -> dict[str, Any]:
    """Score a batch of agent responses.

    Each response item must include ``question_id`` and ``answer`` keys.
    Optional ``atom_calls`` and ``tool_calls`` support efficiency metrics.
    """

    scored = [score_answer(item["question_id"], item["answer"]) for item in responses]
    passed = sum(1 for item in scored if item["passed"])

    return {
        "total": len(scored),
        "passed": passed,
        "accuracy": passed / len(scored) if scored else 0.0,
        "results": scored,
    }
