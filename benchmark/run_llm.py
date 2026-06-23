from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Protocol

from litellm import completion

from benchmark.agent_tools import (
    TOOL_DEFINITIONS as CONTROLLER_TOOLS,
    ControllerToolSession,
)
from benchmark.raw_tools import TOOL_DEFINITIONS as RAW_TOOLS, PydicomToolSession
from benchmark.score import score_answer

BENCHMARK_DIR = Path(__file__).parent
QUESTIONS_PATH = BENCHMARK_DIR / "clinical_questions.json"
DEFAULT_OUTPUT_DIR = BENCHMARK_DIR / "results"

LOG = logging.getLogger("benchmark.run_llm")
_RESULT_PREVIEW_LEN = 240


class ToolSession(Protocol):
    tool_calls: list[dict[str, Any]]

    def dispatch(self, name: str, arguments: str | dict[str, Any]) -> str: ...
    def reset_trace(self) -> None: ...


def setup_logging(
    *,
    level: str,
    log_file: Path | None,
    verbose: bool,
) -> None:
    log_level = logging.DEBUG if verbose else getattr(logging, level.upper(), logging.INFO)

    LOG.setLevel(log_level)
    LOG.handlers.clear()
    LOG.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)-5s %(message)s",
        datefmt="%H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(log_level)
    LOG.addHandler(stream_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)
        LOG.addHandler(file_handler)


def _preview(value: Any, max_len: int = _RESULT_PREVIEW_LEN) -> str:
    text = json.dumps(value, default=str) if not isinstance(value, str) else value
    if len(text) <= max_len:
        return text
    return f"{text[:max_len]}..."


def load_questions(question_ids: list[str] | None = None) -> list[dict[str, Any]]:
    questions = json.loads(QUESTIONS_PATH.read_text())
    if question_ids:
        selected = {qid for qid in question_ids}
        questions = [q for q in questions if q["id"] in selected]
        missing = selected - {q["id"] for q in questions}
        if missing:
            msg = f"Unknown question id(s): {', '.join(sorted(missing))}"
            raise ValueError(msg)
    return questions


def _json_keys_hint(question: dict[str, Any]) -> str:
    return json.dumps(list(question["expected"].keys()))


def build_controller_prompt(question: dict[str, Any]) -> str:
    return (
        "You are a radiology imaging assistant answering questions about a "
        "DICOM patient case.\n\n"
        f"Patient case: {question['patient']}\n"
        "Use the provided tools to inspect the case. Do not guess values.\n"
        "Start with get_dicom_patient_info, then list_dicom_atoms, then call_dicom_atom.\n"
        "When you have enough evidence, respond with ONLY a JSON object "
        f"containing exactly these keys: {_json_keys_hint(question)}\n"
        "Use the correct JSON types (strings, booleans, arrays) for each field."
    )


def build_raw_prompt(question: dict[str, Any]) -> str:
    return (
        "You are a radiology imaging assistant answering questions by reading "
        "raw DICOM files. You do NOT have a high-level imaging API.\n\n"
        f"Patient case: {question['patient']}\n"
        f"Data directory: data/{question['patient']}/\n\n"
        "Use list_dir to see series folders, find_series_samples to get one file "
        "per series, and read_dicom to inspect tags.\n"
        "Group files by SeriesInstanceUID. Nested tags appear as JSON objects/arrays.\n"
        "Do not guess values.\n"
        "When you have enough evidence, respond with ONLY a JSON object "
        f"containing exactly these keys: {_json_keys_hint(question)}\n"
        "Use the correct JSON types (strings, booleans, arrays) for each field."
    )


def get_mode_config(mode: str) -> tuple[list[dict[str, Any]], Callable[[], ToolSession], Callable]:
    if mode == "controller":
        return CONTROLLER_TOOLS, ControllerToolSession, build_controller_prompt
    if mode == "raw":
        return RAW_TOOLS, PydicomToolSession, build_raw_prompt
    raise ValueError(f"Unknown mode: {mode!r}")


def _question_result(
    question: dict[str, Any],
    *,
    mode: str,
    model: str,
    answer: dict[str, Any] | None,
    raw_final_content: str | None,
    atom_calls: list[dict[str, Any]],
    files_opened: list[str],
    tool_calls: list[dict[str, Any]],
    llm_turns: int,
    message_count: int,
    elapsed_seconds: float,
    error: str | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "question_id": question["id"],
        "patient": question["patient"],
        "question": question["question"],
        "mode": mode,
        "model": model,
        "answer": answer,
        "raw_final_content": raw_final_content,
        "atom_calls": atom_calls,
        "files_opened": files_opened,
        "tool_calls": tool_calls,
        "llm_turns": llm_turns,
        "message_count": message_count,
        "elapsed_seconds": round(elapsed_seconds, 2),
        "error": error,
    }
    if answer is not None:
        result["score"] = score_answer(question["id"], answer)
    return result


def extract_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return json.loads(cleaned)


def message_to_dict(message: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "role": getattr(message, "role", None) or "assistant",
        "content": message.content,
    }
    tool_calls = getattr(message, "tool_calls", None)
    if tool_calls:
        payload["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
            for call in tool_calls
        ]
    return payload


def _trace_from_session(mode: str, session: ToolSession) -> tuple[list[dict[str, Any]], list[str]]:
    if mode == "controller" and isinstance(session, ControllerToolSession):
        return list(session.atom_calls), []
    if mode == "raw" and isinstance(session, PydicomToolSession):
        return [], list(session.files_opened)
    return [], []


def _log_question_outcome(
    qid: str,
    *,
    mode: str,
    result: dict[str, Any],
    question: dict[str, Any],
    elapsed: float,
) -> None:
    atom_calls = result["atom_calls"]
    files_opened = result["files_opened"]
    cost_label = (
        f"atom_calls={len(atom_calls)}"
        if mode == "controller"
        else f"files_opened={len(files_opened)}"
    )

    score = result.get("score")
    if score:
        if score["passed"]:
            LOG.info(
                "[%s] PASS outcome=%.0f%% %s tool_calls=%d elapsed=%.2fs",
                qid,
                score["outcome_score"] * 100,
                cost_label,
                len(result["tool_calls"]),
                elapsed,
            )
            return

        failed = [field for field, ok in score["field_scores"].items() if not ok]
        LOG.warning(
            "[%s] FAIL outcome=%.0f%% failed_fields=%s %s elapsed=%.2fs",
            qid,
            score["outcome_score"] * 100,
            failed,
            cost_label,
            elapsed,
        )
        LOG.warning("[%s] expected=%s", qid, _preview(question["expected"]))
        LOG.warning("[%s] answer=%s", qid, _preview(result["answer"]))
        return

    if result.get("error"):
        LOG.error("[%s] ERROR %s elapsed=%.2fs — %s", qid, cost_label, elapsed, result["error"])


def run_question(
    question: dict[str, Any],
    *,
    mode: str,
    model: str,
    tools: list[dict[str, Any]],
    build_prompt: Callable[[dict[str, Any]], str],
    session: ToolSession,
    max_turns: int,
    temperature: float,
    question_index: int,
    question_total: int,
) -> dict[str, Any]:
    qid = question["id"]
    session.reset_trace()
    started = time.perf_counter()

    LOG.info(
        "[%s] (%d/%d) mode=%s patient=%s",
        qid,
        question_index,
        question_total,
        mode,
        question["patient"],
    )
    LOG.info("[%s] question: %s", qid, question["question"])

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": build_prompt(question)},
        {"role": "user", "content": question["question"]},
    ]

    final_answer: dict[str, Any] | None = None
    raw_final_content: str | None = None
    error: str | None = None
    llm_turns = 0

    for turn in range(1, max_turns + 1):
        llm_turns = turn
        LOG.info("[%s] turn %d/%d — calling LLM", qid, turn, max_turns)

        response = completion(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=temperature,
        )
        message = response.choices[0].message
        messages.append(message_to_dict(message))

        tool_calls = getattr(message, "tool_calls", None) or []
        if tool_calls:
            LOG.info("[%s] turn %d — %d tool call(s)", qid, turn, len(tool_calls))
            for call in tool_calls:
                LOG.info(
                    "[%s] tool → %s(%s)",
                    qid,
                    call.function.name,
                    call.function.arguments,
                )
                tool_started = time.perf_counter()
                try:
                    tool_result = session.dispatch(
                        call.function.name,
                        call.function.arguments,
                    )
                except Exception as exc:
                    tool_result = json.dumps({"error": str(exc)})
                    LOG.warning("[%s] tool error: %s", qid, exc)

                LOG.info(
                    "[%s] tool ← %s in %.2fs result=%s",
                    qid,
                    call.function.name,
                    time.perf_counter() - tool_started,
                    _preview(tool_result),
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": tool_result,
                    }
                )
            continue

        if message.content:
            raw_final_content = message.content
            try:
                final_answer = extract_json(message.content)
            except json.JSONDecodeError as exc:
                error = f"Final response was not valid JSON: {exc}"
                LOG.error("[%s] %s", qid, error)
            break

        LOG.warning("[%s] turn %d — empty assistant response, continuing", qid, turn)
    else:
        error = f"Exceeded max turns ({max_turns}) without a final answer"
        LOG.error("[%s] %s", qid, error)

    elapsed = time.perf_counter() - started
    atom_calls, files_opened = _trace_from_session(mode, session)
    result = _question_result(
        question,
        mode=mode,
        model=model,
        answer=final_answer,
        raw_final_content=raw_final_content,
        atom_calls=atom_calls,
        files_opened=files_opened,
        tool_calls=list(session.tool_calls),
        llm_turns=llm_turns,
        message_count=len(messages),
        elapsed_seconds=elapsed,
        error=error,
    )
    _log_question_outcome(qid, mode=mode, result=result, question=question, elapsed=elapsed)
    return result


def run_benchmark(
    *,
    mode: str,
    model: str,
    question_ids: list[str] | None = None,
    max_turns: int = 12,
    temperature: float = 0.0,
) -> dict[str, Any]:
    tools, session_factory, build_prompt = get_mode_config(mode)
    questions = load_questions(question_ids)
    session = session_factory()
    total = len(questions)

    LOG.info("Loaded %d question(s) from %s", total, QUESTIONS_PATH)
    LOG.info(
        "Run config mode=%s model=%s max_turns=%d temperature=%s",
        mode,
        model,
        max_turns,
        temperature,
    )

    results: list[dict[str, Any]] = []
    run_started = time.perf_counter()

    for index, question in enumerate(questions, start=1):
        try:
            result = run_question(
                question,
                mode=mode,
                model=model,
                tools=tools,
                build_prompt=build_prompt,
                session=session,
                max_turns=max_turns,
                temperature=temperature,
                question_index=index,
                question_total=total,
            )
        except Exception as exc:
            LOG.exception("[%s] unhandled error", question["id"])
            result = _question_result(
                question,
                mode=mode,
                model=model,
                answer=None,
                raw_final_content=None,
                atom_calls=[],
                files_opened=[],
                tool_calls=[],
                llm_turns=0,
                message_count=0,
                elapsed_seconds=0.0,
                error=str(exc),
            )
        results.append(result)
        LOG.info("")

    run_elapsed = time.perf_counter() - run_started
    scored = [r for r in results if r.get("score")]
    passed = sum(1 for r in scored if r["score"]["passed"])
    errored = sum(1 for r in results if r.get("error") and not r.get("score"))

    LOG.info("=" * 60)
    LOG.info(
        "Run complete mode=%s in %.2fs — passed %d/%d scored, %d error(s)",
        mode,
        run_elapsed,
        passed,
        len(scored),
        errored,
    )

    cost_key = "atom_calls" if mode == "controller" else "files_opened"
    for item in results:
        qid = item["question_id"]
        if item.get("error") and not item.get("score"):
            LOG.info("  [ERROR] %s — %s", qid, item["error"])
            continue
        status = "PASS" if item.get("score", {}).get("passed") else "FAIL"
        LOG.info(
            "  [%s] %s — %s=%d llm_turns=%s elapsed=%ss",
            status,
            qid,
            cost_key,
            len(item.get(cost_key, [])),
            item.get("llm_turns", "?"),
            item.get("elapsed_seconds", "?"),
        )

    return {
        "mode": mode,
        "model": model,
        "run_at": datetime.now(UTC).isoformat(),
        "elapsed_seconds": round(run_elapsed, 2),
        "total": len(results),
        "scored": len(scored),
        "passed": passed,
        "errored": errored,
        "accuracy": passed / len(scored) if scored else 0.0,
        "results": results,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run clinical benchmark questions with LiteLLM.",
    )
    parser.add_argument(
        "--mode",
        choices=["controller", "raw"],
        default="controller",
        help="Tool environment: controller API or raw pydicom (default: controller)",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("MODEL", "gemini/gemini-2.0-flash"),
        help="LiteLLM model name (default: MODEL env or gemini/gemini-2.0-flash)",
    )
    parser.add_argument(
        "--ids",
        nargs="*",
        help="Optional question ids to run (default: all)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=12,
        help="Maximum tool-calling turns per question",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to write JSON results",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Optional path to write log output (default: alongside JSON as .log)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging",
    )
    return parser.parse_args(argv)


def default_output_path(mode: str, model: str) -> Path:
    safe_model = re.sub(r"[^\w.-]+", "_", model)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return DEFAULT_OUTPUT_DIR / f"{mode}_{safe_model}_{timestamp}.json"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_path = args.output or default_output_path(args.mode, args.model)
    log_path = args.log_file or output_path.with_suffix(".log")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    setup_logging(level=args.log_level, log_file=log_path, verbose=args.verbose)

    LOG.info("Output JSON: %s", output_path)
    LOG.info("Log file: %s", log_path)

    report = run_benchmark(
        mode=args.mode,
        model=args.model,
        question_ids=args.ids,
        max_turns=args.max_turns,
        temperature=args.temperature,
    )

    output_path.write_text(json.dumps(report, indent=2, default=str))
    LOG.info("Wrote results to %s", output_path)
    LOG.info(
        "Accuracy: %d/%d (%.0f%%)",
        report["passed"],
        report["scored"],
        report["accuracy"] * 100,
    )

    return 0 if report["passed"] == report["scored"] and report["errored"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
