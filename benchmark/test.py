"""Interactive LLM Q&A over multimodal patient cases via the Controller."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

from litellm import completion
from rich import print
from rich.console import Console
from rich.panel import Panel

from benchmark.agent_tools import (
    PATIENT_DIRS,
    TOOL_DEFINITIONS,
    ControllerToolSession,
)

console = Console()

DEFAULT_MODEL = os.environ.get("MODEL", "azure/gpt-4o")
DEFAULT_PATIENT = "TCGA-DD-A4NF"
MAX_TURNS = 150

SYSTEM_PROMPT = """\
You are a clinical assistant summarizing and answering questions about a \
multimodal patient case.

Patient case: {patient}

Available data via tools:
- Clinical notes (text documents)
- Pathology whole-slide images (metadata only, not pixel data)
- Radiology DICOM series (metadata via atoms)

Use the provided tools to inspect the case. Do not guess values.

Typical workflow:
1. list_clinical_notes → get_clinical_note for relevant notes
2. get_pathology_info → list_slide_atoms → call_slide_atom for key fields
3. get_dicom_patient_info → list_dicom_atoms → call_dicom_atom on \
representative series (do not exhaustively query every series)

When writing summaries, include cross-modality observations: connect facts \
across clinical notes, pathology metadata, and radiology series. Note \
agreements, discrepancies, and gaps between modalities (for example dates, \
anatomy, diagnosis, or findings mentioned in notes vs. what imaging/pathology \
metadata supports). Clearly separate retrieved facts from inference.

Answer clearly in plain language. Cite note IDs, slide IDs, or series IDs \
when stating specific facts.
"""


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


def ask(
    session: ControllerToolSession,
    *,
    patient: str,
    question: str,
    model: str,
    verbose: bool = False,
    max_turns: int = MAX_TURNS,
) -> str:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT.format(patient=patient)},
        {"role": "user", "content": question},
    ]

    for turn in range(1, max_turns + 1):
        if verbose:
            console.print(f"[dim]turn {turn}/{max_turns} — calling {model}[/dim]")

        response = completion(
            model=model,
            messages=messages,
            tools=TOOL_DEFINITIONS,
            tool_choice="auto",
        )
        message = response.choices[0].message
        messages.append(message_to_dict(message))

        tool_calls = getattr(message, "tool_calls", None) or []
        if tool_calls:
            for call in tool_calls:
                name = call.function.name
                args = call.function.arguments
                if verbose:
                    console.print(f"[cyan]→ {name}({args})[/cyan]")
                try:
                    result = session.dispatch(name, args)
                except Exception as exc:
                    result = json.dumps({"error": str(exc)})
                    if verbose:
                        console.print(f"[red]tool error: {exc}[/red]")
                if verbose:
                    preview = result if len(result) <= 300 else f"{result[:300]}..."
                    console.print(f"[green]← {preview}[/green]")
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": result,
                    }
                )
            continue

        if message.content:
            return message.content

    return f"(No answer after {max_turns} turns.)"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask an LLM questions about a patient case via the Controller.",
    )
    parser.add_argument(
        "-p",
        "--patient",
        choices=sorted(PATIENT_DIRS),
        default=DEFAULT_PATIENT,
        help=f"Patient case directory (default: {DEFAULT_PATIENT})",
    )
    parser.add_argument(
        "-m",
        "--model",
        default=DEFAULT_MODEL,
        help=f"LiteLLM model id (default: MODEL env or {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "-q",
        "--question",
        help="Single question to ask (omit for interactive REPL)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print tool calls and truncated results",
    )
    args = parser.parse_args()

    console.print(
        Panel(
            f"[bold]Patient[/bold]  {args.patient}\n"
            f"[bold]Model[/bold]    {args.model}\n"
            f"[bold]Data[/bold]      {PATIENT_DIRS[args.patient]}",
            title="Controller LLM chat",
        )
    )

    session = ControllerToolSession()

    if args.question:
        answer = ask(
            session,
            patient=args.patient,
            question=args.question,
            model=args.model,
            verbose=args.verbose,
        )
        console.print(Panel(answer, title="Answer"))
        return

    console.print(
        "Type a question and press Enter. "
        "Commands: [bold]quit[/bold], [bold]patient <name>[/bold], [bold]trace[/bold]\n"
    )
    patient = args.patient

    while True:
        try:
            question = console.input(f"[bold blue]{patient}>[/bold blue] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break

        if not question:
            continue
        if question.lower() in {"q", "quit", "exit"}:
            break
        if question.lower().startswith("patient "):
            name = question.split(maxsplit=1)[1].strip()
            if name not in PATIENT_DIRS:
                console.print(f"[red]Unknown patient {name!r}. Choices: {', '.join(sorted(PATIENT_DIRS))}[/red]")
                continue
            patient = name
            console.print(f"[green]Switched to {patient}[/green]")
            continue
        if question.lower() == "trace":
            console.print(session.tool_calls)
            continue

        answer = ask(
            session,
            patient=patient,
            question=question,
            model=args.model,
            verbose=args.verbose,
        )
        console.print(Panel(answer, title="Answer"))
        if args.verbose:
            console.print(f"[dim]{len(session.atom_calls)} atom call(s), {len(session.tool_calls)} tool call(s)[/dim]")


if __name__ == "__main__":
    main()
