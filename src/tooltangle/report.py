import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tooltangle.dataset import Query
from tooltangle.findings import Finding, estimate_definition_tokens
from tooltangle.metrics import Evaluation, is_correct
from tooltangle.runner import Pick
from tooltangle.toolset import Toolset


def build_report(
    evaluation: Evaluation,
    findings: list[Finding],
    queries: list[Query],
    picks: dict[str, Pick],
    toolset: Toolset,
) -> dict[str, Any]:
    low, high = evaluation.interval
    tools = {}
    for tool in toolset:
        score = evaluation.tools.get(tool.key)
        tools[tool.key] = {
            "description": tool.description,
            "queries": score.total if score else 0,
            "correct": score.correct if score else 0,
            "accuracy": score.accuracy if score and score.total else None,
            "precision": score.precision if score else None,
        }

    rows = []
    for query in queries:
        pick = picks.get(query.id)
        if pick is None:
            continue
        rows.append(
            {
                **query.model_dump(),
                "picked": pick.tool,
                "calls": pick.calls,
                "correct": None if pick.error else is_correct(query, pick),
                "error": pick.error,
            }
        )

    return {
        "model": evaluation.model,
        "fingerprint": toolset.fingerprint(),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "accuracy": evaluation.accuracy,
        "interval": [low, high],
        "evaluated": evaluation.total,
        "failed": evaluation.failed,
        "definition_tokens": estimate_definition_tokens(toolset),
        "findings": [asdict(finding) for finding in findings],
        "confusions": [
            {**asdict(confusion), "rate": confusion.rate} for confusion in evaluation.confusions
        ],
        "tools": tools,
        "queries": rows,
    }


def write_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
