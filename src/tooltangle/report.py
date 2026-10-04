import json
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, PackageLoader, select_autoescape

from tooltangle.dataset import Query
from tooltangle.findings import Finding, describe_pick, estimate_definition_tokens
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


MATRIX_TOOLS = 16


def confusion_matrix(report: dict[str, Any]) -> dict[str, Any]:
    errors: Counter[str] = Counter()
    for confusion in report["confusions"]:
        for key in (confusion["expected"], confusion["picked"]):
            errors[key or "no tool"] += confusion["count"]
    labels = [key for key, _ in errors.most_common(MATRIX_TOOLS)]

    cells: Counter[tuple[str, str]] = Counter()
    for row in report["queries"]:
        if row["correct"] is None:
            continue
        expected, picked = row["tool"] or "no tool", row["picked"] or "no tool"
        if expected in labels and picked in labels:
            cells[(expected, picked)] += 1

    return {
        "labels": labels,
        "cells": {f"{expected}|{picked}": count for (expected, picked), count in cells.items()},
    }


def render_html(report: dict[str, Any]) -> str:
    environment = Environment(
        loader=PackageLoader("tooltangle"), autoescape=select_autoescape(["html"])
    )
    environment.filters["pick"] = describe_pick
    template = environment.get_template("report.html")
    tools = sorted(
        report["tools"].items(),
        key=lambda item: (item[1]["accuracy"] is None, item[1]["accuracy"] or 0),
    )
    return template.render(report=report, tools=tools, matrix=confusion_matrix(report))


def write_html(report: dict[str, Any], path: Path) -> None:
    path.write_text(render_html(report))
