import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import StrEnum

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from tooltangle.metrics import Evaluation
from tooltangle.toolset import Toolset

LARGE_DEFINITIONS = 10_000
SHORT_DESCRIPTION = 20
ERROR_RATE, ERROR_COUNT = 0.2, 3
WARN_RATE, WARN_COUNT = 0.1, 2
LOW_ACCURACY, LOW_ACCURACY_MIN_QUERIES = 0.6, 5


class Severity(StrEnum):
    ERROR = "error"
    WARN = "warn"
    INFO = "info"


@dataclass
class Finding:
    severity: Severity
    code: str
    title: str
    detail: str = ""
    examples: list[str] = field(default_factory=list)
    tools: tuple[str, ...] = ()


def estimate_definition_tokens(toolset: Toolset) -> int:
    return len(json.dumps(toolset.function_schemas(), ensure_ascii=False)) // 5


def static_findings(toolset: Toolset) -> list[Finding]:
    findings = []

    servers_by_name = defaultdict(list)
    for tool in toolset:
        servers_by_name[tool.name].append(tool.server)
    for name, servers in servers_by_name.items():
        if len(servers) > 1:
            findings.append(
                Finding(
                    Severity.ERROR,
                    "duplicate-name",
                    f'duplicate tool name "{name}" in {", ".join(servers)}',
                    tools=tuple(f"{server}.{name}" for server in servers),
                )
            )

    for tool in toolset:
        description = tool.description.strip()
        if not description:
            findings.append(
                Finding(Severity.ERROR, "no-description", f"{tool.key} has no description")
            )
        elif len(description) < SHORT_DESCRIPTION:
            findings.append(
                Finding(
                    Severity.WARN,
                    "short-description",
                    f"{tool.key} has a very short description",
                    detail=f'"{description}"',
                    tools=(tool.key,),
                )
            )
        if re.search(r"\bdeprecated\b", description, re.IGNORECASE):
            findings.append(
                Finding(
                    Severity.WARN,
                    "deprecated",
                    f"{tool.key} is marked deprecated but is still exposed",
                    tools=(tool.key,),
                )
            )

    definition_tokens = estimate_definition_tokens(toolset)
    if definition_tokens > LARGE_DEFINITIONS:
        findings.append(
            Finding(
                Severity.WARN,
                "large-definitions",
                f"tool definitions take about {definition_tokens:,} tokens on every request",
            )
        )
    return findings


def describe_pick(picked: str | None) -> str:
    if picked is None:
        return "no tool"
    if picked.startswith("?"):
        return f"{picked[1:]} (not a real tool)"
    return picked


def evaluation_findings(evaluation: Evaluation) -> list[Finding]:
    findings = []
    explained = set()

    for confusion in evaluation.confusions:
        if confusion.expected is None:
            continue
        rate, count = confusion.rate, confusion.count
        if rate >= ERROR_RATE and count >= ERROR_COUNT:
            severity = Severity.ERROR
        elif rate >= WARN_RATE and count >= WARN_COUNT:
            severity = Severity.WARN
        else:
            continue
        explained.add(confusion.expected)
        target = (
            "answered without a tool"
            if confusion.picked is None
            else describe_pick(confusion.picked)
        )
        findings.append(
            Finding(
                severity,
                "missed-call" if confusion.picked is None else "confusion",
                f"{confusion.expected} → {target}",
                detail=f"{rate:.0%} ({count} of {confusion.total})",
                examples=[f'"{text}"' for text in confusion.examples[:1]],
                tools=tuple(key for key in (confusion.expected, confusion.picked) if key),
            )
        )

    wrong_calls = evaluation.no_tool_total - evaluation.no_tool_correct
    if evaluation.no_tool_total and wrong_calls >= WARN_COUNT:
        rate = wrong_calls / evaluation.no_tool_total
        if rate >= WARN_RATE:
            examples = [
                f'"{confusion.examples[0]}" → {describe_pick(confusion.picked)}'
                for confusion in evaluation.confusions
                if confusion.expected is None
            ]
            findings.append(
                Finding(
                    Severity.WARN,
                    "needless-call",
                    "a tool was called for a message that needs none",
                    detail=f"{rate:.0%} ({wrong_calls} of {evaluation.no_tool_total})",
                    examples=examples[:2],
                )
            )

    for score in evaluation.tools.values():
        if score.key in explained or score.total < LOW_ACCURACY_MIN_QUERIES:
            continue
        if score.accuracy < LOW_ACCURACY:
            findings.append(
                Finding(
                    Severity.WARN,
                    "low-accuracy",
                    f"{score.key} is picked correctly only {score.accuracy:.0%} of the time",
                    detail=f"({score.correct} of {score.total})",
                    tools=(score.key,),
                )
            )

    if evaluation.failed:
        findings.append(
            Finding(
                Severity.WARN,
                "failed-requests",
                f"{evaluation.failed} requests failed and are not counted",
                detail=(evaluation.first_error or "")[:160],
            )
        )

    order = {Severity.ERROR: 0, Severity.WARN: 1, Severity.INFO: 2}
    return sorted(findings, key=lambda finding: order[finding.severity])


def tool_document(name: str, description: str) -> str:
    words = re.sub(r"([a-z])([A-Z])", r"\1 \2", name).replace("_", " ").replace("-", " ")
    return f"{words} {description}"


def lookalike_pairs(
    toolset: Toolset, threshold: float = 0.35, limit: int = 10
) -> list[tuple[str, str, float]]:
    if len(toolset) < 2:
        return []
    documents = [tool_document(tool.name, tool.description) for tool in toolset]
    matrix = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform(documents)
    scores = cosine_similarity(matrix)

    keys = toolset.keys
    pairs = [
        (keys[i], keys[j], float(scores[i, j]))
        for i in range(len(keys))
        for j in range(i + 1, len(keys))
        if scores[i, j] >= threshold
    ]
    pairs.sort(key=lambda pair: pair[2], reverse=True)
    return pairs[:limit]
