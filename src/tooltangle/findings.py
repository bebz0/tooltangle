import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import StrEnum

from tooltangle.toolset import Toolset

LARGE_DEFINITIONS = 10_000
SHORT_DESCRIPTION = 20


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
