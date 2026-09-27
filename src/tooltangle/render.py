from collections import Counter

from rich.console import Console
from rich.table import Table

from tooltangle.findings import Finding, Severity, estimate_definition_tokens
from tooltangle.sources import LoadedTools

EXAMPLE_WIDTH = 120
SEVERITY_STYLES = {Severity.ERROR: "bold red", Severity.WARN: "yellow", Severity.INFO: "cyan"}


def label(name: str) -> str:
    return f"[dim]{name:<9}[/dim]"


def print_sources(console: Console, loaded: LoadedTools) -> None:
    per_server = Counter(tool.server for tool in loaded.toolset)
    servers = " · ".join(f"{server} {count}" for server, count in per_server.items())
    console.print(f"{label('servers')} {servers or 'none'}")
    console.print(
        f"{label('tools')} {len(loaded.toolset)}    "
        f"definitions ≈ {estimate_definition_tokens(loaded.toolset):,} tokens per request"
    )
    for server, reason in loaded.failures.items():
        console.print(f"{label('failed')} [red]{server}[/red]: {reason}", highlight=False)
    for server, reason in loaded.skipped.items():
        console.print(f"{label('skipped')} {server}: {reason}", highlight=False)


def print_findings(console: Console, findings: list[Finding]) -> None:
    for finding in findings:
        style = SEVERITY_STYLES[finding.severity]
        tag = f"[{style}]{finding.severity.upper():<6}[/{style}]"
        detail = f"  [dim]{finding.detail}[/dim]" if finding.detail else ""
        console.print(f"\n{tag} {finding.title}{detail}", highlight=False)
        for example in finding.examples:
            shown = example if len(example) <= EXAMPLE_WIDTH else example[: EXAMPLE_WIDTH - 1] + "…"
            console.print(f"       [dim]e.g.[/dim] {shown}", highlight=False)


def print_lookalikes(console: Console, pairs: list[tuple[str, str, float]]) -> None:
    if not pairs:
        return
    console.print("\n[bold]look-alike descriptions[/bold] [dim](text similarity only)[/dim]")
    for first, second, score in pairs:
        console.print(f"  {score:.2f}  {first} ~ {second}", highlight=False)


def print_tool_list(console: Console, loaded: LoadedTools) -> None:
    table = Table(box=None, pad_edge=False, show_header=True, header_style="dim")
    table.add_column("tool", no_wrap=True)
    table.add_column("description", overflow="fold")
    for tool in loaded.toolset:
        first_line = tool.description.strip().splitlines()[0] if tool.description.strip() else ""
        table.add_row(tool.key, first_line)
    console.print(table)
