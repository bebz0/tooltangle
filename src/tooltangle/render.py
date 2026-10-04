from collections import Counter

from rich.console import Console
from rich.table import Table

from tooltangle.compare import ModelResult
from tooltangle.dataset import Dataset
from tooltangle.findings import Finding, Severity, estimate_definition_tokens
from tooltangle.fixer import Verdict
from tooltangle.generation import GenerationReport
from tooltangle.metrics import Evaluation
from tooltangle.models import UsageLog, model_name
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


def print_generation(console: Console, report: GenerationReport | None, dataset: Dataset) -> None:
    if report is not None:
        dropped = ", ".join(f"{reason} {count}" for reason, count in report.dropped.most_common())
        console.print(
            f"{label('messages')} wrote {report.generated}, kept {report.kept}"
            + (f"  [dim](dropped: {dropped})[/dim]" if dropped else "")
        )
    splits = Counter(query.split for query in dataset.queries)
    console.print(
        f"{label('dataset')} {len(dataset.queries)} messages "
        f"[dim](dev {splits['dev']} · holdout {splits['holdout']})[/dim]"
    )


def print_summary(console: Console, evaluation: Evaluation) -> None:
    low, high = evaluation.interval
    scored = [score for score in evaluation.tools.values() if score.total]
    good = sum(score.accuracy >= 0.9 for score in scored)
    console.print(
        f"\n[green]{'OK':<6}[/green] {good} of {len(scored)} tools picked correctly 90%+ "
        f"of the time"
    )
    console.print(
        f"\n{label('accuracy')} [bold]{evaluation.accuracy:.2f}[/bold] "
        f"[dim][{low:.2f}, {high:.2f}][/dim] on {evaluation.total} messages "
        f"with {model_name(evaluation.model)}"
    )


def print_usage(console: Console, usage: UsageLog) -> None:
    parts = []
    for (phase, spec), entry in usage.entries.items():
        if not entry.calls and not entry.cached:
            continue
        cached = f" + {entry.cached} cached" if entry.cached else ""
        calls = "call" if entry.calls + entry.cached == 1 else "calls"
        parts.append(
            f"{phase} {entry.calls}{cached} {calls} to {model_name(spec)} "
            f"[dim]({entry.input_tokens:,} in · {entry.output_tokens:,} out)[/dim]"
        )
    if not parts:
        return
    console.print(f"{label('usage')} " + f"\n{' ' * 10}".join(parts))
    cost = usage.total_cost()
    if cost:
        amount = "< $0.01" if cost < 0.01 else f"≈ ${cost:.2f}"
        console.print(f"{' ' * 10}[dim]{amount} at paid-tier prices[/dim]")


def print_verdict(console: Console, pair: tuple[str, str], verdict: Verdict | None) -> None:
    console.print(f"\n[bold]{pair[0]} ↔ {pair[1]}[/bold]")
    if verdict is None:
        console.print("  [yellow]✗ no usable rewrite[/yellow]")
        return
    if verdict.added:
        console.print(f"  [dim]wrote {verdict.added} more messages for this pair[/dim]")
    if verdict.underpowered:
        console.print(
            f"  [yellow]✗ only {verdict.before_errors} mistakes on {verdict.evaluated} unseen "
            f"messages, and proving a fix needs at least {verdict.needed}[/yellow]"
        )
        return
    for key, description in verdict.descriptions.items():
        console.print(f'  [dim]{key}:[/dim] "{description}"', highlight=False)
    console.print(
        f"  {label('holdout')} errors {verdict.before_errors} → {verdict.after_errors} "
        f"of {verdict.evaluated}   fixed {verdict.fixed} · broke {verdict.broken}   "
        f"p = {verdict.p_value:.3f} [dim](needs < {verdict.alpha:.3g})[/dim]"
    )
    console.print(
        f"  {label('others')} fixed {verdict.others_fixed} · broke {verdict.others_broken} "
        f"on {verdict.others} messages for related tools"
    )
    if verdict.accepted:
        console.print("  [green]✓ accepted[/green]")
    elif not verdict.proven:
        console.print("  [yellow]✗ not proven better on unseen messages[/yellow]")
    else:
        console.print("  [yellow]✗ breaks other tools[/yellow]")


def print_comparison(
    console: Console, results: list[ModelResult], pick: ModelResult | None
) -> None:
    table = Table(box=None, pad_edge=False, header_style="dim")
    for column in ("model", "accuracy", "missed", "needless", "p50", "$ / 1k", ""):
        table.add_column(column, no_wrap=True)
    for result in sorted(results, key=lambda result: result.evaluation.accuracy, reverse=True):
        low, high = result.evaluation.interval
        note = ""
        if result.p_worse_than_best is not None:
            verdict = "worse" if result.significantly_worse else "not significantly worse"
            note = f"{verdict} than the best (p = {result.p_worse_than_best:.3f})"
        table.add_row(
            model_name(result.spec),
            f"{result.evaluation.accuracy:.2f} [{low:.2f}, {high:.2f}]",
            f"{result.missed_rate:.0%}",
            f"{result.needless_rate:.0%}",
            f"{result.median_latency:.1f}s" if result.median_latency else "-",
            f"{result.cost_per_thousand:.2f}" if result.cost_per_thousand is not None else "?",
            f"[dim]{note}[/dim]",
        )
    console.print(table)
    if pick is not None:
        console.print(
            f"\n{label('pick')} [bold]{model_name(pick.spec)}[/bold] is the cheapest model "
            "that isn't significantly worse than the best one"
        )
