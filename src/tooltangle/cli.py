# heavy modules (models, scikit-learn, MCP) are imported inside the commands that use them,
# so --help and --version start instantly
from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Callable, Coroutine, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, NoReturn

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from tooltangle import __version__
from tooltangle.sources import KNOWN_CLIENTS

if TYPE_CHECKING:
    from tooltangle.compare import ModelResult
    from tooltangle.dataset import Dataset
    from tooltangle.fixer import Verdict
    from tooltangle.settings import Settings
    from tooltangle.sources import LoadedTools
    from tooltangle.toolset import Toolset
    from tooltangle.workspace import Estimate, Workspace

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Find which tools your LLM agent confuses and fix their descriptions with proof.",
)
console = Console(highlight=False)

CONFIRM_ABOVE = 25
FIX_CALLS_PER_ATTEMPT = 60

Source = Annotated[
    str,
    typer.Argument(
        help=f"MCP config file, one of {', '.join(KNOWN_CLIENTS)}, "
        "or a Python target like agent.tools:TOOLS"
    ),
]
ModelOption = Annotated[
    str | None,
    typer.Option("--model", "-m", help="Model under test, e.g. google_genai:gemini-3.5-flash-lite"),
]
GeneratorOption = Annotated[
    str | None,
    typer.Option("--generator", "-g", help="Model that writes and labels the test messages."),
]
YesOption = Annotated[bool, typer.Option("--yes", "-y", help="Don't ask before model calls.")]
Timeout = Annotated[float, typer.Option(help="Seconds to wait for each MCP server.")]


def show_version(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", callback=show_version, is_eager=True, help="Show the version."),
    ] = False,
) -> None:
    load_dotenv(Path.cwd() / ".env")


def show_event(message: str) -> None:
    console.print(f"[dim]{message}[/dim]")


def fail(message: str, code: int = 2) -> NoReturn:
    console.print(f"[red]error:[/red] {message}")
    raise typer.Exit(code)


def get_settings(**overrides: Any) -> Settings:
    from tooltangle.settings import SettingsError, load_settings

    try:
        return load_settings(**overrides)
    except SettingsError as error:
        fail(str(error))


def load(source: str, timeout: float) -> LoadedTools:
    from tooltangle.sources import SourceError, load_tools

    try:
        with console.status("starting MCP servers and listing their tools..."):
            loaded = asyncio.run(load_tools(source, timeout=timeout))
    except SourceError as error:
        fail(str(error))
    if not len(loaded.toolset):
        fail("no tools were loaded")
    return loaded


def confirm_cost(estimate: Estimate, workspace: Workspace, yes: bool) -> None:
    from tooltangle.models import model_name

    if yes or estimate.total_calls <= CONFIRM_ABOVE:
        return
    parts = []
    if estimate.generator_calls:
        parts.append(
            f"~{estimate.generator_calls} calls to {model_name(workspace.settings.generator)} "
            "to write and label test messages"
        )
    for spec, calls in estimate.target_calls.items():
        if not calls:
            continue
        tokens = calls * estimate.tokens_per_call
        cost = workspace.usage.cost(spec, tokens, 0)
        price = f", ≈ ${cost:.2f} at paid-tier prices" if cost else ""
        parts.append(f"~{calls} calls to {model_name(spec)} (~{tokens:,} input tokens{price})")
    console.print("this run makes " + "\n  and ".join(parts))
    if not sys.stdin.isatty():
        fail("pass --yes to run it without a prompt")
    if not typer.confirm("continue?"):
        raise typer.Exit(1)


@contextmanager
def progress_bar(description: str, total: int) -> Iterator[Callable[[], None]]:
    with Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task(description, total=total)
        yield lambda: progress.advance(task)


def run_async[T](workspace: Workspace, coroutine: Coroutine[Any, Any, T]) -> T:
    from tooltangle.models import ModelError, QuotaExhausted
    from tooltangle.render import print_usage

    try:
        return asyncio.run(coroutine)
    except QuotaExhausted as error:
        print_usage(console, workspace.usage)
        fail(str(error), code=3)
    except ModelError as error:
        fail(str(error))
    except KeyboardInterrupt:
        print_usage(console, workspace.usage)
        fail("stopped; finished model calls are cached, so the next run continues", code=130)
    except Exception as error:
        if os.environ.get("TOOLTANGLE_DEBUG"):
            raise
        fail(f"{type(error).__name__}: {error} (set TOOLTANGLE_DEBUG=1 for the traceback)")


@app.command()
def tools(
    source: Source,
    show_list: Annotated[bool, typer.Option("--list", help="Print every tool.")] = False,
    timeout: Timeout = 60,
) -> None:
    """Load the tools and run the checks that don't need a model."""
    from tooltangle.findings import lookalike_pairs, static_findings
    from tooltangle.render import print_findings, print_lookalikes, print_sources, print_tool_list

    loaded = load(source, timeout)
    print_sources(console, loaded)
    if show_list:
        console.print()
        print_tool_list(console, loaded)
    print_findings(console, static_findings(loaded.toolset))
    print_lookalikes(console, lookalike_pairs(loaded.toolset))


@app.command()
def check(
    source: Source,
    model: ModelOption = None,
    generator: GeneratorOption = None,
    yes: YesOption = False,
    use_overrides: Annotated[
        bool, typer.Option("--overrides/--no-overrides", help="Apply the verified descriptions.")
    ] = True,
    timeout: Timeout = 60,
) -> None:
    """Measure which tools the model confuses and report what to fix."""
    from tooltangle.findings import Severity
    from tooltangle.models import ModelError
    from tooltangle.overrides import OverridesError, apply_to_toolset, stale_overrides
    from tooltangle.render import label
    from tooltangle.workspace import Workspace, estimate_check

    settings = get_settings(model=model, generator=generator)
    loaded = load(source, timeout)
    with Workspace(settings, show_event) as workspace:
        try:
            overrides = workspace.overrides() if use_overrides else {}
            tested = apply_to_toolset(loaded.toolset, overrides)
            stale = stale_overrides(loaded.toolset, overrides)
            estimate = estimate_check(workspace, loaded.toolset, tested, [settings.model])
        except (OverridesError, ModelError) as error:
            fail(str(error))
        confirm_cost(estimate, workspace, yes)
        report = run_async(workspace, run_check(workspace, loaded, tested, stale))

    has_errors = any(finding["severity"] == Severity.ERROR for finding in report["findings"])
    if any(finding["code"] == "confusion" for finding in report["findings"]):
        console.print(f"{label('next')} tooltangle fix {source}")
    raise typer.Exit(1 if has_errors else 0)


async def run_check(
    workspace: Workspace,
    loaded: LoadedTools,
    tested: Toolset,
    stale: list[str],
) -> dict[str, Any]:
    from tooltangle.findings import evaluation_findings, static_findings
    from tooltangle.metrics import allow_looking_first, evaluate
    from tooltangle.models import model_name
    from tooltangle.render import (
        label,
        print_findings,
        print_generation,
        print_sources,
        print_summary,
        print_usage,
    )
    from tooltangle.report import build_report, write_html, write_report
    from tooltangle.workspace import pick_tools, prepare_dataset, require_answers

    settings = workspace.settings
    client = workspace.target()
    print_sources(console, loaded)
    pairs = zip(loaded.toolset, tested, strict=True)
    applied = sum(old.description != new.description for old, new in pairs)
    if applied:
        console.print(f"{label('overrides')} {applied} verified descriptions applied")
    if stale:
        console.print(
            f"{label('overrides')} skipped {', '.join(stale)}: the server changed the "
            "description since the rewrite was verified, run fix again"
        )

    calls = 0
    with console.status("writing test messages...") as status:

        def on_call() -> None:
            nonlocal calls
            calls += 1
            status.update(f"writing test messages ({calls} model calls so far)...")

        dataset, generation = await prepare_dataset(workspace, loaded.toolset, on_call)
    print_generation(console, generation, dataset)
    dataset = allow_looking_first(dataset, tested)

    with progress_bar(f"asking {model_name(client.spec)}", len(dataset.queries)) as advance:
        picks = await pick_tools(workspace, client, tested, dataset.queries, advance)
    require_answers(picks)

    evaluation = evaluate(client.spec, dataset.queries, picks)
    findings = static_findings(tested) + evaluation_findings(evaluation)
    print_findings(console, findings)
    print_summary(console, evaluation)
    console.print()
    print_usage(console, workspace.usage)

    report = build_report(evaluation, findings, dataset.queries, picks, tested)
    write_report(report, settings.state_dir / "report.json")
    html_path = settings.state_dir / "report.html"
    write_html(report, html_path)
    console.print(f"{label('saved')} {settings.dataset_path} · {html_path}")
    console.print(f"{label('report')} {html_path.resolve().as_uri()}", soft_wrap=True)
    return report


@app.command()
def fix(
    source: Source,
    model: ModelOption = None,
    generator: GeneratorOption = None,
    yes: YesOption = False,
    pairs: Annotated[int, typer.Option(help="How many confused pairs to work on.")] = 3,
    attempts: Annotated[int, typer.Option(help="Rewrites to try for each pair.")] = 2,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show the verdicts without saving anything.")
    ] = False,
    top_up: Annotated[
        bool,
        typer.Option(
            "--top-up/--no-top-up", help="Write more messages for a pair when there are too few."
        ),
    ] = True,
    timeout: Timeout = 60,
) -> None:
    """Rewrite the descriptions of confused tools and keep only the proven ones."""
    from tooltangle.fixer import TOP_UP_PER_SIDE, TOP_UP_ROUNDS
    from tooltangle.models import ModelError
    from tooltangle.overrides import (
        Override,
        OverridesError,
        apply_to_toolset,
        fingerprint,
        save_overrides,
    )
    from tooltangle.render import label, print_usage
    from tooltangle.runner import Runner
    from tooltangle.workspace import Estimate, Workspace, tokens_per_call

    settings = get_settings(model=model, generator=generator)
    loaded = load(source, timeout)
    with Workspace(settings, show_event) as workspace:
        dataset = workspace.dataset()
        if dataset is None:
            fail(f"no test messages yet, run: tooltangle check {source}")
        try:
            overrides = workspace.overrides()
            tested = apply_to_toolset(loaded.toolset, overrides)
            runner = Runner(workspace.target(), tested, settings.system_prompt)
            baseline_calls = len(runner.uncached(dataset.queries))
        except (OverridesError, ModelError) as error:
            fail(str(error))
        fix_calls = pairs * attempts * FIX_CALLS_PER_ATTEMPT
        top_up_calls = pairs * TOP_UP_ROUNDS * 2 * TOP_UP_PER_SIDE if top_up else 0
        estimate = Estimate(
            generator_calls=pairs * (attempts + (TOP_UP_ROUNDS * 2 if top_up else 0)),
            target_calls={settings.model: baseline_calls + fix_calls + top_up_calls},
            tokens_per_call=tokens_per_call(tested),
        )
        confirm_cost(estimate, workspace, yes)
        verdicts = run_async(
            workspace, run_fix(workspace, loaded, tested, dataset, pairs, attempts, top_up)
        )
        accepted = [verdict for verdict in verdicts if verdict and verdict.accepted]
        console.print()
        print_usage(console, workspace.usage)
        if not accepted:
            console.print(f"{label('saved')} nothing, no rewrite was proven better")
            raise typer.Exit(0)
        for verdict in accepted:
            for key, description in verdict.descriptions.items():
                tool = loaded.toolset[key]
                overrides[(tool.server, tool.name)] = Override(
                    description, fingerprint(tool.description)
                )
        if dry_run:
            console.print(f"{label('saved')} nothing (dry run)")
            raise typer.Exit(0)
        save_overrides(overrides, settings.overrides_file)
        console.print(
            f"{label('saved')} {len(accepted)} verified rewrites to {settings.overrides_file}"
        )
        console.print(f"{label('next')} tooltangle check {source}")


async def run_fix(
    workspace: Workspace,
    loaded: LoadedTools,
    tested: Toolset,
    dataset: Dataset,
    max_pairs: int,
    attempts: int,
    top_up: bool,
) -> list[Verdict | None]:
    from tooltangle.dataset import Dataset, Query
    from tooltangle.fixer import TOP_UP_PER_SIDE, Fixer, confused_pairs
    from tooltangle.generation import QueryGenerator
    from tooltangle.metrics import allow_looking_first, evaluate
    from tooltangle.models import model_name
    from tooltangle.render import print_verdict
    from tooltangle.runner import Pick
    from tooltangle.workspace import pick_tools, require_answers

    settings = workspace.settings
    client = workspace.target()
    judged = allow_looking_first(dataset, tested)
    writer = QueryGenerator(loaded.toolset, workspace.generator(), settings)

    async def more_messages(pair: tuple[str, str]) -> list[Query]:
        fresh = await writer.more_contrast(pair, dataset.queries, TOP_UP_PER_SIDE)
        dataset.queries.extend(fresh)
        dataset.info.generators = sorted(set(dataset.info.generators) | set(writer.client.used))
        dataset.save(settings.dataset_path)
        added = Dataset(info=dataset.info, queries=fresh)
        return allow_looking_first(added, tested).queries

    with console.status("") as status:

        async def pick(toolset: Toolset, queries: list[Query]) -> dict[str, Pick]:
            done = 0

            def advance() -> None:
                nonlocal done
                done += 1
                status.update(f"asking {model_name(client.spec)} ({done}/{len(queries)})...")

            return await pick_tools(workspace, client, toolset, queries, advance)

        baseline = await pick(tested, judged.queries)
        require_answers(baseline)
        evaluation = evaluate(client.spec, judged.queries, baseline)
        chosen = confused_pairs(evaluation, max_pairs)
        if not chosen:
            console.print("no confused pairs to fix")
            return []

        fixer = Fixer(
            tested,
            judged,
            workspace.generator(),
            pick,
            attempts,
            more_messages if top_up else None,
        )
        verdicts = []
        for pair in chosen:
            status.update(f"working on {pair[0]} and {pair[1]}...")
            verdict = await fixer.try_pair(pair, evaluation, baseline)
            print_verdict(console, pair, verdict)
            verdicts.append(verdict)
    return verdicts


@app.command()
def compare(
    source: Source,
    models: Annotated[
        list[str], typer.Option("--model", "-m", help="A model to compare; repeat the option.")
    ],
    generator: GeneratorOption = None,
    yes: YesOption = False,
    timeout: Timeout = 60,
) -> None:
    """Run the same messages through several models and find the cheapest one that holds up."""
    from tooltangle.models import ModelError
    from tooltangle.overrides import OverridesError, apply_to_toolset
    from tooltangle.render import print_comparison, print_usage
    from tooltangle.workspace import Workspace, estimate_check

    models = list(dict.fromkeys(models))
    if len(models) < 2:
        fail("pass at least two --model options")
    settings = get_settings(generator=generator)
    loaded = load(source, timeout)
    with Workspace(settings, show_event) as workspace:
        try:
            tested = apply_to_toolset(loaded.toolset, workspace.overrides())
            estimate = estimate_check(workspace, loaded.toolset, tested, models)
        except (OverridesError, ModelError) as error:
            fail(str(error))
        confirm_cost(estimate, workspace, yes)
        results, best = run_async(workspace, run_compare(workspace, loaded, tested, models))
        print_comparison(console, results, best)
        console.print()
        print_usage(console, workspace.usage)


async def run_compare(
    workspace: Workspace, loaded: LoadedTools, tested: Toolset, models: list[str]
) -> tuple[list[ModelResult], ModelResult | None]:
    from tooltangle.compare import rank, summarize
    from tooltangle.metrics import allow_looking_first
    from tooltangle.models import model_name
    from tooltangle.workspace import pick_tools, prepare_dataset, require_answers

    with console.status("writing test messages..."):
        dataset, _ = await prepare_dataset(workspace, loaded.toolset)
    dataset = allow_looking_first(dataset, tested)

    picks, results = {}, []
    for spec in models:
        client = workspace.target(spec)
        with progress_bar(f"asking {model_name(spec)}", len(dataset.queries)) as advance:
            picks[spec] = await pick_tools(workspace, client, tested, dataset.queries, advance)
        require_answers(picks[spec])
        results.append(summarize(spec, dataset.queries, picks[spec], workspace.usage))
    return results, rank(results, picks, dataset.queries)
