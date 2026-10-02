import asyncio
import os
from collections.abc import Callable, Coroutine, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Any, NoReturn

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from tooltangle import __version__
from tooltangle.findings import Severity, evaluation_findings, lookalike_pairs, static_findings
from tooltangle.metrics import evaluate
from tooltangle.models import ModelError, QuotaExhausted, model_name
from tooltangle.render import (
    label,
    print_findings,
    print_generation,
    print_lookalikes,
    print_sources,
    print_summary,
    print_tool_list,
    print_usage,
)
from tooltangle.report import build_report, write_report
from tooltangle.settings import Settings, SettingsError, load_settings
from tooltangle.sources import KNOWN_CLIENTS, LoadedTools, SourceError, load_tools
from tooltangle.workspace import Workspace, pick_tools, prepare_dataset

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Find which tools your LLM agent confuses and fix their descriptions with proof.",
)
console = Console(highlight=False)

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
    try:
        return load_settings(**overrides)
    except SettingsError as error:
        fail(str(error))


def load(source: str, timeout: float) -> LoadedTools:
    try:
        with console.status("starting MCP servers and listing their tools..."):
            loaded = asyncio.run(load_tools(source, timeout=timeout))
    except SourceError as error:
        fail(str(error))
    if not len(loaded.toolset):
        fail("no tools were loaded")
    return loaded


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
    timeout: Timeout = 60,
) -> None:
    """Measure which tools the model confuses and report what to fix."""
    settings = get_settings(model=model, generator=generator)
    loaded = load(source, timeout)
    with Workspace(settings, show_event) as workspace:
        report = run_async(workspace, run_check(workspace, loaded))

    has_errors = any(finding["severity"] == Severity.ERROR for finding in report["findings"])
    raise typer.Exit(1 if has_errors else 0)


async def run_check(workspace: Workspace, loaded: LoadedTools) -> dict[str, Any]:
    settings = workspace.settings
    client = workspace.target()
    print_sources(console, loaded)

    calls = 0
    with console.status("writing test messages...") as status:

        def on_call() -> None:
            nonlocal calls
            calls += 1
            status.update(f"writing test messages ({calls} model calls so far)...")

        dataset, generation = await prepare_dataset(workspace, loaded.toolset, on_call)
    print_generation(console, generation, dataset)

    with progress_bar(f"asking {model_name(client.spec)}", len(dataset.queries)) as advance:
        picks = await pick_tools(workspace, client, loaded.toolset, dataset.queries, advance)

    evaluation = evaluate(client.spec, dataset.queries, picks)
    findings = static_findings(loaded.toolset) + evaluation_findings(evaluation)
    print_findings(console, findings)
    print_summary(console, evaluation)
    console.print()
    print_usage(console, workspace.usage)

    report = build_report(evaluation, findings, dataset.queries, picks, loaded.toolset)
    report_path = settings.state_dir / "report.json"
    write_report(report, report_path)
    console.print(f"{label('saved')} {settings.dataset_path} · {report_path}")
    return report
