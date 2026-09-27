import asyncio
from typing import Annotated, NoReturn

import typer
from rich.console import Console

from tooltangle import __version__
from tooltangle.findings import lookalike_pairs, static_findings
from tooltangle.render import print_findings, print_lookalikes, print_sources, print_tool_list
from tooltangle.sources import KNOWN_CLIENTS, LoadedTools, SourceError, load_tools

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
    pass


def fail(message: str, code: int = 2) -> NoReturn:
    console.print(f"[red]error:[/red] {message}")
    raise typer.Exit(code)


def load(source: str, timeout: float) -> LoadedTools:
    try:
        with console.status("starting MCP servers and listing their tools..."):
            loaded = asyncio.run(load_tools(source, timeout=timeout))
    except SourceError as error:
        fail(str(error))
    if not len(loaded.toolset):
        fail("no tools were loaded")
    return loaded


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
