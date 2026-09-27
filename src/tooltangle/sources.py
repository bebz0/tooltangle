import asyncio
import importlib
import importlib.util
import json
import os
import sys
import tempfile
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

import httpx
from langchain_core.utils.function_calling import convert_to_openai_tool
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client

from tooltangle.toolset import EMPTY_PARAMETERS, Toolset, ToolSpec

KNOWN_CLIENTS = ("claude-desktop", "claude-code", "cursor", "vscode")


class SourceError(Exception):
    pass


@dataclass
class ServerConfig:
    name: str
    transport: str
    command: str | None = None
    args: list[str] = field(default_factory=list)
    env: dict[str, str] | None = None
    cwd: str | None = None
    url: str | None = None
    headers: dict[str, str] | None = None


@dataclass
class LoadedTools:
    toolset: Toolset
    failures: dict[str, str] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)


def known_config_paths(client: str) -> list[Path]:
    home = Path.home()
    if client == "claude-desktop":
        if sys.platform == "darwin":
            return [home / "Library/Application Support/Claude/claude_desktop_config.json"]
        if sys.platform == "win32":
            return [Path(os.environ.get("APPDATA", home)) / "Claude/claude_desktop_config.json"]
        return [home / ".config/Claude/claude_desktop_config.json"]
    if client == "claude-code":
        return [Path(".mcp.json"), home / ".claude.json"]
    if client == "cursor":
        return [Path(".cursor/mcp.json"), home / ".cursor/mcp.json"]
    if client == "vscode":
        return [Path(".vscode/mcp.json")]
    raise SourceError(f"unknown client {client!r}, expected one of: {', '.join(KNOWN_CLIENTS)}")


def read_config(path: Path) -> tuple[list[ServerConfig], dict[str, str]]:
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise SourceError(f"{path} is not valid JSON: {error}") from error

    entries = data.get("mcpServers") or data.get("servers") or {}

    servers, skipped = [], {}
    for name, entry in entries.items():
        if entry.get("disabled"):
            skipped[name] = "disabled in config"
            continue
        try:
            servers.append(parse_server(name, entry))
        except SourceError as error:
            skipped[name] = str(error)
    return servers, skipped


def parse_server(name: str, entry: dict[str, Any]) -> ServerConfig:
    transport = entry.get("type") or entry.get("transport")
    if transport is None:
        transport = "stdio" if "command" in entry else "http"
    if transport in ("streamable_http", "streamable-http", "streamableHttp"):
        transport = "http"

    if transport == "stdio":
        if not entry.get("command"):
            raise SourceError("stdio server without a command")
        return ServerConfig(
            name=name,
            transport="stdio",
            command=entry["command"],
            args=[str(argument) for argument in entry.get("args", [])],
            env={key: str(value) for key, value in entry["env"].items()}
            if entry.get("env")
            else None,
            cwd=entry.get("cwd"),
        )
    if transport == "http":
        url = entry.get("url") or entry.get("serverUrl")
        if not url:
            raise SourceError("http server without a url")
        return ServerConfig(
            name=name,
            transport="http",
            url=url,
            headers={key: str(value) for key, value in entry.get("headers", {}).items()},
        )
    raise SourceError(f"unsupported transport {transport!r}")


async def load_tools(source: str, timeout: float = 60) -> LoadedTools:
    if source in KNOWN_CLIENTS:
        paths = [path for path in known_config_paths(source) if path.is_file()]
        if not paths:
            searched = ", ".join(str(path) for path in known_config_paths(source))
            raise SourceError(f"no {source} config found (looked at {searched})")
        return await load_config_files(paths, timeout)

    path = Path(source)
    if path.suffix == ".json" or path.is_file():
        if not path.is_file():
            raise SourceError(f"{source} does not exist")
        return await load_config_files([path], timeout)

    if ":" in source:
        return LoadedTools(Toolset(load_python_tools(source)))

    raise SourceError(
        f"can't tell what {source!r} is: pass an MCP config file, one of "
        f"{', '.join(KNOWN_CLIENTS)}, or a Python target like my_agent.tools:TOOLS"
    )


async def load_config_files(paths: list[Path], timeout: float) -> LoadedTools:
    servers: dict[str, ServerConfig] = {}
    skipped: dict[str, str] = {}
    for path in paths:
        found, not_loaded = read_config(path)
        servers.update({server.name: server for server in found})
        skipped.update(not_loaded)
    if not servers and not skipped:
        raise SourceError(f"no MCP servers in {', '.join(str(path) for path in paths)}")
    return await load_servers(list(servers.values()), timeout, skipped)


async def load_servers(
    servers: list[ServerConfig], timeout: float = 60, skipped: dict[str, str] | None = None
) -> LoadedTools:
    results = await asyncio.gather(
        *(load_server(server, timeout) for server in servers), return_exceptions=True
    )
    tools, failures = [], {}
    for server, result in zip(servers, results, strict=True):
        if isinstance(result, BaseException):
            failures[server.name] = str(result)
        else:
            tools.extend(result)
    return LoadedTools(Toolset(tools), failures, skipped or {})


async def load_server(server: ServerConfig, timeout: float = 60) -> list[ToolSpec]:
    with tempfile.TemporaryFile("w+") as errlog:
        try:
            return await asyncio.wait_for(_list_tools(server, errlog), timeout)
        except TimeoutError as error:
            raise SourceError(f"no answer within {timeout:.0f}s{_tail(errlog)}") from error
        except Exception as error:
            raise SourceError(f"{_describe(error)}{_tail(errlog)}") from error


async def _list_tools(server: ServerConfig, errlog: IO[str]) -> list[ToolSpec]:
    async with AsyncExitStack() as stack:
        if server.transport == "stdio":
            parameters = StdioServerParameters(
                command=server.command, args=server.args, env=server.env, cwd=server.cwd
            )
            read, write = await stack.enter_async_context(stdio_client(parameters, errlog=errlog))
        else:
            client = await stack.enter_async_context(
                httpx.AsyncClient(headers=server.headers, timeout=httpx.Timeout(30, read=300))
            )
            read, write, _ = await stack.enter_async_context(
                streamable_http_client(server.url, http_client=client)
            )
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()

        tools, cursor = [], None
        while True:
            page = await session.list_tools(cursor)
            tools.extend(page.tools)
            cursor = page.nextCursor
            if not cursor:
                break

    specs = []
    for tool in tools:
        data = tool.model_dump(by_alias=True, exclude_none=True)
        specs.append(
            ToolSpec(
                server=server.name,
                name=data["name"],
                description=data.get("description", ""),
                parameters=data.get("inputSchema") or EMPTY_PARAMETERS,
            )
        )
    return specs


def _describe(error: BaseException) -> str:
    if isinstance(error, BaseExceptionGroup):
        return "; ".join(_describe(inner) for inner in error.exceptions)
    if isinstance(error, FileNotFoundError):
        return f"command not found: {error.filename or error}"
    return str(error) or type(error).__name__


def _tail(errlog: IO[str], lines: int = 3) -> str:
    errlog.seek(0)
    output = [line.strip() for line in errlog.read().splitlines() if line.strip()]
    if not output:
        return ""
    return " | stderr: " + " / ".join(output[-lines:])


def load_python_tools(target: str) -> list[ToolSpec]:
    module_path, _, attribute = target.rpartition(":")
    module = _import_module(module_path)
    try:
        tools = getattr(module, attribute)
    except AttributeError as error:
        raise SourceError(f"{module_path} has no attribute {attribute!r}") from error

    specs = []
    for tool in tools:
        function = convert_to_openai_tool(tool)["function"]
        specs.append(
            ToolSpec(
                server="python",
                name=function["name"],
                description=function.get("description", ""),
                parameters=function.get("parameters") or EMPTY_PARAMETERS,
            )
        )
    return specs


def _import_module(module_path: str):
    if module_path.endswith(".py") or "/" in module_path:
        path = Path(module_path)
        if not path.is_file():
            raise SourceError(f"{module_path} does not exist")
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[path.stem] = module
        spec.loader.exec_module(module)
        return module

    if str(Path.cwd()) not in sys.path:
        sys.path.insert(0, str(Path.cwd()))
    try:
        return importlib.import_module(module_path)
    except ImportError as error:
        raise SourceError(f"can't import {module_path}: {error}") from error
