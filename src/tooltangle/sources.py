import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

KNOWN_CLIENTS = ("claude-desktop", "claude-code", "cursor", "vscode") # Supported at the moment
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
