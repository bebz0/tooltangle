import hashlib
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

from tooltangle.toolset import Toolset

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

DEFAULT_PATH = Path("tooltangle.overrides.yaml")


@dataclass(frozen=True)
class Override:
    description: str
    # fingerprint of the description the rewrite was verified against
    replaces: str | None = None

    def fits(self, description: str) -> bool:
        return self.replaces is None or self.replaces == fingerprint(description)


Overrides = dict[tuple[str, str], Override]


class OverridesError(Exception):
    pass


def fingerprint(description: str) -> str:
    return hashlib.sha256(" ".join(description.split()).encode()).hexdigest()[:12]


def load_overrides(path: str | Path = DEFAULT_PATH) -> Overrides:
    path = Path(path)
    if not path.is_file():
        return {}
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except yaml.YAMLError as error:
        raise OverridesError(f"{path} is not valid YAML: {error}") from error
    if not isinstance(data, dict):
        raise OverridesError(f"{path} should map server names to tools")

    overrides = {}
    for server, tools in data.items():
        if not isinstance(tools, dict):
            raise OverridesError(f"{path}: {server} should map tool names to descriptions")
        for tool, value in tools.items():
            overrides[(str(server), str(tool))] = read_override(value, f"{path}: {server}.{tool}")
    return overrides


def read_override(value: Any, where: str) -> Override:
    if isinstance(value, dict):
        description, replaces = value.get("description"), value.get("replaces")
    else:
        description, replaces = value, None
    if not isinstance(description, str) or not description.strip():
        raise OverridesError(f"{where} needs a description as text")
    return Override(description.strip(), str(replaces) if replaces else None)


def save_overrides(overrides: Overrides, path: str | Path = DEFAULT_PATH) -> None:
    nested: dict[str, dict[str, Any]] = defaultdict(dict)
    for (server, tool), override in sorted(overrides.items()):
        if override.replaces:
            nested[server][tool] = {
                "description": override.description,
                "replaces": override.replaces,
            }
        else:
            nested[server][tool] = override.description
    text = yaml.dump(
        dict(nested), sort_keys=False, allow_unicode=True, width=88, default_flow_style=False
    )
    Path(path).write_text(text)


def apply_to_toolset(toolset: Toolset, overrides: Overrides) -> Toolset:
    descriptions = {}
    for tool in toolset:
        override = overrides.get((tool.server, tool.name))
        if override is not None and override.fits(tool.description):
            descriptions[tool.key] = override.description
    return toolset.with_descriptions(descriptions)


def stale_overrides(toolset: Toolset, overrides: Overrides) -> list[str]:
    """Keys of tools whose description changed since their override was verified."""
    return [
        tool.key
        for tool in toolset
        if (override := overrides.get((tool.server, tool.name)))
        and not override.fits(tool.description)
    ]


def apply_overrides(
    tools: Sequence["BaseTool"], path: str | Path = DEFAULT_PATH, server: str | None = None
) -> list["BaseTool"]:
    """Return copies of LangChain tools with the descriptions tooltangle verified."""
    overrides = load_overrides(path)
    by_name: dict[str, list[tuple[str, Override]]] = defaultdict(list)
    for (server_name, tool_name), override in overrides.items():
        by_name[tool_name].append((server_name, override))
        by_name[f"{server_name}_{tool_name}"].append((server_name, override))

    patched = []
    for tool in tools:
        candidates = [
            (server_name, override)
            for server_name, override in by_name.get(tool.name, [])
            if (server is None or server_name == server) and override.fits(tool.description)
        ]
        if len(candidates) > 1:
            servers = ", ".join(name for name, _ in candidates)
            raise OverridesError(
                f"{tool.name} has overrides for several servers ({servers}); pass server="
            )
        if candidates:
            tool = tool.model_copy(update={"description": candidates[0][1].description})
        patched.append(tool)
    return patched
