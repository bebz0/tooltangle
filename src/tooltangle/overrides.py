from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import yaml
from langchain_core.tools import BaseTool

from tooltangle.toolset import Toolset

DEFAULT_PATH = Path("tooltangle.overrides.yaml")

Overrides = dict[tuple[str, str], str]


class OverridesError(Exception):
    pass


def load_overrides(path: str | Path = DEFAULT_PATH) -> Overrides:
    path = Path(path)
    if not path.is_file():
        return {}
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise OverridesError(f"{path} should map server names to tools")

    overrides = {}
    for server, tools in data.items():
        if not isinstance(tools, dict):
            raise OverridesError(f"{path}: {server} should map tool names to descriptions")
        for tool, description in tools.items():
            overrides[(str(server), str(tool))] = str(description).strip()
    return overrides


def save_overrides(overrides: Overrides, path: str | Path = DEFAULT_PATH) -> None:
    nested: dict[str, dict[str, str]] = defaultdict(dict)
    for (server, tool), description in sorted(overrides.items()):
        nested[server][tool] = description
    text = yaml.dump(
        dict(nested), sort_keys=False, allow_unicode=True, width=88, default_flow_style=False
    )
    Path(path).write_text(text)


def apply_to_toolset(toolset: Toolset, overrides: Overrides) -> Toolset:
    descriptions = {
        tool.key: overrides[(tool.server, tool.name)]
        for tool in toolset
        if (tool.server, tool.name) in overrides
    }
    return toolset.with_descriptions(descriptions)


def apply_overrides(
    tools: Sequence[BaseTool], path: str | Path = DEFAULT_PATH, server: str | None = None
) -> list[BaseTool]:
    """Return copies of LangChain tools with the descriptions tooltangle verified."""
    overrides = load_overrides(path)
    by_name: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for (server_name, tool_name), description in overrides.items():
        by_name[tool_name].append((server_name, description))
        by_name[f"{server_name}_{tool_name}"].append((server_name, description))

    patched = []
    for tool in tools:
        candidates = by_name.get(tool.name, [])
        if server is not None:
            candidates = [candidate for candidate in candidates if candidate[0] == server]
        if len(candidates) > 1:
            servers = ", ".join(name for name, _ in candidates)
            raise OverridesError(
                f"{tool.name} has overrides for several servers ({servers}); pass server="
            )
        if candidates:
            tool = tool.model_copy(update={"description": candidates[0][1]})
        patched.append(tool)
    return patched
