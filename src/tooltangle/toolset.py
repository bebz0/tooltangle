import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

EMPTY_PARAMETERS = {"type": "object", "properties": {}}
MAX_FUNCTION_NAME = 64


@dataclass(frozen=True)
class ToolSpec:
    server: str
    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict, compare=False)
    read_only: bool | None = field(default=None, compare=False)

    @property
    def key(self) -> str:
        return f"{self.server}.{self.name}"


class Toolset:
    def __init__(self, tools: list[ToolSpec]):
        self.tools = list(tools)
        self.by_key = {tool.key: tool for tool in self.tools}
        self.function_names = _function_names(self.tools)
        self._keys_by_function_name = {name: key for key, name in self.function_names.items()}

    def __iter__(self) -> Iterator[ToolSpec]:
        return iter(self.tools)

    def __len__(self) -> int:
        return len(self.tools)

    def __contains__(self, key: object) -> bool:
        return key in self.by_key

    def __getitem__(self, key: str) -> ToolSpec:
        return self.by_key[key]

    @property
    def keys(self) -> list[str]:
        return [tool.key for tool in self.tools]

    @property
    def servers(self) -> list[str]:
        return list(dict.fromkeys(tool.server for tool in self.tools))

    def key_for(self, function_name: str) -> str | None:
        return self._keys_by_function_name.get(function_name)

    def function_schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": self.function_names[tool.key],
                    "description": tool.description,
                    "parameters": tool.parameters or EMPTY_PARAMETERS,
                },
            }
            for tool in self.tools
        ]

    def fingerprint(self) -> str:
        payload = json.dumps(self.function_schemas(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def with_descriptions(self, descriptions: Mapping[str, str]) -> "Toolset":
        return Toolset(
            [
                replace(tool, description=descriptions.get(tool.key, tool.description))
                for tool in self
            ]
        )

    def subset(self, keys: set[str]) -> "Toolset":
        return Toolset([tool for tool in self if tool.key in keys])

    def catalog(self) -> str:
        lines = []
        for tool in self:
            arguments = ", ".join((tool.parameters or {}).get("properties", {}))
            description = " ".join(tool.description.split()) or "(no description)"
            lines.append(f"- {self.function_names[tool.key]}({arguments}): {description}")
        return "\n".join(lines)


def _function_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]", "_", name)
    if not re.match(r"[A-Za-z_]", cleaned):
        cleaned = f"_{cleaned}"
    return cleaned[:MAX_FUNCTION_NAME]


def _function_names(tools: list[ToolSpec]) -> dict[str, str]:
    names = {tool.key: _function_name(tool.name) for tool in tools}
    prefixed: set[str] = set()
    while True:
        counts = Counter(names.values())
        clashing = [
            tool for tool in tools if counts[names[tool.key]] > 1 and tool.key not in prefixed
        ]
        if not clashing:
            break
        for tool in clashing:
            names[tool.key] = _function_name(f"{tool.server}_{tool.name}")
            prefixed.add(tool.key)

    # names that still clash after the server prefix, or after being cut, get a number
    taken: set[str] = set()
    for key, base in names.items():
        name, number = base, 1
        while name in taken:
            number += 1
            suffix = f"_{number}"
            name = base[: MAX_FUNCTION_NAME - len(suffix)] + suffix
        taken.add(name)
        names[key] = name
    return names
