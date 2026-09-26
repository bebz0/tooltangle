import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, replace

EMPTY_PARAMETERS = {"type": "object", "properties": {}}

@dataclass(frozen=True)
class ToolSpec:
    server: str
    name: str
    description: str
    parameters: dict = field(default_factory=dict, compare=False)

    @property
    def key(self) -> str:
        return f"{self.server}.{self.name}"

class Toolset:
    def __init__(self, tools: list[ToolSpec]):
        self.tools = list(tools)
        self.by_key = {tool.key: tool for tool in self.tools}
        name_counts = Counter(tool.name for tool in self.tools)
        self.function_names = {
            tool.key: _function_name(
                tool.name if name_counts[tool.name] == 1 else f"{tool.server}_{tool.name}"
            )
            for tool in self.tools
        }
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

    def function_schemas(self) -> list[dict]:
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
    return cleaned[:64]
