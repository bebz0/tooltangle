import hashlib
import json
from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

Kind = Literal["plain", "contrast", "no_tool"]
Split = Literal["dev", "holdout"]
Group = tuple[str | None, Kind, str | None]


class Query(BaseModel):
    id: str
    text: str
    tool: str | None = None
    accepted: list[str]
    kind: Kind
    split: Split = "dev"
    against: str | None = None

    @property
    def needs_tool(self) -> bool:
        return bool(self.accepted)


def query_id(kind: str, tool: str | None, text: str) -> str:
    return hashlib.sha256(f"{kind}|{tool}|{text}".encode()).hexdigest()[:12]


class DatasetInfo(BaseModel):
    generators: list[str]
    language: str
    labeled_tools: list[str]
    labeling: str = ""
    updated_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))


class Dataset(BaseModel):
    info: DatasetInfo
    queries: list[Query]

    @classmethod
    def load(cls, path: Path) -> "Dataset | None":
        info_path = path.with_suffix(".info.json")
        if not path.is_file() or not info_path.is_file():
            return None
        queries = [
            Query.model_validate_json(line) for line in path.read_text().splitlines() if line
        ]
        return cls(info=DatasetInfo.model_validate_json(info_path.read_text()), queries=queries)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [query.model_dump_json(exclude_none=True) for query in self.queries]
        path.write_text("\n".join(lines) + "\n")
        info_path = path.with_suffix(".info.json")
        info_path.write_text(json.dumps(self.info.model_dump(), indent=2) + "\n")

    def for_split(self, split: Split) -> list[Query]:
        return [query for query in self.queries if query.split == split]


def assign_splits(queries: list[Query], existing: Sequence[Query] = ()) -> None:
    groups: dict[Group, dict[str, int]] = defaultdict(lambda: {"dev": 0, "holdout": 0})
    overall = {"dev": 0, "holdout": 0}
    for query in existing:
        groups[(query.tool, query.kind, query.against)][query.split] += 1
        overall[query.split] += 1
    for query in sorted(queries, key=lambda query: query.id):
        group = groups[(query.tool, query.kind, query.against)]
        if group["dev"] != group["holdout"]:
            query.split = "dev" if group["dev"] < group["holdout"] else "holdout"
        else:
            query.split = "dev" if overall["dev"] <= overall["holdout"] else "holdout"
        group[query.split] += 1
        overall[query.split] += 1
