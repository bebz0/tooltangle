import asyncio
import hashlib
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field, replace

from pydantic import BaseModel, Field

from tooltangle import prompts
from tooltangle.dataset import Dataset, DatasetInfo, Kind, Query, assign_splits, query_id
from tooltangle.models import ModelClient
from tooltangle.settings import Settings
from tooltangle.toolset import Toolset

PLAIN_BATCH = 4
LABEL_BATCH = 50
NO_TOOL_BATCH = 25
NO_TOOL_SHARE = 0.1


class Requests(BaseModel):
    requests: list[str] = Field(description="The user messages, one per item")


class ToolRequests(BaseModel):
    tool: str = Field(description="The tool name exactly as written in the list")
    requests: list[str] = Field(description="The user messages for this tool")


class RequestsByTool(BaseModel):
    tools: list[ToolRequests]


class Label(BaseModel):
    number: int
    tools: list[str]


class Labels(BaseModel):
    labels: list[Label]


@dataclass
class Draft:
    text: str
    tool: str | None
    kind: Kind


@dataclass
class GenerationReport:
    generated: int = 0
    kept: int = 0
    dropped: Counter[str] = field(default_factory=Counter)


LABELING_VERSION = hashlib.sha256(prompts.LABELS.encode()).hexdigest()[:12]


def no_tool_count(settings: Settings, tool_count: int) -> int:
    return max(6, round(NO_TOOL_SHARE * tool_count * settings.queries_per_tool))


class QueryGenerator:
    def __init__(
        self,
        toolset: Toolset,
        client: ModelClient,
        settings: Settings,
        on_call: Callable[[], None] = lambda: None,
    ):
        self.toolset = toolset
        self.client = client
        self.settings = settings
        self.on_call = on_call
        self.catalog = toolset.catalog()
        self.semaphore = asyncio.Semaphore(settings.concurrency)

    async def build(self) -> tuple[Dataset, GenerationReport]:
        report = GenerationReport()
        keys = self.toolset.keys
        batches = [keys[start : start + PLAIN_BATCH] for start in range(0, len(keys), PLAIN_BATCH)]
        jobs = [self.plain_queries(batch) for batch in batches]
        total = no_tool_count(self.settings, len(self.toolset))
        for start in range(0, total, NO_TOOL_BATCH):
            jobs.append(self.no_tool_queries(min(NO_TOOL_BATCH, total - start)))

        drafts = [draft for batch in await asyncio.gather(*jobs) for draft in batch]
        report.generated = len(drafts)
        drafts = self.clean(drafts, report)

        queries = await self.label(drafts, report)
        assign_splits(queries)

        order: dict[str | None, int] = {key: index for index, key in enumerate(self.toolset.keys)}
        queries.sort(key=lambda query: (order.get(query.tool, len(order)), query.kind, query.id))
        report.kept = len(queries)
        info = DatasetInfo(
            generators=[self.client.spec],
            language=self.settings.language,
            labeled_tools=self.toolset.keys,
            labeling=LABELING_VERSION,
        )
        return Dataset(info=info, queries=queries), report

    async def ask[T: BaseModel](self, schema: type[T], prompt: str, phase: str) -> T:
        async with self.semaphore:
            answer = await self.client.ask(schema, prompt, phase)
        self.on_call()
        return answer

    def function(self, key: str) -> str:
        return self.toolset.function_names[key]

    async def plain_queries(self, keys: list[str]) -> list[Draft]:
        count = self.settings.queries_per_tool
        prompt = prompts.PLAIN_QUERIES.format(
            catalog=self.catalog,
            count=count,
            functions=", ".join(f"`{self.function(key)}`" for key in keys),
            language=self.settings.language,
        )
        answer = await self.ask(RequestsByTool, prompt, "generate")
        drafts = []
        for item in answer.tools:
            key = self.toolset.key_for(item.tool.strip("` "))
            if key in keys:
                drafts += [Draft(text, key, "plain") for text in item.requests[:count]]
        return drafts

    async def no_tool_queries(self, count: int) -> list[Draft]:
        prompt = prompts.NO_TOOL_QUERIES.format(
            catalog=self.catalog, count=count, language=self.settings.language
        )
        answer = await self.ask(Requests, prompt, "generate")
        return [Draft(text, None, "no_tool") for text in answer.requests[:count]]

    def clean(self, drafts: list[Draft], report: GenerationReport) -> list[Draft]:
        seen = set()
        cleaned = []
        for draft in drafts:
            text = " ".join(draft.text.split())
            if not text or text.casefold() in seen:
                report.dropped["duplicate"] += 1
                continue
            if draft.tool and self.function(draft.tool).casefold() in text.casefold():
                report.dropped["names the tool"] += 1
                continue
            seen.add(text.casefold())
            cleaned.append(replace(draft, text=text))

        return cleaned

    async def label(self, drafts: list[Draft], report: GenerationReport) -> list[Query]:
        labels = await self.labels_for([draft.text for draft in drafts])
        queries = []
        for draft, labeled in zip(drafts, labels, strict=True):
            query = self.to_query(draft, labeled, report)
            if query:
                queries.append(query)
        return queries

    def to_query(
        self, draft: Draft, labeled: set[str] | None, report: GenerationReport
    ) -> Query | None:
        if labeled is None:
            report.dropped["not labeled"] += 1
            return None
        if draft.tool is None and labeled:
            report.dropped["labeler wants a tool"] += 1
            return None
        if draft.tool is not None and draft.tool not in labeled:
            report.dropped["labeler disagrees"] += 1
            return None
        return Query(
            id=query_id(draft.kind, draft.tool, draft.text),
            text=draft.text,
            tool=draft.tool,
            accepted=sorted(labeled),
            kind=draft.kind,
        )

    async def labels_for(self, texts: list[str]) -> list[set[str] | None]:
        starts = range(0, len(texts), LABEL_BATCH)
        batches = [texts[start : start + LABEL_BATCH] for start in starts]
        results = await asyncio.gather(*(self.label_batch(batch) for batch in batches))
        return [labels for batch in results for labels in batch]

    async def label_batch(self, texts: list[str]) -> list[set[str] | None]:
        messages = "\n".join(f"{number}. {text}" for number, text in enumerate(texts, 1))
        prompt = prompts.LABELS.format(catalog=self.catalog, messages=messages)
        answer = await self.ask(Labels, prompt, "label")
        by_number = {label.number: label.tools for label in answer.labels}
        labels: list[set[str] | None] = []
        for number in range(1, len(texts) + 1):
            if number not in by_number:
                labels.append(None)
                continue
            keys = {self.toolset.key_for(name) for name in by_number[number]}
            labels.append({key for key in keys if key})
        return labels
