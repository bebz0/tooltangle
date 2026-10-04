import asyncio
import hashlib
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from math import ceil

from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from tooltangle import prompts
from tooltangle.dataset import Dataset, DatasetInfo, Kind, Query, assign_splits, query_id
from tooltangle.findings import lookalike_pairs
from tooltangle.models import FallbackClient, ModelClient
from tooltangle.settings import Settings
from tooltangle.toolset import Toolset

PLAIN_BATCH = 4
LABEL_BATCH = 50
NO_TOOL_BATCH = 25
NO_TOOL_SHARE = 0.1
PARROT_THRESHOLD = 0.6
DUPLICATE_THRESHOLD = 0.85


def similarity_to_references(texts: list[str], references: list[str]) -> list[float]:
    if not texts:
        return []
    vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True)
    matrix = vectorizer.fit_transform(texts + references)
    text_vectors, reference_vectors = matrix[: len(texts)], matrix[len(texts) :]
    return [
        float(cosine_similarity(text_vectors[i], reference_vectors[i])[0, 0])
        for i in range(len(texts))
    ]


class Requests(BaseModel):
    requests: list[str] = Field(description="The user messages, one per item")


class ToolRequests(BaseModel):
    tool: str = Field(description="The tool name exactly as written in the list")
    requests: list[str] = Field(description="The user messages for this tool")


class RequestsByTool(BaseModel):
    tools: list[ToolRequests]


class ContrastRequests(BaseModel):
    first: list[str] = Field(description="Messages where the first tool is the best first step")
    second: list[str] = Field(description="Messages where the second tool is the best first step")


class ToolPair(BaseModel):
    first: str
    second: str


class ToolPairs(BaseModel):
    pairs: list[ToolPair]


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
    against: str | None = None


@dataclass
class GenerationPlan:
    kept: list[Query]
    tools: list[str]
    needs_pairs: bool
    needs_no_tool: bool
    relabel: bool
    generators: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not (self.tools or self.needs_pairs or self.needs_no_tool or self.relabel)

    def estimated_queries(self, settings: Settings, tool_count: int) -> int:
        queries = len(self.tools) * settings.queries_per_tool
        if self.needs_pairs:
            queries += settings.contrast_pairs * 2 * settings.queries_per_contrast
        if self.needs_no_tool:
            queries += no_tool_count(settings, tool_count)
        return queries

    def estimated_calls(self, settings: Settings, tool_count: int) -> int:
        calls = ceil(len(self.tools) / PLAIN_BATCH)
        if self.needs_pairs:
            calls += 1 + settings.contrast_pairs
        if self.needs_no_tool:
            calls += ceil(no_tool_count(settings, tool_count) / NO_TOOL_BATCH)
        relabeled = len(self.kept) if self.relabel else 0
        new_queries = self.estimated_queries(settings, tool_count)
        return calls + ceil((new_queries + relabeled) / LABEL_BATCH)


@dataclass
class GenerationReport:
    generated: int = 0
    kept: int = 0
    dropped: Counter[str] = field(default_factory=Counter)
    pairs: list[tuple[str, str]] = field(default_factory=list)


LABELING_VERSION = hashlib.sha256(prompts.LABELS.encode()).hexdigest()[:12]


def no_tool_count(settings: Settings, tool_count: int) -> int:
    return max(6, round(NO_TOOL_SHARE * tool_count * settings.queries_per_tool))


def plan_generation(toolset: Toolset, existing: Dataset | None) -> GenerationPlan:
    if existing is None:
        return GenerationPlan([], toolset.keys, True, True, False)

    kept = []
    for query in existing.queries:
        if query.tool is not None and query.tool not in toolset:
            continue
        accepted = [key for key in query.accepted if key in toolset]
        kept.append(query.model_copy(update={"accepted": accepted}))

    covered = {query.tool for query in kept if query.kind == "plain"}
    new_tools = [key for key in toolset.keys if key not in covered]
    return GenerationPlan(
        kept=kept,
        tools=new_tools,
        needs_pairs=bool(new_tools) or not any(query.kind == "contrast" for query in kept),
        needs_no_tool=not any(query.kind == "no_tool" for query in kept),
        relabel=bool(set(toolset.keys) - set(existing.info.labeled_tools))
        or existing.info.labeling != LABELING_VERSION,
        generators=existing.info.generators,
    )


class QueryGenerator:
    def __init__(
        self,
        toolset: Toolset,
        client: ModelClient | FallbackClient,
        settings: Settings,
        on_call: Callable[[], None] = lambda: None,
    ):
        self.toolset = toolset
        self.client = client
        self.settings = settings
        self.on_call = on_call
        self.catalog = toolset.catalog()
        self.semaphore = asyncio.Semaphore(settings.concurrency)

    async def build(self, plan: GenerationPlan) -> tuple[Dataset, GenerationReport]:
        report = GenerationReport()
        covered_pairs = {
            frozenset((query.tool, query.against)) for query in plan.kept if query.against
        }

        if plan.needs_pairs:
            pairs = await self.confusable_pairs()
            report.pairs = [pair for pair in pairs if frozenset(pair) not in covered_pairs]

        batches = [
            plan.tools[start : start + PLAIN_BATCH]
            for start in range(0, len(plan.tools), PLAIN_BATCH)
        ]
        jobs = [self.plain_queries(batch) for batch in batches]
        jobs += [self.contrast_queries(first, second) for first, second in report.pairs]
        if plan.needs_no_tool:
            total = no_tool_count(self.settings, len(self.toolset))
            for start in range(0, total, NO_TOOL_BATCH):
                jobs.append(self.no_tool_queries(min(NO_TOOL_BATCH, total - start)))

        drafts = [draft for batch in await asyncio.gather(*jobs) for draft in batch]
        report.generated = len(drafts)
        drafts = self.clean(drafts, {query.text.casefold() for query in plan.kept}, report)

        kept = plan.kept
        if plan.relabel:
            kept = await self.relabel(kept, report)
        new_queries = await self.label(drafts, report)
        assign_splits(new_queries, kept)

        order: dict[str | None, int] = {key: index for index, key in enumerate(self.toolset.keys)}
        queries = sorted(
            kept + new_queries,
            key=lambda query: (order.get(query.tool, len(order)), query.kind, query.id),
        )
        report.kept = len(queries)
        info = DatasetInfo(
            generators=sorted(set(self.client.used) | set(plan.generators)),
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

    async def contrast_queries(self, first: str, second: str) -> list[Draft]:
        count = self.settings.queries_per_contrast
        prompt = prompts.CONTRAST_QUERIES.format(
            catalog=self.catalog,
            count=count,
            first=self.function(first),
            second=self.function(second),
            language=self.settings.language,
        )
        answer = await self.ask(ContrastRequests, prompt, "generate")
        return [Draft(text, first, "contrast", second) for text in answer.first[:count]] + [
            Draft(text, second, "contrast", first) for text in answer.second[:count]
        ]

    async def more_contrast(
        self, pair: tuple[str, str], existing: list[Query], count: int
    ) -> list[Query]:
        first, second = pair
        shown = [f"- {query.text}" for query in existing if query.tool in pair][:40]
        prompt = prompts.MORE_CONTRAST_QUERIES.format(
            catalog=self.catalog,
            count=count,
            first=self.function(first),
            second=self.function(second),
            existing="\n".join(shown) or "(none yet)",
            language=self.settings.language,
        )
        answer = await self.ask(ContrastRequests, prompt, "generate")
        drafts = [Draft(text, first, "contrast", second) for text in answer.first[:count]]
        drafts += [Draft(text, second, "contrast", first) for text in answer.second[:count]]

        report = GenerationReport()
        drafts = self.clean(drafts, {query.text.casefold() for query in existing}, report)
        queries = await self.label(drafts, report)
        assign_splits(queries, existing)
        return queries

    async def no_tool_queries(self, count: int) -> list[Draft]:
        prompt = prompts.NO_TOOL_QUERIES.format(
            catalog=self.catalog, count=count, language=self.settings.language
        )
        answer = await self.ask(Requests, prompt, "generate")
        return [Draft(text, None, "no_tool") for text in answer.requests[:count]]

    async def confusable_pairs(self) -> list[tuple[str, str]]:
        limit = self.settings.contrast_pairs
        prompt = prompts.CONFUSABLE_PAIRS.format(catalog=self.catalog, limit=limit)
        answer = await self.ask(ToolPairs, prompt, "generate")

        pairs = []
        for pair in answer.pairs:
            first, second = self.toolset.key_for(pair.first), self.toolset.key_for(pair.second)
            if first and second and first != second:
                pairs.append((first, second))
        pairs += [(first, second) for first, second, _ in lookalike_pairs(self.toolset)]

        unique: dict[frozenset[str], tuple[str, str]] = {}
        for first, second in pairs:
            unique.setdefault(frozenset((first, second)), (first, second))
        return list(unique.values())[:limit]

    def clean(self, drafts: list[Draft], seen: set[str], report: GenerationReport) -> list[Draft]:
        seen = set(seen)
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

        with_tool = [draft for draft in cleaned if draft.tool]
        scores = similarity_to_references(
            [draft.text for draft in with_tool],
            [self.toolset[draft.tool].description for draft in with_tool if draft.tool],
        )
        parrots = {
            id(draft)
            for draft, score in zip(with_tool, scores, strict=True)
            if score >= PARROT_THRESHOLD
        }
        if parrots:
            report.dropped["copies the description"] += len(parrots)
        cleaned = [draft for draft in cleaned if id(draft) not in parrots]

        return self.drop_near_duplicates(cleaned, report)

    def drop_near_duplicates(self, drafts: list[Draft], report: GenerationReport) -> list[Draft]:
        groups: dict[tuple[str | None, Kind], list[Draft]] = {}
        for draft in drafts:
            groups.setdefault((draft.tool, draft.kind), []).append(draft)

        kept = []
        for group in groups.values():
            if len(group) < 2:
                kept.extend(group)
                continue
            matrix = TfidfVectorizer().fit_transform([draft.text for draft in group])
            scores = cosine_similarity(matrix)
            chosen: list[int] = []
            for index in range(len(group)):
                if all(scores[index, other] < DUPLICATE_THRESHOLD for other in chosen):
                    chosen.append(index)
                else:
                    report.dropped["near duplicate"] += 1
            kept.extend(group[index] for index in chosen)
        return kept

    async def label(self, drafts: list[Draft], report: GenerationReport) -> list[Query]:
        labels = await self.labels_for([draft.text for draft in drafts])
        queries = []
        for draft, labeled in zip(drafts, labels, strict=True):
            query = self.to_query(draft, labeled, report)
            if query:
                queries.append(query)
        return queries

    async def relabel(self, queries: list[Query], report: GenerationReport) -> list[Query]:
        drafts = [Draft(query.text, query.tool, query.kind, query.against) for query in queries]
        labels = await self.labels_for([draft.text for draft in drafts])
        relabeled = []
        for query, draft, labeled in zip(queries, drafts, labels, strict=True):
            fresh = self.to_query(draft, labeled, report)
            if fresh:
                relabeled.append(fresh.model_copy(update={"split": query.split}))
        return relabeled

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
            against=draft.against,
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
