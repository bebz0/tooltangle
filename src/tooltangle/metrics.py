from collections import Counter, defaultdict
from dataclasses import dataclass, field

from scipy.stats import binomtest

from tooltangle.dataset import Dataset, Query
from tooltangle.runner import Pick
from tooltangle.toolset import Toolset

Pair = tuple[str | None, str | None]


def allow_looking_first(dataset: Dataset, toolset: Toolset) -> Dataset:
    reads = [tool for tool in toolset if tool.read_only is True]
    writes = {tool.key for tool in toolset if tool.read_only is False}
    queries = []
    for query in dataset.queries:
        if query.accepted and all(key in writes for key in query.accepted):
            servers = {toolset[key].server for key in query.accepted}
            extra = [
                tool.key
                for tool in reads
                if tool.server in servers and tool.key not in query.accepted
            ]
            if extra:
                query = query.model_copy(
                    update={"accepted": [*query.accepted, *extra], "look_first": extra}
                )
        queries.append(query)
    return dataset.model_copy(update={"queries": queries})


def is_correct(query: Query, pick: Pick) -> bool:
    if query.accepted:
        return pick.tool in query.accepted
    return pick.tool is None


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return (0.0, 1.0)
    interval = binomtest(successes, total).proportion_ci(method="wilson")
    return (interval.low, interval.high)


@dataclass
class ToolScore:
    key: str
    total: int = 0
    correct: int = 0
    picked: int = 0
    picked_wrongly: int = 0

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @property
    def precision(self) -> float | None:
        if not self.picked:
            return None
        return 1 - self.picked_wrongly / self.picked


@dataclass
class Confusion:
    expected: str | None
    picked: str | None
    count: int
    total: int
    examples: list[str] = field(default_factory=list)

    @property
    def rate(self) -> float:
        return self.count / self.total


@dataclass
class Evaluation:
    model: str
    total: int
    correct: int
    failed: int
    tools: dict[str, ToolScore]
    confusions: list[Confusion]
    no_tool_total: int
    no_tool_correct: int
    first_error: str | None = None
    looked_first: list[Confusion] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @property
    def interval(self) -> tuple[float, float]:
        return wilson_interval(self.correct, self.total)


def evaluate(model: str, queries: list[Query], picks: dict[str, Pick]) -> Evaluation:
    tools: dict[str, ToolScore] = {}
    wrong: Counter[Pair] = Counter()
    looked: Counter[Pair] = Counter()
    examples: dict[Pair, list[str]] = defaultdict(list)
    expected_totals: Counter[str | None] = Counter()
    total = correct = failed = no_tool_total = no_tool_correct = 0
    first_error = None

    for query in queries:
        pick = picks.get(query.id)
        if pick is None:
            continue
        if pick.error:
            failed += 1
            first_error = first_error or pick.error
            continue

        total += 1
        expected_totals[query.tool] += 1
        right = is_correct(query, pick)
        correct += right

        if query.tool is None:
            no_tool_total += 1
            no_tool_correct += right
        else:
            score = tools.setdefault(query.tool, ToolScore(query.tool))
            score.total += 1
            score.correct += right

        if pick.tool is not None and not pick.tool.startswith("?"):
            score = tools.setdefault(pick.tool, ToolScore(pick.tool))
            score.picked += 1
            score.picked_wrongly += not right

        if not right:
            wrong[(query.tool, pick.tool)] += 1
            examples[(query.tool, pick.tool)].append(query.text)
        elif pick.tool in query.look_first:
            looked[(query.tool, pick.tool)] += 1
            examples[(query.tool, pick.tool)].append(query.text)

    confusions = [
        Confusion(expected, picked, count, expected_totals[expected], examples[(expected, picked)])
        for (expected, picked), count in wrong.most_common()
    ]
    looked_first = [
        Confusion(expected, picked, count, expected_totals[expected], examples[(expected, picked)])
        for (expected, picked), count in looked.most_common()
    ]
    return Evaluation(
        model=model,
        total=total,
        correct=correct,
        failed=failed,
        tools=tools,
        confusions=confusions,
        no_tool_total=no_tool_total,
        no_tool_correct=no_tool_correct,
        first_error=first_error,
        looked_first=looked_first,
    )
