from dataclasses import dataclass
from statistics import median

from scipy.stats import binomtest

from tooltangle.dataset import Query
from tooltangle.metrics import Evaluation, evaluate, is_correct
from tooltangle.models import UsageLog
from tooltangle.runner import Pick

ALPHA = 0.05


@dataclass
class ModelResult:
    spec: str
    evaluation: Evaluation
    missed_rate: float
    needless_rate: float
    median_latency: float | None
    cost_per_thousand: float | None
    p_worse_than_best: float | None = None

    @property
    def significantly_worse(self) -> bool:
        return self.p_worse_than_best is not None and self.p_worse_than_best < ALPHA


def summarize(
    spec: str, queries: list[Query], picks: dict[str, Pick], usage: UsageLog
) -> ModelResult:
    evaluation = evaluate(spec, queries, picks)
    answered = [(query, picks[query.id]) for query in queries if query.id in picks]
    answered = [(query, pick) for query, pick in answered if not pick.error]

    needs_tool = [pick for query, pick in answered if query.accepted]
    no_tool = [pick for query, pick in answered if not query.accepted]
    missed = sum(pick.tool is None for pick in needs_tool)
    needless = sum(pick.tool is not None for pick in no_tool)

    latencies = [pick.latency for _, pick in answered if pick.latency]
    cost = None
    if answered:
        input_tokens = sum(pick.input_tokens for _, pick in answered)
        output_tokens = sum(pick.output_tokens for _, pick in answered)
        total = usage.cost(spec, input_tokens, output_tokens)
        cost = total / len(answered) * 1000 if total is not None else None

    return ModelResult(
        spec=spec,
        evaluation=evaluation,
        missed_rate=missed / len(needs_tool) if needs_tool else 0.0,
        needless_rate=needless / len(no_tool) if no_tool else 0.0,
        median_latency=median(latencies) if latencies else None,
        cost_per_thousand=cost,
    )


def worse_than(best: dict[str, Pick], other: dict[str, Pick], queries: list[Query]) -> float:
    best_only = other_only = 0
    for query in queries:
        first, second = best.get(query.id), other.get(query.id)
        if first is None or second is None or first.error or second.error:
            continue
        first_right, second_right = is_correct(query, first), is_correct(query, second)
        best_only += first_right and not second_right
        other_only += second_right and not first_right
    if best_only + other_only == 0:
        return 1.0
    pvalue = binomtest(best_only, best_only + other_only, 0.5, alternative="greater").pvalue
    return float(pvalue)


def rank(
    results: list[ModelResult], picks: dict[str, dict[str, Pick]], queries: list[Query]
) -> ModelResult | None:
    if not results:
        return None
    best = max(results, key=lambda result: result.evaluation.accuracy)
    for result in results:
        if result is not best:
            result.p_worse_than_best = worse_than(picks[best.spec], picks[result.spec], queries)

    candidates = [result for result in results if not result.significantly_worse]
    priced = [
        (result.cost_per_thousand, result)
        for result in candidates
        if result.cost_per_thousand is not None
    ]
    if not priced:
        return best
    return min(priced, key=lambda item: item[0])[1]
