from collections import Counter, defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from math import floor, log2

from pydantic import BaseModel, Field
from scipy.stats import binomtest

from tooltangle import prompts
from tooltangle.dataset import Dataset, Query
from tooltangle.findings import lookalike_pairs
from tooltangle.metrics import Evaluation, is_correct
from tooltangle.models import FallbackClient
from tooltangle.runner import Pick
from tooltangle.toolset import Toolset

ALPHA = 0.05
TOLERATED_REGRESSIONS = 1
EXAMPLES = 6
RELATED_TOOLS = 6
TOP_UP_ROUNDS = 2
TOP_UP_PER_SIDE = 8


class NewDescription(BaseModel):
    tool: str = Field(description="The tool name exactly as written in the list")
    description: str


class DescriptionFix(BaseModel):
    reasoning: str = Field(description="One or two sentences on what was ambiguous")
    descriptions: list[NewDescription]


@dataclass
class Verdict:
    pair: tuple[str, str]
    descriptions: dict[str, str]
    reasoning: str
    before_errors: int
    after_errors: int
    evaluated: int
    fixed: int
    broken: int
    p_value: float
    others: int
    others_fixed: int
    others_broken: int
    alpha: float = ALPHA
    needed: int = 0
    added: int = 0
    underpowered: bool = False

    @property
    def proven(self) -> bool:
        return self.p_value < self.alpha

    @property
    def accepted(self) -> bool:
        net_regressions = self.others_broken - self.others_fixed
        return self.proven and net_regressions <= TOLERATED_REGRESSIONS


def confused_pairs(evaluation: Evaluation, limit: int, minimum: int = 2) -> list[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()
    for confusion in evaluation.confusions:
        expected, picked = confusion.expected, confusion.picked
        if expected is None or picked is None or picked.startswith("?"):
            continue
        first, second = sorted((expected, picked))
        counts[(first, second)] += confusion.count
    return [pair for pair, count in counts.most_common() if count >= minimum][:limit]


def needed_fixes(alpha: float) -> int:
    return floor(log2(1 / alpha)) + 1


def count_errors(queries: list[Query], picks: dict[str, Pick]) -> int:
    return sum(
        not is_correct(query, picks[query.id])
        for query in queries
        if query.id in picks and not picks[query.id].error
    )


def improvement_p_value(fixed: int, broken: int) -> float:
    if fixed + broken == 0:
        return 1.0
    return float(binomtest(fixed, fixed + broken, 0.5, alternative="greater").pvalue)


def related_tools(
    pair: tuple[str, str], evaluation: Evaluation, toolset: Toolset, limit: int = RELATED_TOOLS
) -> list[str]:
    scores: defaultdict[str, float] = defaultdict(float)
    for confusion in evaluation.confusions:
        expected, picked = confusion.expected, confusion.picked
        if expected in pair and picked and picked not in pair and not picked.startswith("?"):
            scores[picked] += confusion.count
        if picked in pair and expected and expected not in pair:
            scores[expected] += confusion.count
    for first, second, similarity in lookalike_pairs(toolset, threshold=0.15, limit=100):
        if first in pair and second not in pair:
            scores[second] += similarity
        elif second in pair and first not in pair:
            scores[first] += similarity
    return sorted(scores, key=scores.__getitem__, reverse=True)[:limit]


class Fixer:
    def __init__(
        self,
        toolset: Toolset,
        dataset: Dataset,
        generator: FallbackClient,
        pick: Callable[[Toolset, list[Query]], Awaitable[dict[str, Pick]]],
        attempts: int = 2,
        top_up: Callable[[tuple[str, str]], Awaitable[list[Query]]] | None = None,
    ):
        self.toolset = toolset
        self.dataset = dataset
        self.generator = generator
        self.pick = pick
        self.attempts = attempts
        self.top_up = top_up

    def function(self, key: str) -> str:
        return self.toolset.function_names[key]

    def evidence(
        self, pair: tuple[str, str], picks: dict[str, Pick]
    ) -> tuple[list[str], list[str]]:
        failures, successes = [], []
        for query in self.dataset.for_split("dev"):
            if query.tool not in pair or query.id not in picks:
                continue
            pick = picks[query.id]
            picked = "no tool"
            if pick.tool is not None and pick.tool in self.toolset:
                picked = self.function(pick.tool)
            line = f'- "{query.text}" → should be `{self.function(query.tool)}`'
            if is_correct(query, pick):
                successes.append(line)
            elif pick.tool in pair:
                failures.append(f"{line}, picked `{picked}`")
        return failures[:EXAMPLES], successes[:EXAMPLES]

    async def propose(
        self, pair: tuple[str, str], picks: dict[str, Pick], last: Verdict | None
    ) -> DescriptionFix:
        failures, successes = self.evidence(pair, picks)
        feedback = ""
        if last is not None:
            feedback = prompts.FIX_FEEDBACK.format(
                fixed=last.fixed, broken=last.broken, others_broken=last.others_broken
            )
        prompt = prompts.FIX_DESCRIPTIONS.format(
            catalog=self.toolset.catalog(),
            first=self.function(pair[0]),
            second=self.function(pair[1]),
            failures="\n".join(failures) or "(none in the examples)",
            successes="\n".join(successes) or "(none in the examples)",
            feedback=feedback,
        )
        return await self.generator.ask(DescriptionFix, prompt, "fix")

    def descriptions_from(self, fix: DescriptionFix, pair: tuple[str, str]) -> dict[str, str]:
        descriptions = {}
        for item in fix.descriptions:
            key = self.toolset.key_for(item.tool.strip("` "))
            if key in pair and item.description.strip():
                descriptions[key] = " ".join(item.description.split())
        return descriptions

    async def try_pair(
        self, pair: tuple[str, str], evaluation: Evaluation, picks: dict[str, Pick]
    ) -> Verdict | None:
        alpha = ALPHA / self.attempts
        needed = needed_fixes(alpha)
        holdout = self.dataset.for_split("holdout")
        pair_queries = [query for query in holdout if query.tool in pair]
        related = set(related_tools(pair, evaluation, self.toolset))
        other_queries = [query for query in holdout if query.tool in related or query.tool is None]
        before = await self.pick(self.toolset, pair_queries + other_queries)

        added = 0
        for _ in range(TOP_UP_ROUNDS):
            if self.top_up is None or count_errors(pair_queries, before) >= needed:
                break
            fresh = await self.top_up(pair)
            if not fresh:
                break
            added += len(fresh)
            self.dataset.queries.extend(fresh)
            before |= await self.pick(self.toolset, fresh)
            picks = picks | before
            pair_queries += [query for query in fresh if query.split == "holdout"]

        errors = count_errors(pair_queries, before)
        if errors < needed:
            return Verdict(
                pair=pair,
                descriptions={},
                reasoning="",
                before_errors=errors,
                after_errors=errors,
                evaluated=len(pair_queries),
                fixed=0,
                broken=0,
                p_value=1.0,
                others=0,
                others_fixed=0,
                others_broken=0,
                alpha=alpha,
                needed=needed,
                added=added,
                underpowered=True,
            )

        last = None
        for _ in range(self.attempts):
            fix = await self.propose(pair, picks, last)
            descriptions = self.descriptions_from(fix, pair)
            if not descriptions:
                continue
            candidate = self.toolset.with_descriptions(descriptions)
            after = await self.pick(candidate, pair_queries + other_queries)
            last = compare(
                pair, descriptions, fix.reasoning, pair_queries, other_queries, before, after
            )
            last.alpha, last.needed, last.added = alpha, needed, added
            if last.accepted:
                self.toolset = candidate
                return last
        return last


def compare(
    pair: tuple[str, str],
    descriptions: dict[str, str],
    reasoning: str,
    pair_queries: list[Query],
    other_queries: list[Query],
    before: dict[str, Pick],
    after: dict[str, Pick],
) -> Verdict:
    def outcomes(queries: list[Query]) -> tuple[int, int, int, int, int]:
        fixed = broken = before_errors = after_errors = evaluated = 0
        for query in queries:
            old, new = before.get(query.id), after.get(query.id)
            if old is None or new is None or old.error or new.error:
                continue
            evaluated += 1
            was_right, is_right = is_correct(query, old), is_correct(query, new)
            before_errors += not was_right
            after_errors += not is_right
            fixed += is_right and not was_right
            broken += was_right and not is_right
        return fixed, broken, before_errors, after_errors, evaluated

    fixed, broken, before_errors, after_errors, evaluated = outcomes(pair_queries)
    others_fixed, others_broken, _, _, others = outcomes(other_queries)
    return Verdict(
        pair=pair,
        descriptions=descriptions,
        reasoning=reasoning,
        before_errors=before_errors,
        after_errors=after_errors,
        evaluated=evaluated,
        fixed=fixed,
        broken=broken,
        p_value=improvement_p_value(fixed, broken),
        others=others,
        others_fixed=others_fixed,
        others_broken=others_broken,
    )
