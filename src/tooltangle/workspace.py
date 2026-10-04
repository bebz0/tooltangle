from collections.abc import Callable
from dataclasses import dataclass, field

from tooltangle.dataset import Dataset, Query
from tooltangle.findings import estimate_definition_tokens
from tooltangle.generation import GenerationReport, QueryGenerator, plan_generation
from tooltangle.models import Cache, FallbackClient, ModelClient, UsageLog
from tooltangle.overrides import Overrides, load_overrides
from tooltangle.runner import Pick, Runner
from tooltangle.settings import Settings
from tooltangle.toolset import Toolset

STATE_GITIGNORE = "cache.sqlite*\nreport.html\nreport.json\n"


class Workspace:
    def __init__(self, settings: Settings, on_event: Callable[[str], None] | None = None):
        self.settings = settings
        self.on_event = on_event
        self.usage = UsageLog(settings.prices)
        settings.state_dir.mkdir(parents=True, exist_ok=True)
        gitignore = settings.state_dir / ".gitignore"
        if not gitignore.exists():
            gitignore.write_text(STATE_GITIGNORE)
        self.cache = Cache(settings.cache_path)

    def target(self, spec: str | None = None) -> ModelClient:
        return ModelClient(
            spec or self.settings.model,
            self.usage,
            self.cache,
            self.settings.requests_per_minute,
            **self.settings.model_options,
        )

    def generator(self) -> FallbackClient:
        specs = dict.fromkeys([self.settings.generator, *self.settings.generator_fallbacks])
        return FallbackClient(
            [
                ModelClient(
                    spec,
                    self.usage,
                    self.cache,
                    self.settings.requests_per_minute,
                    on_event=self.on_event,
                )
                for spec in specs
            ]
        )

    def dataset(self) -> Dataset | None:
        return Dataset.load(self.settings.dataset_path)

    def overrides(self) -> Overrides:
        return load_overrides(self.settings.overrides_file)

    def close(self) -> None:
        self.cache.close()

    def __enter__(self) -> "Workspace":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


PROMPT_OVERHEAD_TOKENS = 80


@dataclass
class Estimate:
    generator_calls: int = 0
    target_calls: dict[str, int] = field(default_factory=dict)
    tokens_per_call: int = 0

    @property
    def total_calls(self) -> int:
        return self.generator_calls + sum(self.target_calls.values())


def tokens_per_call(toolset: Toolset) -> int:
    return estimate_definition_tokens(toolset) + PROMPT_OVERHEAD_TOKENS


def estimate_check(
    workspace: Workspace, toolset: Toolset, tested: Toolset, specs: list[str]
) -> Estimate:
    settings = workspace.settings
    existing = workspace.dataset()
    plan = plan_generation(toolset, existing)
    estimate = Estimate(tokens_per_call=tokens_per_call(tested))
    if not plan.is_empty:
        estimate.generator_calls = plan.estimated_calls(settings, len(toolset))

    queries = plan.kept if existing else []
    new_queries = 0 if plan.is_empty else plan.estimated_queries(settings, len(toolset))
    for spec in specs:
        runner = Runner(workspace.target(spec), tested, settings.system_prompt)
        estimate.target_calls[spec] = len(runner.uncached(queries)) + new_queries
    return estimate


async def prepare_dataset(
    workspace: Workspace, toolset: Toolset, on_call: Callable[[], None] = lambda: None
) -> tuple[Dataset, GenerationReport | None]:
    existing = workspace.dataset()
    plan = plan_generation(toolset, existing)
    if existing is not None and plan.is_empty:
        return existing, None
    generator = QueryGenerator(toolset, workspace.generator(), workspace.settings, on_call)
    dataset, report = await generator.build(plan)
    dataset.save(workspace.settings.dataset_path)
    return dataset, report


async def pick_tools(
    workspace: Workspace,
    client: ModelClient,
    toolset: Toolset,
    queries: list[Query],
    on_pick: Callable[[], None] = lambda: None,
) -> dict[str, Pick]:
    settings = workspace.settings
    runner = Runner(client, toolset, settings.system_prompt, settings.concurrency, on_pick)
    return await runner.run(queries)
