from collections.abc import Callable

from tooltangle.dataset import Dataset, Query
from tooltangle.generation import GenerationReport, QueryGenerator, plan_generation
from tooltangle.models import Cache, FallbackClient, ModelClient, UsageLog
from tooltangle.runner import Pick, Runner
from tooltangle.settings import Settings
from tooltangle.toolset import Toolset

STATE_GITIGNORE = "cache.sqlite*\nreport.json\n"


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

    def close(self) -> None:
        self.cache.close()

    def __enter__(self) -> "Workspace":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


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
