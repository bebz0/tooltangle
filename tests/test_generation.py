from fakes import FakeChatModel, fake_generator, requested_functions

from tooltangle.dataset import Dataset
from tooltangle.generation import QueryGenerator, plan_generation
from tooltangle.models import ModelClient, UsageLog
from tooltangle.settings import Settings
from tooltangle.toolset import Toolset, ToolSpec


def make_toolset(*extra: ToolSpec) -> Toolset:
    return Toolset(
        [
            ToolSpec("notes", "search_notes", "Search the user's notes by keyword."),
            ToolSpec("notes", "read_note", "Read one note by its title."),
            ToolSpec("files", "search_files", "Search files on disk by glob pattern."),
            *extra,
        ]
    )


def make_generator(toolset: Toolset) -> tuple[QueryGenerator, FakeChatModel]:
    model = FakeChatModel(responder=fake_generator)
    settings = Settings(queries_per_tool=6, contrast_pairs=2, queries_per_contrast=4)
    client = ModelClient("fake:generator", UsageLog(), requests_per_minute=60_000, model=model)
    return QueryGenerator(toolset, client, settings), model


async def test_builds_a_labeled_dataset():
    toolset = make_toolset()
    generator, _ = make_generator(toolset)
    dataset, report = await generator.build(plan_generation(toolset, None))

    by_kind = {
        kind: [query for query in dataset.queries if query.kind == kind]
        for kind in ("plain", "contrast", "no_tool")
    }
    assert {query.tool for query in by_kind["plain"]} == set(toolset.keys)
    assert all(query.accepted == [query.tool] for query in by_kind["plain"])
    assert all(query.accepted == [] for query in by_kind["no_tool"])
    assert report.pairs[0] == ("notes.search_notes", "files.search_files")
    assert {frozenset((q.tool, q.against)) for q in by_kind["contrast"]} == {
        frozenset(pair) for pair in report.pairs
    }
    assert report.dropped["names the tool"] == 3
    assert report.dropped["labeler disagrees"] > 0
    for key in toolset.keys:
        splits = [query.split for query in by_kind["plain"] if query.tool == key]
        assert abs(splits.count("dev") - splits.count("holdout")) <= 1


async def test_second_run_has_nothing_to_do(tmp_path):
    toolset = make_toolset()
    generator, _ = make_generator(toolset)
    dataset, _ = await generator.build(plan_generation(toolset, None))
    dataset.save(tmp_path / "queries.jsonl")

    loaded = Dataset.load(tmp_path / "queries.jsonl")
    assert loaded.queries == dataset.queries
    assert plan_generation(toolset, loaded).is_empty


async def test_new_tool_only_generates_what_is_missing():
    toolset = make_toolset()
    generator, _ = make_generator(toolset)
    dataset, _ = await generator.build(plan_generation(toolset, None))

    bigger = make_toolset(ToolSpec("mail", "send_email", "Send an email to someone."))
    plan = plan_generation(bigger, dataset)
    assert plan.tools == ["mail.send_email"]
    assert plan.relabel and not plan.needs_no_tool

    generator, model = make_generator(bigger)
    updated, _ = await generator.build(plan)
    plain_prompts = [prompt for prompt in model.prompts if "For each of these tools" in prompt]
    assert [requested_functions(prompt) for prompt in plain_prompts] == [["send_email"]]
    assert any(query.tool == "mail.send_email" for query in updated.queries)
    old_splits = {query.id: query.split for query in dataset.queries}
    assert all(old_splits[q.id] == q.split for q in updated.queries if q.id in old_splits)
