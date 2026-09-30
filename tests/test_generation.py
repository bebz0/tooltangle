from fakes import FakeChatModel, fake_generator

from tooltangle.generation import QueryGenerator
from tooltangle.models import ModelClient, UsageLog
from tooltangle.settings import Settings
from tooltangle.toolset import Toolset, ToolSpec


def make_toolset() -> Toolset:
    return Toolset(
        [
            ToolSpec("notes", "search_notes", "Search the user's notes by keyword."),
            ToolSpec("notes", "read_note", "Read one note by its title."),
            ToolSpec("files", "search_files", "Search files on disk by glob pattern."),
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
    dataset, report = await generator.build()

    by_kind = {
        kind: [query for query in dataset.queries if query.kind == kind]
        for kind in ("plain", "no_tool")
    }
    assert {query.tool for query in by_kind["plain"]} == set(toolset.keys)
    assert all(query.accepted == [query.tool] for query in by_kind["plain"])
    assert all(query.accepted == [] for query in by_kind["no_tool"])
    assert report.dropped["names the tool"] == 3
    assert report.dropped["labeler disagrees"] > 0
    for key in toolset.keys:
        splits = [query.split for query in by_kind["plain"] if query.tool == key]
        assert abs(splits.count("dev") - splits.count("holdout")) <= 1
