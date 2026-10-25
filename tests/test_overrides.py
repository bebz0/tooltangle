import pytest
from langchain_core.tools import tool

from tooltangle import apply_overrides
from tooltangle.overrides import (
    Override,
    OverridesError,
    apply_to_toolset,
    fingerprint,
    load_overrides,
    save_overrides,
    stale_overrides,
)
from tooltangle.toolset import Toolset, ToolSpec


@tool
def git_diff(target: str) -> str:
    """Shows differences between branches or commits"""
    return ""


@tool
def search(query: str) -> str:
    """Search the web."""
    return ""


def test_round_trip(tmp_path):
    path = tmp_path / "overrides.yaml"
    overrides = {
        ("git", "git_diff"): Override("Shows differences between two branches or commits."),
        ("notes", "search"): Override("Search notes.\nNot for files on disk.", "0123456789ab"),
    }
    save_overrides(overrides, path)
    assert load_overrides(path) == overrides
    assert "git:\n  git_diff:" in path.read_text()


def test_apply_overrides_to_langchain_tools(tmp_path):
    path = tmp_path / "overrides.yaml"
    save_overrides({("git", "git_diff"): Override("Diff two commits.")}, path)
    patched = apply_overrides([git_diff], path)
    assert patched[0].description == "Diff two commits."
    assert git_diff.description == "Shows differences between branches or commits"


def test_a_rewrite_only_lands_on_the_tool_it_was_verified_for(tmp_path):
    path = tmp_path / "overrides.yaml"
    notes = Override("Search notes, not the web.", fingerprint("Search the user's notes."))
    save_overrides({("notes", "search"): notes}, path)
    assert apply_overrides([search], path)[0].description == "Search the web."

    toolset = Toolset(
        [
            ToolSpec("notes", "search", "Search the user's notes."),
            ToolSpec("web", "search", "Search the web."),
        ]
    )
    applied = apply_to_toolset(toolset, {("notes", "search"): notes})
    assert applied["notes.search"].description == "Search notes, not the web."
    changed = toolset.with_descriptions({"notes.search": "Search notes, now with tags."})
    assert apply_to_toolset(changed, {("notes", "search"): notes})["notes.search"].description == (
        "Search notes, now with tags."
    )
    assert stale_overrides(changed, {("notes", "search"): notes}) == ["notes.search"]


@pytest.mark.parametrize(
    "text",
    ["git:\n  git_diff:\n", 'git:\n  git_diff: ""\n', "git:\n  git_diff: Use this. Note: no\n"],
)
def test_bad_files_give_a_clear_error(tmp_path, text):
    path = tmp_path / "overrides.yaml"
    path.write_text(text)
    with pytest.raises(OverridesError, match=r"overrides\.yaml"):
        load_overrides(path)
