from langchain_core.tools import tool

from tooltangle import apply_overrides
from tooltangle.overrides import load_overrides, save_overrides


@tool
def git_diff(target: str) -> str:
    """Shows differences between branches or commits"""
    return ""


def test_round_trip(tmp_path):
    path = tmp_path / "overrides.yaml"
    overrides = {
        ("git", "git_diff"): "Shows differences between two branches or commits.",
        ("notes", "search"): "Search notes.\nNot for files on disk.",
    }
    save_overrides(overrides, path)
    assert load_overrides(path) == overrides
    assert "git:\n  git_diff:" in path.read_text()


def test_apply_overrides_to_langchain_tools(tmp_path):
    path = tmp_path / "overrides.yaml"
    save_overrides({("git", "git_diff"): "Diff two commits."}, path)
    patched = apply_overrides([git_diff], path)
    assert patched[0].description == "Diff two commits."
    assert git_diff.description == "Shows differences between branches or commits"
