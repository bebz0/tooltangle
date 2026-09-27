from tooltangle.toolset import Toolset, ToolSpec


def make_toolset() -> Toolset:
    return Toolset(
        [
            ToolSpec("files", "search", "Search files", {"properties": {"pattern": {}}}),
            ToolSpec("notes", "search", "Search notes"),
            ToolSpec("notes", "read_note", "Read a note"),
        ]
    )


def test_duplicate_names_get_server_prefix():
    toolset = make_toolset()
    assert toolset.function_names == {
        "files.search": "files_search",
        "notes.search": "notes_search",
        "notes.read_note": "read_note",
    }
    assert toolset.key_for("files_search") == "files.search"
    assert toolset.key_for("read_note") == "notes.read_note"
    assert toolset.key_for("made_up") is None


def test_fingerprint_follows_descriptions():
    toolset = make_toolset()
    changed = toolset.with_descriptions({"notes.read_note": "Open one note by title"})
    assert changed.fingerprint() != toolset.fingerprint()
    assert changed["notes.read_note"].description == "Open one note by title"
    assert toolset.with_descriptions({}).fingerprint() == toolset.fingerprint()
