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


def test_function_names_never_collide():
    toolset = Toolset(
        [
            ToolSpec("a", "get.user", "Dotted name"),
            ToolSpec("b", "get_user", "Same name once cleaned"),
            ToolSpec("notes", "search", "Search notes"),
            ToolSpec("files", "search", "Search files"),
            ToolSpec("x", "notes_search", "Looks like a prefixed name"),
            ToolSpec("long", "t" * 70 + "_one", "Cut to the same 64 characters"),
            ToolSpec("long", "t" * 70 + "_two", "Cut to the same 64 characters"),
        ]
    )
    names = toolset.function_names
    assert names["a.get.user"] == "a_get_user"
    assert names["b.get_user"] == "b_get_user"
    assert names["x.notes_search"] == "x_notes_search"
    assert len(set(names.values())) == len(names)
    assert all(len(name) <= 64 for name in names.values())
    assert all(toolset.key_for(name) == key for key, name in names.items())


def test_fingerprint_follows_descriptions():
    toolset = make_toolset()
    changed = toolset.with_descriptions({"notes.read_note": "Open one note by title"})
    assert changed.fingerprint() != toolset.fingerprint()
    assert changed["notes.read_note"].description == "Open one note by title"
    assert toolset.with_descriptions({}).fingerprint() == toolset.fingerprint()
