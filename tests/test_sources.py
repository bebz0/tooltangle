from tooltangle.sources import load_tools, parse_server, read_config, load_python_tools


def test_stdio_default_transport():
    server = parse_server("git", {"command": "uvx", "args": ["mcp-server-git"]})
    assert server.transport == "stdio"
    assert server.command == "uvx"
    assert server.args == ["mcp-server-git"]
    assert parse_server("remote", {"url": "https://example.com/mcp"}).transport == "http"


def test_brokenentries_skipped_not_fatal(write_config):
    path = write_config(
        {
            "ok": {"command": "uvx", "args": ["mcp-server-time"]},
            "off": {"command": "uvx", "disabled": True},
            "bad": {"type": "carrier-pigeon"},
        }
    )
    servers, skipped = read_config(path)
    assert [server.name for server in servers] == ["ok"]
    assert set(skipped) == {"off", "bad"}


async def test_loads_tools_from_real_server(write_config, notes_server, broken_server):
    path = write_config({"notes": notes_server, "broken": broken_server})
    loaded = await load_tools(str(path), timeout=30)

    assert loaded.toolset.keys == [
        "notes.search_notes",
        "notes.read_note",
        "notes.create_note",
        "notes.search_files",
    ]
    assert loaded.toolset["notes.create_note"].parameters["required"] == ["title", "body"]
    assert "API_TOKEN is not set" in loaded.failures["broken"]


def test_loads_python_tools(tmp_path):
    module = tmp_path / "agent_tools.py"
    module.write_text(
        "from langchain_core.tools import tool\n\n"
        "@tool\n"
        "def lookup_order(order_id: str) -> str:\n"
        '    """Find an order by its id."""\n'
        "    return ''\n\n"
        "TOOLS = [lookup_order]\n"
    )
    specs = load_python_tools(f"{module}:TOOLS")
    assert [(spec.key, spec.description) for spec in specs] == [
        ("python.lookup_order", "Find an order by its id.")
    ]
    assert "order_id" in specs[0].parameters["properties"]
