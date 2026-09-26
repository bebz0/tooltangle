from tooltangle.sources import parse_server, read_config

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
