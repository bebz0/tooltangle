import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

SERVERS = Path(__file__).parent / "servers"

os.environ.pop("FORCE_COLOR", None)
os.environ["COLUMNS"] = "200"


@pytest.fixture
def notes_server() -> dict:
    return {"command": sys.executable, "args": [str(SERVERS / "notes_server.py")]}


@pytest.fixture
def broken_server() -> dict:
    return {"command": sys.executable, "args": [str(SERVERS / "broken_server.py")]}


@pytest.fixture
def write_config(tmp_path):
    def write(servers: dict, key: str = "mcpServers", name: str = "mcp.json") -> Path:
        path = tmp_path / name
        path.write_text(json.dumps({key: servers}))
        return path

    return write
