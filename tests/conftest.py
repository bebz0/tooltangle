import json
from pathlib import Path
import pytest

@pytest.fixture
def write_config(tmp_path):
    def write(servers: dict, key: str = "mcpServers", name: str = "mcp.json") -> Path:
        path = tmp_path / name
        path.write_text(json.dumps({key: servers}))
        return path
    return write
