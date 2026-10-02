import json
import os
import sys
from pathlib import Path

import pytest
from fakes import FakeChatModel

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


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "tooltangle.toml").write_text("requests_per_minute = 600000\n")
    return tmp_path


@pytest.fixture
def use_fake_models(monkeypatch):
    def install(picker, responder) -> list[FakeChatModel]:
        models = []

        def create(spec, **options):
            model = FakeChatModel(responder=responder, picker=picker)
            models.append(model)
            return model

        monkeypatch.setattr("tooltangle.models.create_model", create)
        return models

    return install
