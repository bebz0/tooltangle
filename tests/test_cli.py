import json

import pytest
from fakes import confused_picker, fake_generator
from typer.testing import CliRunner

from tooltangle.cli import app

runner = CliRunner()


@pytest.fixture
def fake_models(use_fake_models):
    return use_fake_models(confused_picker, fake_generator)


def test_tools_command(write_config, notes_server):
    path = write_config({"notes": notes_server})
    result = runner.invoke(app, ["tools", str(path), "--list"])
    assert result.exit_code == 0, result.stdout
    assert "notes 4" in result.stdout
    assert "notes.search_files" in result.stdout


def test_check_end_to_end(project, write_config, notes_server, fake_models):
    config = write_config({"notes": notes_server})

    result = runner.invoke(app, ["check", str(config), "--yes"])
    assert result.exit_code == 1, result.stdout
    assert "notes.search_notes → notes.search_files" in result.stdout

    report = json.loads((project / ".tooltangle/report.json").read_text())
    assert report["tools"]["notes.search_notes"]["accuracy"] < 0.7
    assert (project / ".tooltangle/queries.jsonl").is_file()

    calls_before = sum(len(model.prompts) for model in fake_models)
    again = runner.invoke(app, ["check", str(config), "--yes"])
    assert again.exit_code == 1
    assert sum(len(model.prompts) for model in fake_models) == calls_before
    assert "cached" in again.stdout


def test_check_asks_before_spending(project, write_config, notes_server, fake_models):
    config = write_config({"notes": notes_server})
    result = runner.invoke(app, ["check", str(config)])
    assert result.exit_code == 2
    assert "pass --yes" in result.stdout
    assert all(not model.prompts for model in fake_models)
