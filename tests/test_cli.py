from typer.testing import CliRunner

from tooltangle.cli import app

runner = CliRunner()


def test_tools_command(write_config, notes_server):
    path = write_config({"notes": notes_server})
    result = runner.invoke(app, ["tools", str(path), "--list"])
    assert result.exit_code == 0, result.stdout
    assert "notes 4" in result.stdout
    assert "notes.search_files" in result.stdout
