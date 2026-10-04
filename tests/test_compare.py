from fakes import FakeChatModel, fake_generator, pick_by_phrase
from typer.testing import CliRunner

from tooltangle.cli import app

runner = CliRunner()


def sloppy(text: str, tools: list[dict]) -> str | None:
    picked = pick_by_phrase(text, tools)
    return "search_files" if picked == "search_notes" else picked


def setup(project, monkeypatch, write_config, notes_server, pickers: dict) -> str:
    (project / "tooltangle.toml").write_text(
        "requests_per_minute = 600000\n"
        "[prices]\n"
        '"fake:big" = [2.0, 8.0]\n'
        '"fake:small" = [0.1, 0.4]\n'
    )
    monkeypatch.setattr(
        "tooltangle.models.create_model",
        lambda spec, **options: FakeChatModel(responder=fake_generator, picker=pickers.get(spec)),
    )
    return str(write_config({"notes": notes_server}))


def test_cheaper_model_that_holds_up_wins(project, monkeypatch, write_config, notes_server):
    pickers = {"fake:big": pick_by_phrase, "fake:small": pick_by_phrase}
    config = setup(project, monkeypatch, write_config, notes_server, pickers)
    result = runner.invoke(app, ["compare", config, "-m", "fake:big", "-m", "fake:small", "--yes"])
    assert result.exit_code == 0, result.stdout
    assert "small is the cheapest model" in result.stdout


def test_significantly_worse_model_is_skipped(project, monkeypatch, write_config, notes_server):
    pickers = {"fake:big": pick_by_phrase, "fake:small": sloppy}
    config = setup(project, monkeypatch, write_config, notes_server, pickers)
    result = runner.invoke(app, ["compare", config, "-m", "fake:big", "-m", "fake:small", "--yes"])
    assert result.exit_code == 0, result.stdout
    assert "worse than the best" in result.stdout
    assert "big is the cheapest model" in result.stdout
