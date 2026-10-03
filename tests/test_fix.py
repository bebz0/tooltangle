from fakes import confused_picker, fake_generator
from typer.testing import CliRunner

from tooltangle.cli import app
from tooltangle.fixer import DescriptionFix, NewDescription
from tooltangle.overrides import load_overrides

runner = CliRunner()

GOOD_FIX = "Search what the user jotted down in notes. For files on disk use search_files."
USELESS_FIX = "Search notes. Very useful tool for searching."


def rewriting(description: str):
    def respond(schema, prompt):
        if schema is DescriptionFix:
            return DescriptionFix(
                reasoning="The notes search sounded like a file search.",
                descriptions=[NewDescription(tool="search_notes", description=description)],
            )
        return fake_generator(schema, prompt)

    return respond


def setup(project, write_config, notes_server, use_fake_models, rewrite, **settings) -> str:
    settings = {"requests_per_minute": 600000, "queries_per_contrast": 8, **settings}
    (project / "tooltangle.toml").write_text(
        "".join(f"{key} = {value}\n" for key, value in settings.items())
    )
    use_fake_models(confused_picker, rewriting(rewrite))
    config = str(write_config({"notes": notes_server}))
    assert runner.invoke(app, ["check", config, "--yes"]).exit_code == 1
    return config


def test_fix_keeps_a_proven_rewrite(project, write_config, notes_server, use_fake_models):
    config = setup(project, write_config, notes_server, use_fake_models, GOOD_FIX)

    result = runner.invoke(app, ["fix", config, "--yes"])
    assert result.exit_code == 0, result.stdout
    assert "✓ accepted" in result.stdout
    assert load_overrides(project / "tooltangle.overrides.yaml") == {
        ("notes", "search_notes"): GOOD_FIX
    }

    after = runner.invoke(app, ["check", config, "--yes"])
    assert after.exit_code == 0, after.stdout
    assert "1 verified descriptions applied" in after.stdout


def test_fix_rejects_a_useless_rewrite(project, write_config, notes_server, use_fake_models):
    config = setup(project, write_config, notes_server, use_fake_models, USELESS_FIX)

    result = runner.invoke(app, ["fix", config, "--yes", "--attempts", "1"])
    assert result.exit_code == 0, result.stdout
    assert "not proven better" in result.stdout
    assert not (project / "tooltangle.overrides.yaml").exists()
