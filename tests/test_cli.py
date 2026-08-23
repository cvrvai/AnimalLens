"""
Unit tests for Typer CLI commands.
"""
from typer.testing import CliRunner
from animallens.cli.main import app

runner = CliRunner()


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "AnimalLens" in result.output


def test_cli_doctor():
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "Diagnostic Summary" in result.output


def test_cli_species_list():
    result = runner.invoke(app, ["species", "list"], env={"COLUMNS": "160"})
    assert result.exit_code == 0
    assert "Redclaw Crayfish" in result.output


def test_cli_models_list():
    result = runner.invoke(app, ["models"], env={"COLUMNS": "160"})
    assert result.exit_code == 0
    assert "redclaw-behavior-v1" in result.output


def test_cli_models_pull():
    result = runner.invoke(app, ["pull", "redclaw-behavior-v1"], env={"COLUMNS": "160"})
    assert result.exit_code == 0
    assert "Successfully installed model" in result.output


def test_cli_ollama_list():
    result = runner.invoke(app, ["ollama", "list"], env={"COLUMNS": "160"})
    assert result.exit_code == 0
