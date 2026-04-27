"""Tests for spanish_tts.cli module (Click commands)."""

import pytest
from click.testing import CliRunner

from spanish_tts.cli import cli


@pytest.fixture
def runner():
    return CliRunner()


class TestCLI:
    def test_help(self, runner):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Curated Spanish TTS voices" in result.output

    def test_version(self, runner):
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0


class TestListCommand:
    def test_list_shows_voices(self, runner):
        result = runner.invoke(cli, ["list"])
        assert result.exit_code == 0
        assert "NAME" in result.output
        assert "TYPE" in result.output

    def test_voices_alias(self, runner):
        result = runner.invoke(cli, ["voices"])
        assert result.exit_code == 0
        assert "NAME" in result.output


class TestSayCommand:
    def test_say_unknown_voice(self, runner):
        result = runner.invoke(cli, ["say", "test", "--voice", "nonexistent_voice_xyz"])
        assert result.exit_code != 0
        assert "not found" in result.output

    def test_say_help(self, runner):
        result = runner.invoke(cli, ["say", "--help"])
        assert result.exit_code == 0
        assert "--voice" in result.output
        assert "--speed" in result.output
        assert "--play" in result.output
        assert "--stream" in result.output


class TestAddRefCommand:
    def test_add_ref_help(self, runner):
        result = runner.invoke(cli, ["add-ref", "--help"])
        assert result.exit_code == 0
        assert "--accent" in result.output
        assert "--gender" in result.output
        assert "--license" in result.output


class TestAddDesignCommand:
    def test_add_design_help(self, runner):
        result = runner.invoke(cli, ["add-design", "--help"])
        assert result.exit_code == 0
        assert "--gender" in result.output


class TestRemoveCommand:
    def test_remove_nonexistent(self, runner):
        result = runner.invoke(cli, ["remove", "nonexistent_voice_xyz"])
        assert result.exit_code != 0
        assert "not found" in result.output


class TestDemoCommand:
    def test_demo_help(self, runner):
        result = runner.invoke(cli, ["demo", "--help"])
        assert result.exit_code == 0
        assert "--output-dir" in result.output
