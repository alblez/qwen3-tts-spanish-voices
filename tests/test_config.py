"""Tests for spanish_tts.config module."""

import logging

import pytest
import yaml

from spanish_tts.config import (
    VOICES_FILENAME,
    add_voice,
    get_defaults,
    get_voice,
    list_voices,
    load_voices,
    save_voices,
)


@pytest.fixture
def tmp_voices_file(tmp_path):
    """Create a temporary voices.yaml for testing."""
    data = {
        "defaults": {"language": "Spanish", "speed": 1.0, "output_dir": "~/tts-output/test"},
        "voices": {
            "test_clone": {
                "type": "clone",
                "ref_audio": "/tmp/test.wav",
                "ref_text": "Hola mundo",
                "accent": "mexico",
                "gender": "male",
                "language": "Spanish",
            },
            "test_design": {
                "type": "design",
                "instruct": "A calm male voice",
                "gender": "male",
                "language": "Spanish",
            },
        },
    }
    voices_file = tmp_path / VOICES_FILENAME
    with open(voices_file, "w") as f:
        yaml.dump(data, f)
    return voices_file


class TestConstants:
    def test_voices_filename(self):
        assert VOICES_FILENAME == "voices.yaml"


class TestLoadVoices:
    def test_load_from_file(self, tmp_voices_file):
        data = load_voices(tmp_voices_file)
        assert "voices" in data
        assert "defaults" in data
        assert len(data["voices"]) == 2

    def test_load_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_voices(tmp_path / "nonexistent.yaml")


class TestSaveVoices:
    def test_save_and_reload(self, tmp_path):
        voices_file = tmp_path / VOICES_FILENAME
        data = {"voices": {"v1": {"type": "design", "instruct": "test"}}}
        save_voices(data, voices_file)

        assert voices_file.exists()
        reloaded = load_voices(voices_file)
        assert reloaded["voices"]["v1"]["instruct"] == "test"

    def test_creates_parent_dirs(self, tmp_path):
        deep_path = tmp_path / "a" / "b" / VOICES_FILENAME
        save_voices({"voices": {}}, deep_path)
        assert deep_path.exists()


class TestGetVoice:
    def test_existing_voice(self, tmp_voices_file):
        voice = get_voice("test_clone", tmp_voices_file)
        assert voice is not None
        assert voice["type"] == "clone"
        assert voice["accent"] == "mexico"

    def test_missing_voice(self, tmp_voices_file):
        voice = get_voice("nonexistent", tmp_voices_file)
        assert voice is None


class TestListVoices:
    def test_list_all(self, tmp_voices_file):
        voices = list_voices(tmp_voices_file)
        assert len(voices) == 2
        assert "test_clone" in voices
        assert "test_design" in voices


class TestAddVoice:
    def test_add_new_voice(self, tmp_voices_file):
        add_voice("new_voice", {"type": "design", "instruct": "A new voice"}, tmp_voices_file)
        voice = get_voice("new_voice", tmp_voices_file)
        assert voice is not None
        assert voice["instruct"] == "A new voice"

    def test_overwrite_existing(self, tmp_voices_file):
        add_voice("test_clone", {"type": "design", "instruct": "replaced"}, tmp_voices_file)
        voice = get_voice("test_clone", tmp_voices_file)
        assert voice["type"] == "design"

    def test_overwrite_same_type_emits_warning(self, tmp_voices_file, caplog):
        """Overwriting a voice with the same type emits a WARNING."""
        with caplog.at_level(logging.WARNING, logger="spanish-tts.config"):
            add_voice(
                "test_clone",
                {"type": "clone", "ref_audio": "/tmp/new.wav"},
                tmp_voices_file,
            )
        assert len(caplog.records) == 1
        rec = caplog.records[0]
        assert rec.levelno == logging.WARNING
        assert "test_clone" in rec.message
        assert "overwritten" in rec.message.lower()
        assert "CHANGES" not in rec.message  # must not route through the type-change branch
        # Verify the overwrite actually persisted
        updated = get_voice("test_clone", tmp_voices_file)
        assert updated["ref_audio"] == "/tmp/new.wav"

    def test_overwrite_type_change_emits_loud_warning(self, tmp_voices_file, caplog):
        """Overwriting a voice with a different type emits a WARNING flagging the type change."""
        with caplog.at_level(logging.WARNING, logger="spanish-tts.config"):
            add_voice(
                "test_design",
                {"type": "clone", "ref_audio": "/tmp/new.wav"},
                tmp_voices_file,
            )
        assert len(caplog.records) == 1
        rec = caplog.records[0]
        assert rec.levelno == logging.WARNING
        assert "test_design" in rec.message
        assert "design" in rec.message
        assert "clone" in rec.message

    def test_allow_overwrite_false_raises(self, tmp_voices_file):
        """allow_overwrite=False raises ValueError when voice already exists."""
        with pytest.raises(ValueError, match="test_clone"):
            add_voice(
                "test_clone",
                {"type": "clone", "ref_audio": "/tmp/new.wav"},
                tmp_voices_file,
                allow_overwrite=False,
            )

    def test_allow_overwrite_false_new_voice_ok(self, tmp_voices_file):
        """allow_overwrite=False does NOT raise when adding a brand-new voice."""
        add_voice(
            "brand_new",
            {"type": "design", "instruct": "novel"},
            tmp_voices_file,
            allow_overwrite=False,
        )
        voice = get_voice("brand_new", tmp_voices_file)
        assert voice is not None
        assert voice["instruct"] == "novel"

    def test_allow_overwrite_false_type_change_also_raises(self, tmp_voices_file):
        """allow_overwrite=False raises ValueError even when types differ."""
        with pytest.raises(ValueError, match="test_design"):
            add_voice(
                "test_design",
                {"type": "clone", "ref_audio": "/tmp/new.wav"},
                tmp_voices_file,
                allow_overwrite=False,
            )

    def test_add_new_voice_emits_no_warning(self, tmp_voices_file, caplog):
        """Adding a brand-new voice emits no warnings."""
        with caplog.at_level(logging.WARNING, logger="spanish-tts.config"):
            add_voice("brand_new_2", {"type": "design", "instruct": "x"}, tmp_voices_file)
        assert len(caplog.records) == 0


class TestGetDefaults:
    def test_returns_defaults(self, tmp_voices_file):
        defaults = get_defaults(tmp_voices_file)
        assert defaults["language"] == "Spanish"
        assert defaults["speed"] == 1.0

    def test_fallback_defaults(self, tmp_path):
        voices_file = tmp_path / VOICES_FILENAME
        save_voices({"voices": {}}, voices_file)
        defaults = get_defaults(voices_file)
        assert "language" in defaults
