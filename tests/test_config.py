"""Tests for spanish_tts.config module."""

import logging
import os
from pathlib import Path

import pytest
import yaml

from spanish_tts.config import (
    VOICES_FILENAME,
    _validate_voices_schema,  # private helper; tested directly to pin the contract
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
        with caplog.at_level(logging.WARNING, logger="spanish_tts.config"):
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
        with caplog.at_level(logging.WARNING, logger="spanish_tts.config"):
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
        with caplog.at_level(logging.WARNING, logger="spanish_tts.config"):
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


# ---------------------------------------------------------------------------
# U3-3: YAML resilience — load_voices, save_voices, schema, env guard
# ---------------------------------------------------------------------------


class TestLoadVoicesResilience:
    def test_corrupt_yaml_falls_back_to_presets(self, tmp_path, caplog):
        """Corrupt YAML logs error and returns bundled presets instead."""
        bad = tmp_path / VOICES_FILENAME
        bad.write_text("{invalid yaml: [}", encoding="utf-8")
        with caplog.at_level(logging.ERROR, logger="spanish_tts.config"):
            data = load_voices(bad)
        assert "Corrupt" in caplog.text or "corrupt" in caplog.text.lower()
        # Bundled presets must include the known neutral_male design voice.
        assert isinstance(data.get("voices"), dict)
        assert "neutral_male" in data["voices"], (
            "Bundled presets changed — update this test to name a known voice"
        )

    def test_empty_yaml_returns_empty_dict(self, tmp_path):
        """Empty YAML file (safe_load → None) is normalised to {}."""
        empty = tmp_path / VOICES_FILENAME
        empty.write_text("", encoding="utf-8")
        data = load_voices(empty)
        assert data == {}

    def test_non_dict_yaml_raises(self, tmp_path):
        """A YAML list at the top level raises ValueError."""
        bad = tmp_path / VOICES_FILENAME
        bad.write_text("- item1\n- item2\n", encoding="utf-8")
        with pytest.raises(ValueError, match="mapping"):
            load_voices(bad)

    def test_corrupt_file_not_overwritten(self, tmp_path):
        """Corrupt user file is NOT modified by load_voices — user must recover manually."""
        bad = tmp_path / VOICES_FILENAME
        original_content = "{invalid yaml: ["
        bad.write_text(original_content, encoding="utf-8")
        load_voices(bad)  # should not raise, should not overwrite
        assert bad.read_text(encoding="utf-8") == original_content

    def test_load_voices_rejects_unknown_type(self, tmp_path, caplog):
        """Voice with unknown type logs error and returns bundled presets (not raises)."""
        bad = tmp_path / VOICES_FILENAME
        bad.write_text(
            "voices:\n  bogus_voice:\n    type: bogus\n    instruct: x\n", encoding="utf-8"
        )
        with caplog.at_level(logging.ERROR, logger="spanish_tts.config"):
            data = load_voices(bad)
        assert any("Schema-invalid" in r.message for r in caplog.records)
        # Falls back to bundled presets.
        assert isinstance(data.get("voices"), dict)
        assert "neutral_male" in data["voices"]

    def test_load_voices_rejects_missing_ref_audio(self, tmp_path, caplog):
        """Clone voice without ref_audio logs error and returns bundled presets."""
        bad = tmp_path / VOICES_FILENAME
        bad.write_text("voices:\n  no_ref_clone:\n    type: clone\n", encoding="utf-8")
        with caplog.at_level(logging.ERROR, logger="spanish_tts.config"):
            data = load_voices(bad)
        assert any("Schema-invalid" in r.message for r in caplog.records)
        assert isinstance(data.get("voices"), dict)
        assert "neutral_male" in data["voices"]

    def test_schema_invalid_load_does_not_overwrite_user_file(self, tmp_path):
        """Schema-invalid user file is NOT modified by load_voices — user must recover manually."""
        bad = tmp_path / VOICES_FILENAME
        original_content = "voices:\n  bad_voice:\n    type: bogus\n    instruct: x\n"
        bad.write_text(original_content, encoding="utf-8")
        load_voices(bad)  # should not raise, should not overwrite
        assert bad.read_text(encoding="utf-8") == original_content


class TestSaveVoicesAtomic:
    def test_atomic_write_succeeds(self, tmp_path):
        """Normal save produces the target file, no tmp left behind."""
        vf = tmp_path / VOICES_FILENAME
        data = {"voices": {"v1": {"type": "design", "instruct": "test"}}}
        save_voices(data, vf)
        assert vf.exists()
        assert not vf.with_suffix(".yaml.tmp").exists()

    def test_tmp_file_absent_after_success(self, tmp_path):
        """The .yaml.tmp sibling is cleaned up (renamed) after successful save."""
        vf = tmp_path / VOICES_FILENAME
        save_voices({"voices": {}}, vf)
        assert not list(tmp_path.glob("*.tmp"))

    def test_atomic_failure_leaves_original_intact(self, tmp_path, monkeypatch):
        """If os.replace raises, the original file is NOT corrupted."""
        vf = tmp_path / VOICES_FILENAME
        original_data = {"voices": {"original": {"type": "design", "instruct": "keep me"}}}
        save_voices(original_data, vf)

        def boom(src, dst):
            raise OSError("simulated disk-full")

        monkeypatch.setattr(os, "replace", boom)
        with pytest.raises(OSError, match="simulated disk-full"):
            save_voices({"voices": {"new_voice": {"type": "design", "instruct": "valid"}}}, vf)

        # Original file unchanged; the "new_voice" key must NOT be present.
        reloaded = load_voices(vf)
        assert "original" in reloaded["voices"]
        assert "new_voice" not in reloaded["voices"]
        # The tmp file may remain on disk after a failed replace — that is
        # acceptable (it won't be confused with the real file due to the .tmp
        # suffix), but the original target must be intact.


class TestSaveVoicesSchemaValidation:
    def test_save_voices_rejects_invalid_schema(self, tmp_path):
        """save_voices raises ValueError for invalid schema BEFORE writing tmp file."""
        vf = tmp_path / VOICES_FILENAME
        invalid_data = {"voices": {"bad_voice": {"type": "bogus", "instruct": "x"}}}
        with pytest.raises(ValueError, match="invalid type"):
            save_voices(invalid_data, vf)
        # No tmp file should have been written.
        assert not vf.with_suffix(".yaml.tmp").exists()
        # Validation fires before any I/O; target file was never created either.
        # (This is vacuously true for a fresh tmp_path but documents intent.)
        assert not vf.exists()


class TestVoicesSchemaValidation:
    def test_valid_schema_passes(self):
        data = {
            "voices": {
                "v1": {"type": "design", "instruct": "x"},
                "v2": {"type": "clone", "ref_audio": "/tmp/a.wav"},
            }
        }
        _validate_voices_schema(data, Path("test.yaml"))  # should not raise

    def test_voices_not_dict_raises(self):
        data = {"voices": ["not", "a", "dict"]}
        with pytest.raises(ValueError, match="mapping"):
            _validate_voices_schema(data, Path("test.yaml"))

    def test_invalid_type_raises(self):
        data = {"voices": {"bad": {"type": "unsupported"}}}
        with pytest.raises(ValueError, match="invalid type"):
            _validate_voices_schema(data, Path("test.yaml"))

    def test_clone_missing_ref_audio_raises(self):
        data = {"voices": {"v1": {"type": "clone"}}}
        with pytest.raises(ValueError, match="ref_audio"):
            _validate_voices_schema(data, Path("test.yaml"))

    def test_clone_empty_ref_audio_raises(self):
        data = {"voices": {"v1": {"type": "clone", "ref_audio": ""}}}
        with pytest.raises(ValueError, match="ref_audio"):
            _validate_voices_schema(data, Path("test.yaml"))

    def test_non_dict_entry_raises(self):
        data = {"voices": {"v1": "not_a_dict"}}
        with pytest.raises(ValueError, match="mapping"):
            _validate_voices_schema(data, Path("test.yaml"))


class TestConfigDirEnvGuard:
    def test_env_var_under_home_accepted(self, tmp_path, monkeypatch):
        """SPANISH_TTS_CONFIG under $HOME is accepted.

        Uses a subdirectory of $HOME to avoid creating a real directory
        outside tmp (the dir is created by get_config_dir via mkdir).
        """
        # Create a dir under $HOME that we can safely clean up.
        safe = Path.home() / ".spanish-tts-pytest-tmp"
        monkeypatch.setenv("SPANISH_TTS_CONFIG", str(safe))
        from spanish_tts.config import get_config_dir

        try:
            d = get_config_dir()
            assert d == safe
        finally:
            # Clean up: the dir was created by get_config_dir's mkdir call.
            if safe.exists():
                safe.rmdir()

    def test_env_var_outside_home_rejected(self, monkeypatch):
        """SPANISH_TTS_CONFIG outside $HOME (and outside tmpdir) raises ValueError."""
        # /etc is never under $HOME or the system tmpdir.
        monkeypatch.setenv("SPANISH_TTS_CONFIG", "/etc/spanish-tts-evil")
        # get_config_dir reads os.environ at call time — no module reload needed.
        from spanish_tts.config import get_config_dir

        with pytest.raises(ValueError, match="HOME"):
            get_config_dir()
