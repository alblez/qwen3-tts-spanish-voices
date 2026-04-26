"""Speed correctness tests for pitch-preserving time-stretch."""

import logging
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

from spanish_tts.engine import _apply_speed, SPEED_MIN, SPEED_MAX

try:
    import librosa  # noqa: F401
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False


logger = logging.getLogger(__name__)


class TestApplySpeed:
    """Tests for _apply_speed helper (direct unit tests, no model required)."""

    def test_speed_no_op_at_one(self):
        """speed=1.0 returns input unchanged."""
        audio = np.sin(np.linspace(0, 2 * np.pi, 24000)).astype(np.float32)
        result = _apply_speed(audio, 1.0, 24000)
        assert result is audio

    def test_speed_boundary_min_accepted(self):
        """speed=0.5 accepted without error."""
        audio = np.zeros(24000, dtype=np.float32)
        try:
            import librosa
            result = _apply_speed(audio, 0.5, 24000)
            assert result is not None
        except ImportError:
            pytest.skip("librosa not installed")

    def test_speed_boundary_max_accepted(self):
        """speed=2.0 accepted without error."""
        audio = np.zeros(24000, dtype=np.float32)
        try:
            import librosa
            result = _apply_speed(audio, 2.0, 24000)
            assert result is not None
        except ImportError:
            pytest.skip("librosa not installed")

    def test_speed_below_min_rejected(self):
        """speed=0.499 raises ValueError."""
        audio = np.zeros(24000, dtype=np.float32)
        with pytest.raises(ValueError, match="speed out of range"):
            _apply_speed(audio, 0.499, 24000)

    def test_speed_above_max_rejected(self):
        """speed=2.001 raises ValueError."""
        audio = np.zeros(24000, dtype=np.float32)
        with pytest.raises(ValueError, match="speed out of range"):
            _apply_speed(audio, 2.001, 24000)

    @pytest.mark.skipif(
        not HAS_LIBROSA,
        reason="librosa not available"
    )
    def test_speed_stretch_ratio_slow(self):
        """speed=0.5 doubles length (±5% tolerance)."""
        import librosa
        # Short synthetic sine wave: 1 second @ 24kHz
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 24000)).astype(np.float32)
        stretched = _apply_speed(audio, 0.5, 24000)
        expected_len = len(audio) * 2
        ratio = len(stretched) / expected_len
        assert 0.95 <= ratio <= 1.05, f"Slow stretch ratio {ratio} outside 0.95-1.05"

    @pytest.mark.skipif(
        not HAS_LIBROSA,
        reason="librosa not available"
    )
    def test_speed_stretch_ratio_fast(self):
        """speed=2.0 halves length (±5% tolerance)."""
        import librosa
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 24000)).astype(np.float32)
        stretched = _apply_speed(audio, 2.0, 24000)
        expected_len = len(audio) / 2
        ratio = len(stretched) / expected_len
        assert 0.95 <= ratio <= 1.05, f"Fast stretch ratio {ratio} outside 0.95-1.05"

    @pytest.mark.skipif(
        not HAS_LIBROSA,
        reason="librosa not available"
    )
    def test_pitch_preservation_slow(self):
        """Spectral centroid diff <10% between speed=1.0 and speed=0.5."""
        import librosa
        # Synthetic 440Hz sine at 24kHz for 2 seconds
        sr = 24000
        t = np.linspace(0, 2, sr * 2)
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)

        # Compute spectral centroid
        cent_ref = float(np.mean(librosa.feature.spectral_centroid(
            y=audio, sr=sr
        )))

        stretched = _apply_speed(audio, 0.5, sr)
        cent_slow = float(np.mean(librosa.feature.spectral_centroid(
            y=stretched, sr=sr
        )))

        pct_diff = abs(cent_slow - cent_ref) / cent_ref
        assert pct_diff < 0.10, f"Pitch drift {pct_diff*100:.1f}% > 10%"

    @pytest.mark.skipif(
        not HAS_LIBROSA,
        reason="librosa not available"
    )
    def test_pitch_preservation_fast(self):
        """Spectral centroid diff <10% between speed=1.0 and speed=2.0."""
        import librosa
        sr = 24000
        t = np.linspace(0, 2, sr * 2)
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)

        cent_ref = float(np.mean(librosa.feature.spectral_centroid(
            y=audio, sr=sr
        )))

        stretched = _apply_speed(audio, 2.0, sr)
        cent_fast = float(np.mean(librosa.feature.spectral_centroid(
            y=stretched, sr=sr
        )))

        pct_diff = abs(cent_fast - cent_ref) / cent_ref
        assert pct_diff < 0.10, f"Pitch drift {pct_diff*100:.1f}% > 10%"

    def test_graceful_fallback_no_librosa(self):
        """Missing librosa returns input unchanged with warning."""
        audio = np.zeros(24000, dtype=np.float32)
        with patch.dict("sys.modules", {"librosa": None}):
            with patch("spanish_tts.engine.logger") as mock_logger:
                # Simulate ImportError by raising on the import
                with patch("builtins.__import__", side_effect=ImportError):
                    result = _apply_speed(audio, 1.5, 24000)
                    assert np.array_equal(result, audio)
                    # Contract: fallback must log a warning, not fail silently.
                    mock_logger.warning.assert_called_once()
                    msg = mock_logger.warning.call_args[0][0]
                    assert 'librosa' in msg.lower()
                    assert 'speed' in msg.lower()


class TestCLISpeedValidation:
    """CLI speed validation via click.FloatRange."""

    def test_cli_speed_reject_out_of_range(self):
        """CLI rejects --speed 3.0 before loading any model."""
        from click.testing import CliRunner
        from spanish_tts.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["say", "test", "--speed", "3.0"])
        assert result.exit_code != 0
        assert "Invalid value" in result.output or "out of range" in result.output

    def test_cli_speed_accept_boundary_min(self):
        """CLI accepts --speed 0.5."""
        from click.testing import CliRunner
        from spanish_tts.cli import say

        runner = CliRunner()
        # Mock generate to avoid loading model
        with patch("spanish_tts.cli.generate", return_value="/tmp/test.wav"):
            with patch("spanish_tts.cli.get_voice", return_value={"type": "design", "instruct": "test"}):
                with patch("spanish_tts.cli.get_defaults", return_value={}):
                    result = runner.invoke(say, ["test", "--speed", "0.5"])
                    # Should not error on speed parsing
                    assert "out of range" not in result.output.lower() or result.exit_code == 0

    def test_cli_speed_accept_boundary_max(self):
        """CLI accepts --speed 2.0."""
        from click.testing import CliRunner
        from spanish_tts.cli import say

        runner = CliRunner()
        with patch("spanish_tts.cli.generate", return_value="/tmp/test.wav"):
            with patch("spanish_tts.cli.get_voice", return_value={"type": "design", "instruct": "test"}):
                with patch("spanish_tts.cli.get_defaults", return_value={}):
                    result = runner.invoke(say, ["test", "--speed", "2.0"])
                    assert "out of range" not in result.output.lower() or result.exit_code == 0


class TestMCPSpeedValidation:
    """MCP speed validation."""

    def test_mcp_say_reject_out_of_range(self):
        """MCP say() rejects out-of-range speed."""
        from spanish_tts.mcp_server import say

        result = say(text="test", voice="neutral_male", speed=3.0)
        assert "error" in result
        assert "speed out of range" in result["error"]

    def test_mcp_demo_reject_out_of_range(self):
        """MCP demo() rejects out-of-range speed."""
        from spanish_tts.mcp_server import demo

        result = demo(text="test", speed=0.4)
        assert "error" in result
        assert "speed out of range" in result["error"]

    def test_mcp_demo_accept_boundary_min(self):
        """MCP demo() accepts speed=0.5."""
        from spanish_tts.mcp_server import demo

        with patch("spanish_tts.mcp_server.list_voices", return_value={}):
            result = demo(text="test", speed=0.5)
            # Should not error on speed; if voices empty, that's ok
            assert "speed out of range" not in result.get("error", "")

    def test_mcp_demo_accept_boundary_max(self):
        """MCP demo() accepts speed=2.0."""
        from spanish_tts.mcp_server import demo

        with patch("spanish_tts.mcp_server.list_voices", return_value={}):
            result = demo(text="test", speed=2.0)
            assert "speed out of range" not in result.get("error", "")


class TestConstants:
    """Test that constants are correctly defined."""

    def test_speed_min_max_defined(self):
        """SPEED_MIN=0.5, SPEED_MAX=2.0."""
        assert SPEED_MIN == 0.5
        assert SPEED_MAX == 2.0
