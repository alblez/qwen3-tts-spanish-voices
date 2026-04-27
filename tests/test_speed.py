"""Speed correctness tests for pitch-preserving time-stretch."""

import logging
from unittest.mock import patch

import numpy as np
import pytest

from spanish_tts.engine import SPEED_MAX, SPEED_MIN, TtsResult, _apply_speed

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
            import librosa  # noqa: F401  # probe for availability

            result = _apply_speed(audio, 0.5, 24000)
            assert result is not None
        except ImportError:
            pytest.skip("librosa not installed")

    def test_speed_boundary_max_accepted(self):
        """speed=2.0 accepted without error."""
        audio = np.zeros(24000, dtype=np.float32)
        try:
            import librosa  # noqa: F401  # probe for availability

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

    @pytest.mark.skipif(not HAS_LIBROSA, reason="librosa not available")
    def test_speed_stretch_ratio_slow(self):
        """speed=0.5 doubles length (±5% tolerance)."""

        # Short synthetic sine wave: 1 second @ 24kHz
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 24000)).astype(np.float32)
        stretched = _apply_speed(audio, 0.5, 24000)
        expected_len = len(audio) * 2
        ratio = len(stretched) / expected_len
        assert 0.95 <= ratio <= 1.05, f"Slow stretch ratio {ratio} outside 0.95-1.05"

    @pytest.mark.skipif(not HAS_LIBROSA, reason="librosa not available")
    def test_speed_stretch_ratio_fast(self):
        """speed=2.0 halves length (±5% tolerance)."""

        audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 24000)).astype(np.float32)
        stretched = _apply_speed(audio, 2.0, 24000)
        expected_len = len(audio) / 2
        ratio = len(stretched) / expected_len
        assert 0.95 <= ratio <= 1.05, f"Fast stretch ratio {ratio} outside 0.95-1.05"

    @pytest.mark.skipif(not HAS_LIBROSA, reason="librosa not available")
    def test_pitch_preservation_slow(self):
        """Spectral centroid diff <10% between speed=1.0 and speed=0.5."""
        import librosa

        # Synthetic 440Hz sine at 24kHz for 2 seconds
        sr = 24000
        t = np.linspace(0, 2, sr * 2)
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)

        # Compute spectral centroid
        cent_ref = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)))

        stretched = _apply_speed(audio, 0.5, sr)
        cent_slow = float(np.mean(librosa.feature.spectral_centroid(y=stretched, sr=sr)))

        pct_diff = abs(cent_slow - cent_ref) / cent_ref
        assert pct_diff < 0.10, f"Pitch drift {pct_diff * 100:.1f}% > 10%"

    @pytest.mark.skipif(not HAS_LIBROSA, reason="librosa not available")
    def test_pitch_preservation_fast(self):
        """Spectral centroid diff <10% between speed=1.0 and speed=2.0."""
        import librosa

        sr = 24000
        t = np.linspace(0, 2, sr * 2)
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)

        cent_ref = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)))

        stretched = _apply_speed(audio, 2.0, sr)
        cent_fast = float(np.mean(librosa.feature.spectral_centroid(y=stretched, sr=sr)))

        pct_diff = abs(cent_fast - cent_ref) / cent_ref
        assert pct_diff < 0.10, f"Pitch drift {pct_diff * 100:.1f}% > 10%"

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
                    assert "librosa" in msg.lower()
                    assert "speed" in msg.lower()


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
            with patch(
                "spanish_tts.cli.get_voice", return_value={"type": "design", "instruct": "test"}
            ):
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
            with patch(
                "spanish_tts.cli.get_voice", return_value={"type": "design", "instruct": "test"}
            ):
                with patch("spanish_tts.cli.get_defaults", return_value={}):
                    result = runner.invoke(say, ["test", "--speed", "2.0"])
                    assert "out of range" not in result.output.lower() or result.exit_code == 0


class TestMCPSpeedValidation:
    """MCP speed validation."""

    def test_mcp_say_none_speed_uses_defaults_speed(self):
        """MCP say(speed=None) honours voices.yaml defaults.speed (CLI parity)."""
        from spanish_tts.mcp_server import say

        with patch(
            "spanish_tts.mcp_server.get_voice", return_value={"type": "design", "instruct": "t"}
        ):
            with patch(
                "spanish_tts.mcp_server.get_defaults",
                return_value={"speed": 1.25, "output_dir": "/tmp/x"},
            ):
                captured = {}

                def fake_generate(**kwargs):
                    captured.update(kwargs)
                    return TtsResult(path="/tmp/ok.wav", duration_seconds=1.0)

                with patch("spanish_tts.mcp_server.generate", side_effect=fake_generate):
                    result = say(text="hola", voice="x", speed=None)
                    assert "error" not in result
                    assert captured["speed"] == 1.25

    def test_mcp_say_default_speed_is_none(self):
        """MCP say() without speed arg uses defaults (parity with CLI)."""
        from spanish_tts.mcp_server import say

        with patch(
            "spanish_tts.mcp_server.get_voice", return_value={"type": "design", "instruct": "t"}
        ):
            with patch(
                "spanish_tts.mcp_server.get_defaults",
                return_value={"speed": 0.9, "output_dir": "/tmp/x"},
            ):
                captured = {}

                def fake_generate(**kwargs):
                    captured.update(kwargs)
                    return TtsResult(path="/tmp/ok.wav", duration_seconds=1.0)

                with patch("spanish_tts.mcp_server.generate", side_effect=fake_generate):
                    say(text="hola", voice="x")
                    assert captured["speed"] == 0.9

    def test_mcp_say_defaults_speed_out_of_range_rejected(self):
        """MCP say() rejects out-of-range defaults.speed after fallback."""
        from spanish_tts.mcp_server import say

        with patch(
            "spanish_tts.mcp_server.get_voice", return_value={"type": "design", "instruct": "t"}
        ):
            with patch(
                "spanish_tts.mcp_server.get_defaults",
                return_value={"speed": 3.0, "output_dir": "/tmp/x"},
            ):
                result = say(text="hola", voice="x")
                assert "error" in result
                assert "speed out of range" in result["error"]

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


class TestApplySpeedInputGuards:
    """ENG-1: _apply_speed input-contract guards."""

    def test_rejects_non_ndarray(self):
        with pytest.raises(ValueError, match="numpy.ndarray"):
            _apply_speed([0.0] * 4096, 1.5, 24000)  # type: ignore[arg-type]

    def test_rejects_stereo_2d(self):
        stereo = np.zeros((2, 4096), dtype=np.float32)
        with pytest.raises(ValueError, match="mono 1-D"):
            _apply_speed(stereo, 1.5, 24000)

    def test_rejects_empty_array(self):
        empty = np.zeros(0, dtype=np.float32)
        with pytest.raises(ValueError, match="empty"):
            _apply_speed(empty, 1.5, 24000)

    def test_rejects_too_short(self):
        short = np.zeros(512, dtype=np.float32)
        with pytest.raises(ValueError, match="at least"):
            _apply_speed(short, 1.5, 24000)

    def test_rejects_nan(self):
        # Finite-check runs before librosa import, so this test doesn't
        # need the librosa extra installed to be meaningful.
        bad = np.full(4096, np.nan, dtype=np.float32)
        with pytest.raises(ValueError, match="non-finite"):
            _apply_speed(bad, 1.5, 24000)

    def test_rejects_inf(self):
        bad = np.zeros(4096, dtype=np.float32)
        bad[100] = np.inf
        with pytest.raises(ValueError, match="non-finite"):
            _apply_speed(bad, 1.5, 24000)

    def test_noop_path_is_early_return_before_guards(self):
        # !!! DO NOT "FIX" BY MOVING GUARDS ABOVE THE speed==1.0 CHECK !!!
        #
        # INTENTIONAL backward-compat: speed==1.0 is a fast-path no-op that
        # returns the input unchanged WITHOUT running any shape/dtype/finite
        # validation. This is not an endorsement of stereo or NaN input —
        # it is a contract pin so that callers who already pass pre-validated
        # data (and never actually stretch) keep working after ENG-1.
        #
        # If you want to tighten the no-op path too, that is a SEPARATE
        # change with a migration plan + test updates across the repo.
        weird = np.zeros((2, 10), dtype=np.float32)
        result = _apply_speed(weird, 1.0, 24000)
        assert result is weird

    def test_boundary_2048_samples_accepted(self):
        # Exactly 2048 samples (librosa STFT n_fft default) must not raise
        # the length guard. We deliberately skip running librosa here — the
        # guard itself is the contract. A slice of a real signal keeps
        # non-finite guard happy too.
        audio = np.sin(np.linspace(0, 2 * np.pi, 2048)).astype(np.float32)
        # Must not raise ValueError about length:
        if HAS_LIBROSA:
            # End-to-end happy path if librosa is available.
            out = _apply_speed(audio, 1.5, 24000)
            assert out.ndim == 1
        else:
            # Without librosa, the function logs a warning and returns
            # input unchanged — still must pass the length guard.
            out = _apply_speed(audio, 1.5, 24000)
            assert out is audio

    def test_boundary_2047_samples_rejected(self):
        audio = np.zeros(2047, dtype=np.float32)
        with pytest.raises(ValueError, match="at least"):
            _apply_speed(audio, 1.5, 24000)


# ---------------------------------------------------------------------------
# U3-6: Speed boundary parametrized tests — NaN, inf, and edge values
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_speed",
    [
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="pos_inf"),
        pytest.param(float("-inf"), id="neg_inf"),
        pytest.param(0.4999, id="just_below_min"),
        pytest.param(2.0001, id="just_above_max"),
        pytest.param(-1.0, id="negative"),
        pytest.param(0.0, id="zero"),
    ],
)
def test_apply_speed_rejects_invalid_speeds(bad_speed):
    """_apply_speed must raise ValueError for all invalid speed values.

    NaN/inf: the range check `not (0.5 <= x <= 2.0)` correctly evaluates
    to True for NaN (all comparisons return False) and for ±inf (inf > 2.0).
    All cases therefore raise ValueError before touching librosa.
    """
    audio = np.zeros(24000, dtype=np.float32)
    with pytest.raises(ValueError):
        _apply_speed(audio, bad_speed, 24000)
