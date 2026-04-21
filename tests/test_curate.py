"""Tests for scripts/curate.py helper functions."""

import sys
from pathlib import Path

import pytest

# Add scripts dir to path so we can import curate
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from curate import _find_speaker_clips, _matches_filter, _score_speakers


class TestMatchesFilter:
    def test_no_filters(self):
        row = {"country": "Mexico", "gender": "male"}
        assert _matches_filter(row) is True

    def test_country_match(self):
        row = {"country": "Mexico", "gender": "male"}
        assert _matches_filter(row, country="mexico") is True
        assert _matches_filter(row, country="Mexico") is True

    def test_country_mismatch(self):
        row = {"country": "Mexico", "gender": "male"}
        assert _matches_filter(row, country="spain") is False

    def test_gender_match(self):
        row = {"country": "Mexico", "gender": "male"}
        assert _matches_filter(row, gender="male") is True

    def test_gender_mismatch(self):
        row = {"country": "Mexico", "gender": "male"}
        assert _matches_filter(row, gender="female") is False

    def test_both_filters_match(self):
        row = {"country": "Spain", "gender": "female"}
        assert _matches_filter(row, country="spain", gender="female") is True

    def test_both_filters_partial_mismatch(self):
        row = {"country": "Spain", "gender": "female"}
        assert _matches_filter(row, country="spain", gender="male") is False

    def test_none_country(self):
        row = {"country": None, "gender": "male"}
        assert _matches_filter(row) is True
        assert _matches_filter(row, country="mexico") is False

    def test_none_gender(self):
        row = {"country": "Mexico", "gender": None}
        assert _matches_filter(row) is True
        assert _matches_filter(row, gender="male") is False


class TestScoreSpeakers:
    @pytest.fixture
    def speakers(self):
        return {
            "SPK001": [
                {"duration": 8.0, "text": "Primera oracion corta", "country": "Mexico", "gender": "male", "audio_id": "a1"},
                {"duration": 7.5, "text": "Segunda oracion un poco mas larga que la primera", "country": "Mexico", "gender": "male", "audio_id": "a2"},
            ],
            "SPK002": [
                {"duration": 9.0, "text": "Una sola oracion", "country": "Spain", "gender": "female", "audio_id": "b1"},
            ],
        }

    def test_returns_all_speakers(self, speakers):
        scored = _score_speakers(speakers)
        assert len(scored) == 2

    def test_sorted_by_clip_count(self, speakers):
        scored = _score_speakers(speakers)
        assert scored[0]["speaker_id"] == "SPK001"  # 2 clips
        assert scored[1]["speaker_id"] == "SPK002"  # 1 clip

    def test_score_fields(self, speakers):
        scored = _score_speakers(speakers)
        s = scored[0]
        assert s["n_clips"] == 2
        assert s["avg_duration"] == pytest.approx(7.75)
        assert "best_clip" in s
        assert s["country"] == "Mexico"
        assert s["gender"] == "male"

    def test_best_clip_selection(self, speakers):
        scored = _score_speakers(speakers)
        best = scored[0]["best_clip"]
        # Best is max(text_len * duration)
        assert best["audio_id"] == "a2"  # longer text * decent duration

    def test_empty_speakers(self):
        scored = _score_speakers({})
        assert scored == []


class TestFindSpeakerClips:
    @pytest.fixture
    def mock_dataset(self):
        """Minimal list-of-dicts mimicking dataset rows."""
        return [
            {"speaker_id": "S1", "duration": 5.0, "normalized_text": "short"},
            {"speaker_id": "S1", "duration": 8.0, "normalized_text": "good length clip"},
            {"speaker_id": "S2", "duration": 7.0, "normalized_text": "other speaker"},
            {"speaker_id": "S1", "duration": 15.0, "normalized_text": "too long"},
            {"speaker_id": "S1", "duration": 9.0, "normalized_text": "another good one"},
        ]

    def test_finds_clips_in_range(self, mock_dataset):
        clips = _find_speaker_clips(mock_dataset, "S1", 6.0, 12.0)
        assert len(clips) == 2
        indices = [i for i, _ in clips]
        assert 1 in indices  # 8.0s
        assert 4 in indices  # 9.0s

    def test_excludes_other_speakers(self, mock_dataset):
        clips = _find_speaker_clips(mock_dataset, "S1", 6.0, 12.0)
        for _, row in clips:
            assert row["speaker_id"] == "S1"

    def test_no_matches(self, mock_dataset):
        clips = _find_speaker_clips(mock_dataset, "S1", 20.0, 30.0)
        assert clips == []

    def test_nonexistent_speaker(self, mock_dataset):
        clips = _find_speaker_clips(mock_dataset, "NOPE", 0.0, 100.0)
        assert clips == []
