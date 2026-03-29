"""Tests for pipeline.collect_weather using mocked API responses."""
import json
import math
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from pipeline.collect_weather import (
    _round_to_nearest_hour,
    _extract_hour_weather,
    collect_weather,
)


def _make_mock_weather_json(date: str, hour: int, temp: float, wind: float, precip: float):
    """Create a minimal Open-Meteo-style JSON response for a single day."""
    times = [f"{date}T{h:02d}:00" for h in range(24)]
    temps = [15.0] * 24
    winds = [10.0] * 24
    precips = [0.0] * 24
    temps[hour] = temp
    winds[hour] = wind
    precips[hour] = precip
    return {
        "hourly": {
            "time": times,
            "temperature_2m": temps,
            "wind_speed_10m": winds,
            "precipitation": precips,
        }
    }


class TestHourRounding:
    def test_round_down(self):
        # 13:22 → 13:00
        ts = pd.Timestamp("2023-09-08 13:22", tz="UTC")
        rounded = _round_to_nearest_hour(ts)
        assert rounded.hour == 13

    def test_round_up(self):
        # 13:42 → 14:00
        ts = pd.Timestamp("2023-09-08 13:42", tz="UTC")
        rounded = _round_to_nearest_hour(ts)
        assert rounded.hour == 14

    def test_round_half_up(self):
        # 13:30 → 14:00
        ts = pd.Timestamp("2023-09-08 13:30", tz="UTC")
        rounded = _round_to_nearest_hour(ts)
        assert rounded.hour == 14

    def test_exact_hour(self):
        ts = pd.Timestamp("2023-09-08 17:00", tz="UTC")
        rounded = _round_to_nearest_hour(ts)
        assert rounded.hour == 17

    def test_nat_returns_nat(self):
        result = _round_to_nearest_hour(pd.NaT)
        assert pd.isna(result)


class TestExtractHourWeather:
    def test_valid_extraction(self):
        wj = _make_mock_weather_json("2023-09-08", 13, 28.5, 12.0, 0.0)
        target = pd.Timestamp("2023-09-08 13:00", tz="UTC")
        result = _extract_hour_weather(wj, target)
        assert math.isclose(result["temperature"], 28.5)
        assert math.isclose(result["wind_speed"], 12.0)
        assert math.isclose(result["precipitation"], 0.0)

    def test_missing_hour_returns_nan(self):
        wj = _make_mock_weather_json("2023-09-08", 13, 28.5, 12.0, 0.0)
        # Ask for a different date
        target = pd.Timestamp("2023-09-09 13:00", tz="UTC")
        result = _extract_hour_weather(wj, target)
        assert math.isnan(result["temperature"])

    def test_none_json_returns_nan(self):
        target = pd.Timestamp("2023-09-08 13:00", tz="UTC")
        result = _extract_hour_weather(None, target)
        assert math.isnan(result["temperature"])


class TestDomeGamesGetNanWeather:
    def test_dome_games_nan(self, sample_games_df, sample_weather_df):
        dome_games = sample_games_df[sample_games_df["dome_indicator"] == 1]
        for _, row in dome_games.iterrows():
            wx_row = sample_weather_df[sample_weather_df["game_id"] == row["game_id"]]
            if not wx_row.empty:
                assert pd.isna(wx_row.iloc[0]["temperature"])
                assert pd.isna(wx_row.iloc[0]["wind_speed"])
                assert pd.isna(wx_row.iloc[0]["precipitation"])


class TestOutdoorGameGetsWeather:
    def test_outdoor_has_values(self, sample_games_df, sample_weather_df):
        outdoor = sample_games_df[
            (sample_games_df["dome_indicator"] == 0) &
            (sample_games_df["home_score"].notna())
        ]
        for _, row in outdoor.iterrows():
            wx_row = sample_weather_df[sample_weather_df["game_id"] == row["game_id"]]
            if not wx_row.empty:
                # Most outdoor games should have non-NaN temperature
                # (the 2024 unplayed game is an exception)
                if row["game_id"] != "2024_01_KC_BAL":
                    assert pd.notna(wx_row.iloc[0]["temperature"]), (
                        f"Game {row['game_id']} missing temperature"
                    )


class TestTemperatureRangeSanity:
    def test_all_in_range(self, sample_weather_df):
        valid = sample_weather_df["temperature"].dropna()
        assert (valid >= -40).all()
        assert (valid <= 45).all()


class TestCachingPreventsRefetch:
    @patch("pipeline.collect_weather._fetch_weather_batch")
    @patch("pipeline.collect_weather._cache_path")
    def test_cache_hit(self, mock_cache_path, mock_fetch, tmp_path):
        """If cache file exists, API should NOT be called."""
        # Create a cache file
        cache_file = tmp_path / "weather_KC_2023.json"
        wj = _make_mock_weather_json("2023-09-07", 0, 28.5, 12.0, 0.0)
        cache_file.write_text(json.dumps(wj))
        mock_cache_path.return_value = cache_file

        # Create a minimal outdoor game
        games = pd.DataFrame([{
            "game_id": "2023_01_KC_DET",
            "home_team": "KC", "season": 2023,
            "dome_indicator": 0,
            "kickoff_utc": pd.Timestamp("2023-09-08 00:20", tz="UTC"),
            "gameday": "2023-09-07",
        }])

        collect_weather(games)
        mock_fetch.assert_not_called()


class TestHandlesApiFailure:
    @patch("pipeline.collect_weather._fetch_weather_batch", return_value=None)
    @patch("pipeline.collect_weather._cache_path")
    def test_api_failure_returns_nan(self, mock_cache_path, mock_fetch, tmp_path):
        """API failure should return NaN weather, not crash."""
        cache_file = tmp_path / "weather_KC_2023.json"
        mock_cache_path.return_value = cache_file  # file doesn't exist → triggers fetch

        games = pd.DataFrame([{
            "game_id": "2023_01_KC_DET",
            "home_team": "KC", "season": 2023,
            "dome_indicator": 0,
            "kickoff_utc": pd.Timestamp("2023-09-08 00:20", tz="UTC"),
            "gameday": "2023-09-07",
        }])

        result = collect_weather(games)
        assert len(result) == 1
        assert pd.isna(result.iloc[0]["temperature"])
