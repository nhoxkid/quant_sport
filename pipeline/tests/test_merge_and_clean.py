"""Tests for pipeline.merge_and_clean using synthetic fixtures."""
import logging

import numpy as np
import pandas as pd
import pytest

from pipeline.merge_and_clean import merge_and_clean
from pipeline.config import E_T_THRESHOLD


class TestMergeNoRowDuplication:
    def test_row_count_preserved(self, sample_games_df, sample_weather_df, sample_odds_df):
        # Drop the game with missing score first (merge_and_clean will drop it)
        valid_games = sample_games_df.dropna(subset=["home_score"]).copy()
        valid_games["home_score"] = valid_games["home_score"].astype(int)
        valid_games["away_score"] = valid_games["away_score"].astype(int)
        valid_games["total_points"] = valid_games["home_score"] + valid_games["away_score"]

        result = merge_and_clean(valid_games, sample_weather_df, sample_odds_df)
        n_unique = valid_games["game_id"].nunique()
        assert len(result) == n_unique


class TestMergeNoRowLoss:
    def test_all_game_ids_present(self, sample_games_df, sample_weather_df, sample_odds_df):
        valid_games = sample_games_df.dropna(subset=["home_score"]).copy()
        valid_games["home_score"] = valid_games["home_score"].astype(int)
        valid_games["away_score"] = valid_games["away_score"].astype(int)
        valid_games["total_points"] = valid_games["home_score"] + valid_games["away_score"]

        result = merge_and_clean(valid_games, sample_weather_df, sample_odds_df)
        for gid in valid_games["game_id"]:
            assert gid in result["game_id"].values, f"Lost game_id: {gid}"


class TestTPrimeComputed:
    def test_t_prime_value(self, sample_games_df, sample_weather_df, sample_odds_df):
        valid_games = sample_games_df.dropna(subset=["home_score"]).copy()
        valid_games["home_score"] = valid_games["home_score"].astype(int)
        valid_games["away_score"] = valid_games["away_score"].astype(int)
        valid_games["total_points"] = valid_games["home_score"] + valid_games["away_score"]

        result = merge_and_clean(valid_games, sample_weather_df, sample_odds_df)

        # For outdoor games with weather, T_prime should be T - T_norm
        outdoor_wx = result[(result["dome_indicator"] == 0) & result["temperature"].notna()]
        for _, row in outdoor_wx.iterrows():
            if pd.notna(row["T_prime"]) and pd.notna(row["T_norm"]):
                expected = row["temperature"] - row["T_norm"]
                assert abs(row["T_prime"] - expected) < 1e-6, (
                    f"Game {row['game_id']}: T_prime={row['T_prime']} != T-T_norm={expected}"
                )


class TestEWExtremeFlag:
    def test_extreme_wind(self):
        """Set up data where one game has extreme wind → E_W=1."""
        games = pd.DataFrame([
            {"game_id": "g1", "season": 2023, "week": 1, "game_type": "REG",
             "home_team": "BUF", "away_team": "NYJ",
             "home_score": 20, "away_score": 10, "total_points": 30,
             "gameday": "2023-09-10", "gametime": "13:00",
             "kickoff_utc": pd.Timestamp("2023-09-10 17:00", tz="UTC"),
             "stadium": "S", "roof": "outdoors", "dome_indicator": 0},
            {"game_id": "g2", "season": 2023, "week": 2, "game_type": "REG",
             "home_team": "BUF", "away_team": "MIA",
             "home_score": 17, "away_score": 14, "total_points": 31,
             "gameday": "2023-09-17", "gametime": "13:00",
             "kickoff_utc": pd.Timestamp("2023-09-17 17:00", tz="UTC"),
             "stadium": "S", "roof": "outdoors", "dome_indicator": 0},
        ])
        # Wind: game 2 has 95+ km/h (well above 90th pct of these 2 games)
        weather = pd.DataFrame([
            {"game_id": "g1", "temperature": 20.0, "wind_speed": 10.0, "precipitation": 0.0},
            {"game_id": "g2", "temperature": 18.0, "wind_speed": 95.0, "precipitation": 0.0},
        ])
        odds = pd.DataFrame([
            {"game_id": "g1", "L_close": 44.0, "L_open": np.nan,
             "over_odds": np.nan, "under_odds": np.nan,
             "over_implied_prob": np.nan, "under_implied_prob": np.nan,
             "overround": np.nan, "p_over_vigfree": np.nan, "p_under_vigfree": np.nan},
            {"game_id": "g2", "L_close": 42.0, "L_open": np.nan,
             "over_odds": np.nan, "under_odds": np.nan,
             "over_implied_prob": np.nan, "under_implied_prob": np.nan,
             "overround": np.nan, "p_over_vigfree": np.nan, "p_under_vigfree": np.nan},
        ])
        result = merge_and_clean(games, weather, odds)
        g2 = result[result["game_id"] == "g2"].iloc[0]
        assert g2["E_W"] == 1, f"E_W should be 1 for extreme wind, got {g2['E_W']}"


class TestETExtremeFlag:
    def test_extreme_temperature(self):
        """|T_prime| = 10°C with threshold 8°C → E_T = 1."""
        games = pd.DataFrame([
            {"game_id": "g1", "season": 2023, "week": 1, "game_type": "REG",
             "home_team": "GB", "away_team": "CHI",
             "home_score": 20, "away_score": 10, "total_points": 30,
             "gameday": "2023-09-10", "gametime": "13:00",
             "kickoff_utc": pd.Timestamp("2023-09-10 17:00", tz="UTC"),
             "stadium": "Lambeau", "roof": "outdoors", "dome_indicator": 0},
            {"game_id": "g2", "season": 2023, "week": 15, "game_type": "REG",
             "home_team": "GB", "away_team": "TB",
             "home_score": 17, "away_score": 14, "total_points": 31,
             "gameday": "2023-12-17", "gametime": "13:00",
             "kickoff_utc": pd.Timestamp("2023-12-17 18:00", tz="UTC"),
             "stadium": "Lambeau", "roof": "outdoors", "dome_indicator": 0},
        ])
        # Game 1: Sep temp 25°C, Game 2: Dec temp -5°C
        # T_norm for GB month 9 = 25, month 12 = -5 (only one game each)
        # So T_prime = 0 for both (each IS the norm for its month)
        # To get E_T=1, we need games in the same month with different temps
        # Let's use 3 games in same month
        games = pd.DataFrame([
            {"game_id": "g1", "season": 2023, "week": 1, "game_type": "REG",
             "home_team": "GB", "away_team": "CHI",
             "home_score": 20, "away_score": 10, "total_points": 30,
             "gameday": "2023-09-10", "gametime": "13:00",
             "kickoff_utc": pd.Timestamp("2023-09-10 17:00", tz="UTC"),
             "stadium": "Lambeau", "roof": "outdoors", "dome_indicator": 0},
            {"game_id": "g2", "season": 2023, "week": 2, "game_type": "REG",
             "home_team": "GB", "away_team": "TB",
             "home_score": 17, "away_score": 14, "total_points": 31,
             "gameday": "2023-09-17", "gametime": "13:00",
             "kickoff_utc": pd.Timestamp("2023-09-17 18:00", tz="UTC"),
             "stadium": "Lambeau", "roof": "outdoors", "dome_indicator": 0},
            {"game_id": "g3", "season": 2023, "week": 3, "game_type": "REG",
             "home_team": "GB", "away_team": "DET",
             "home_score": 24, "away_score": 21, "total_points": 45,
             "gameday": "2023-09-24", "gametime": "13:00",
             "kickoff_utc": pd.Timestamp("2023-09-24 17:00", tz="UTC"),
             "stadium": "Lambeau", "roof": "outdoors", "dome_indicator": 0},
        ])
        # Temps: 20, 20, 30 → T_norm = ~23.33; g3 has T_prime ≈ 6.67 (not extreme)
        # Need bigger spread: 20, 20, 40 → T_norm = 26.67; g3 has T_prime ≈ 13.33 (extreme)
        weather = pd.DataFrame([
            {"game_id": "g1", "temperature": 20.0, "wind_speed": 10.0, "precipitation": 0.0},
            {"game_id": "g2", "temperature": 20.0, "wind_speed": 10.0, "precipitation": 0.0},
            {"game_id": "g3", "temperature": 40.0, "wind_speed": 10.0, "precipitation": 0.0},
        ])
        odds = pd.DataFrame([
            {"game_id": f"g{i}", "L_close": 44.0, "L_open": np.nan,
             "over_odds": np.nan, "under_odds": np.nan,
             "over_implied_prob": np.nan, "under_implied_prob": np.nan,
             "overround": np.nan, "p_over_vigfree": np.nan, "p_under_vigfree": np.nan}
            for i in range(1, 4)
        ])
        result = merge_and_clean(games, weather, odds)
        g3 = result[result["game_id"] == "g3"].iloc[0]
        # T_norm ≈ 26.67, T_prime ≈ 13.33, |T_prime| > 8 → E_T = 1
        assert g3["E_T"] == 1, f"E_T should be 1, got {g3['E_T']} (T_prime={g3['T_prime']:.2f})"

    def test_not_extreme(self):
        """|T_prime| = 5°C with threshold 8°C → E_T = 0."""
        games = pd.DataFrame([
            {"game_id": "g1", "season": 2023, "week": 1, "game_type": "REG",
             "home_team": "GB", "away_team": "CHI",
             "home_score": 20, "away_score": 10, "total_points": 30,
             "gameday": "2023-09-10", "gametime": "13:00",
             "kickoff_utc": pd.Timestamp("2023-09-10 17:00", tz="UTC"),
             "stadium": "Lambeau", "roof": "outdoors", "dome_indicator": 0},
            {"game_id": "g2", "season": 2023, "week": 2, "game_type": "REG",
             "home_team": "GB", "away_team": "TB",
             "home_score": 17, "away_score": 14, "total_points": 31,
             "gameday": "2023-09-17", "gametime": "13:00",
             "kickoff_utc": pd.Timestamp("2023-09-17 18:00", tz="UTC"),
             "stadium": "Lambeau", "roof": "outdoors", "dome_indicator": 0},
        ])
        # Temps: 20, 25 → T_norm = 22.5; T_primes = -2.5, +2.5 → both |T_prime| < 8
        weather = pd.DataFrame([
            {"game_id": "g1", "temperature": 20.0, "wind_speed": 10.0, "precipitation": 0.0},
            {"game_id": "g2", "temperature": 25.0, "wind_speed": 10.0, "precipitation": 0.0},
        ])
        odds = pd.DataFrame([
            {"game_id": f"g{i}", "L_close": 44.0, "L_open": np.nan,
             "over_odds": np.nan, "under_odds": np.nan,
             "over_implied_prob": np.nan, "under_implied_prob": np.nan,
             "overround": np.nan, "p_over_vigfree": np.nan, "p_under_vigfree": np.nan}
            for i in range(1, 3)
        ])
        result = merge_and_clean(games, weather, odds)
        assert (result["E_T"] == 0).all()


class TestMissingScoresDropped:
    def test_nan_score_removed(self, sample_games_df, sample_weather_df, sample_odds_df):
        result = merge_and_clean(sample_games_df, sample_weather_df, sample_odds_df)
        # The game with NaN score should be gone
        assert "2024_01_KC_BAL" not in result["game_id"].values


class TestDomeGamesExtremenessZero:
    def test_dome_ew_et_zero(self, sample_games_df, sample_weather_df, sample_odds_df):
        valid = sample_games_df.dropna(subset=["home_score"]).copy()
        valid["home_score"] = valid["home_score"].astype(int)
        valid["away_score"] = valid["away_score"].astype(int)
        valid["total_points"] = valid["home_score"] + valid["away_score"]

        result = merge_and_clean(valid, sample_weather_df, sample_odds_df)
        dome = result[result["dome_indicator"] == 1]
        assert (dome["E_W"] == 0).all(), "Dome games should have E_W=0"
        assert (dome["E_T"] == 0).all(), "Dome games should have E_T=0"


class TestOutputColumnsComplete:
    def test_all_expected_columns(self, sample_games_df, sample_weather_df, sample_odds_df):
        from pipeline.config import EXPECTED_COLUMNS
        valid = sample_games_df.dropna(subset=["home_score"]).copy()
        valid["home_score"] = valid["home_score"].astype(int)
        valid["away_score"] = valid["away_score"].astype(int)
        valid["total_points"] = valid["home_score"] + valid["away_score"]

        result = merge_and_clean(valid, sample_weather_df, sample_odds_df)
        for col in EXPECTED_COLUMNS:
            assert col in result.columns, f"Missing output column: {col}"


class TestCleaningLogOutput:
    def test_logs_include_row_counts(self, sample_games_df, sample_weather_df, sample_odds_df, caplog):
        with caplog.at_level(logging.INFO, logger="pipeline.merge_and_clean"):
            merge_and_clean(sample_games_df, sample_weather_df, sample_odds_df)
        # Check that cleaning steps are logged
        assert "CLEAN" in caplog.text
        assert "Final dataset" in caplog.text
