"""Tests for pipeline.collect_games using synthetic fixture data."""
import numpy as np
import pandas as pd
import pytest


class TestCollectGames:
    def test_total_points_computed_correctly(self, sample_games_df):
        valid = sample_games_df.dropna(subset=["home_score", "away_score"])
        recomputed = valid["home_score"] + valid["away_score"]
        assert (valid["total_points"] == recomputed).all()

    def test_dome_indicator_logic(self, sample_games_df):
        for _, row in sample_games_df.iterrows():
            roof = str(row["roof"]).lower()
            if roof in ("dome", "closed"):
                assert row["dome_indicator"] == 1, f"roof={roof} should be dome=1"
            elif roof in ("outdoors", "open"):
                assert row["dome_indicator"] == 0, f"roof={roof} should be dome=0"

    def test_no_duplicate_game_ids(self, sample_games_df):
        assert sample_games_df["game_id"].duplicated().sum() == 0

    def test_required_columns_present(self, sample_games_df):
        required = [
            "game_id", "season", "week", "game_type",
            "home_team", "away_team",
            "home_score", "away_score", "total_points",
            "gameday", "gametime", "kickoff_utc",
            "stadium", "roof", "dome_indicator",
        ]
        for col in required:
            assert col in sample_games_df.columns, f"Missing column: {col}"

    def test_kickoff_datetime_valid(self, sample_games_df):
        # All completed games should have valid kickoff_utc
        completed = sample_games_df.dropna(subset=["home_score"])
        for _, row in completed.iterrows():
            assert pd.notna(row["kickoff_utc"]), (
                f"Game {row['game_id']} has NaT kickoff_utc"
            )

    def test_preseason_excluded(self, sample_games_df):
        # Our fixture has no preseason, verify game_type filtering logic
        assert "PRE" not in sample_games_df["game_type"].values

    def test_dome_indicator_from_function(self):
        from pipeline.collect_games import _compute_dome_indicator
        assert _compute_dome_indicator("dome") == 1
        assert _compute_dome_indicator("closed") == 1
        assert _compute_dome_indicator("outdoors") == 0
        assert _compute_dome_indicator("open") == 0

    def test_scores_non_negative(self, sample_games_df):
        valid = sample_games_df.dropna(subset=["home_score"])
        assert (valid["home_score"] >= 0).all()
        assert (valid["away_score"] >= 0).all()
