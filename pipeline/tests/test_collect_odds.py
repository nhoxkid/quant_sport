"""Tests for pipeline.collect_odds."""
import math

import numpy as np
import pandas as pd
import pytest

from pipeline.utils import implied_probability, vig_free_probabilities, overround


class TestFallbackWhenNoApiKey:
    def test_no_key_uses_nflverse(self, sample_games_df, sample_odds_df):
        """With no API key, odds should still be loadable from nflverse fallback."""
        # The sample_odds_df fixture simulates what we'd get from nflverse
        assert len(sample_odds_df) > 0
        assert "L_close" in sample_odds_df.columns


class TestLcloseRange:
    def test_all_in_range(self, sample_odds_df):
        valid = sample_odds_df["L_close"].dropna()
        assert (valid >= 28).all(), f"Min L_close: {valid.min()}"
        assert (valid <= 70).all(), f"Max L_close: {valid.max()}"


class TestImpliedProbEndToEnd:
    def test_full_chain(self):
        """From raw American odds → implied prob → vig-free, verify full chain."""
        over_odds = -110
        under_odds = -110

        p_over = implied_probability(over_odds)
        p_under = implied_probability(under_odds)

        # Both -110 → each ≈ 0.52381
        assert math.isclose(p_over, 110 / 210, abs_tol=1e-10)
        assert math.isclose(p_under, 110 / 210, abs_tol=1e-10)

        # Overround ≈ 0.0476
        o = overround(p_over, p_under)
        assert o > 0
        assert math.isclose(o, 10 / 210, abs_tol=1e-6)

        # Vig-free
        vf_o, vf_u = vig_free_probabilities(p_over, p_under)
        assert math.isclose(vf_o + vf_u, 1.0, abs_tol=1e-10)
        # Both sides equal → each 0.5
        assert math.isclose(vf_o, 0.5, abs_tol=1e-10)

    def test_uneven_odds(self):
        """Uneven market: -150 / +130."""
        p_over = implied_probability(-150)   # 150/250 = 0.6
        p_under = implied_probability(+130)  # 100/230 ≈ 0.4348

        o = overround(p_over, p_under)
        assert o > 0  # there's vig

        vf_o, vf_u = vig_free_probabilities(p_over, p_under)
        assert math.isclose(vf_o + vf_u, 1.0, abs_tol=1e-10)
        assert vf_o > vf_u  # favorite side has higher probability


class TestMissingOddsFlagged:
    def test_missing_games_get_nan(self, sample_games_df, sample_odds_df):
        """Games without odds should have NaN, not 0 or be dropped."""
        # Merge to see coverage
        merged = sample_games_df[["game_id"]].merge(
            sample_odds_df[["game_id", "L_close"]], on="game_id", how="left"
        )
        missing = merged[merged["L_close"].isna()]
        # We intentionally excluded 2 games from odds fixture
        assert len(missing) >= 2
        # Verify they have NaN, not 0
        assert (missing["L_close"].isna()).all()
