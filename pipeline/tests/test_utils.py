"""Tests for pipeline.utils — implied probability, vig-free, overround."""
import math
import logging

import pytest

from pipeline.utils import implied_probability, vig_free_probabilities, overround


class TestImpliedProbability:
    def test_positive_odds_150(self):
        # O=+150 → 100/(150+100) = 100/250 = 0.4
        assert math.isclose(implied_probability(150), 0.4, abs_tol=1e-10)

    def test_positive_odds_200(self):
        # O=+200 → 100/300 = 0.3333...
        assert math.isclose(implied_probability(200), 1 / 3, abs_tol=1e-10)

    def test_negative_odds_110(self):
        # O=-110 → 110/210 ≈ 0.52381
        assert math.isclose(implied_probability(-110), 110 / 210, abs_tol=1e-10)

    def test_negative_odds_150(self):
        # O=-150 → 150/250 = 0.6
        assert math.isclose(implied_probability(-150), 0.6, abs_tol=1e-10)

    def test_edge_positive_100(self):
        # O=+100 → 100/200 = 0.5
        assert math.isclose(implied_probability(100), 0.5, abs_tol=1e-10)

    def test_edge_negative_100(self):
        # O=-100 → 100/200 = 0.5
        assert math.isclose(implied_probability(-100), 0.5, abs_tol=1e-10)

    def test_invalid_zero_raises(self):
        with pytest.raises(ValueError, match="invalid"):
            implied_probability(0)


class TestVigFree:
    def test_sums_to_one_even(self):
        p_o = implied_probability(-110)
        p_u = implied_probability(-110)
        vf_o, vf_u = vig_free_probabilities(p_o, p_u)
        assert math.isclose(vf_o + vf_u, 1.0, abs_tol=1e-10)

    def test_sums_to_one_uneven(self):
        p_o = implied_probability(-150)
        p_u = implied_probability(+130)
        vf_o, vf_u = vig_free_probabilities(p_o, p_u)
        assert math.isclose(vf_o + vf_u, 1.0, abs_tol=1e-10)

    def test_sums_to_one_fair(self):
        # Both +100 → each 0.5, vig-free still 0.5 each
        p_o = implied_probability(100)
        p_u = implied_probability(100)
        vf_o, vf_u = vig_free_probabilities(p_o, p_u)
        assert math.isclose(vf_o, 0.5, abs_tol=1e-10)
        assert math.isclose(vf_u, 0.5, abs_tol=1e-10)

    def test_invalid_negative_raises(self):
        with pytest.raises(ValueError):
            vig_free_probabilities(-0.1, 0.5)

    def test_invalid_one_raises(self):
        with pytest.raises(ValueError):
            vig_free_probabilities(1.0, 0.5)

    def test_invalid_zero_raises(self):
        with pytest.raises(ValueError):
            vig_free_probabilities(0.0, 0.5)


class TestOverround:
    def test_typical_both_minus_110(self):
        # Both -110 → each 110/210 ≈ 0.52381 → overround ≈ 0.04762
        p = implied_probability(-110)
        o = overround(p, p)
        assert math.isclose(o, 2 * p - 1, abs_tol=1e-10)
        assert math.isclose(o, 10 / 210, abs_tol=1e-6)

    def test_fair_market(self):
        # Both +100 → each 0.5 → overround = 0.0
        p = implied_probability(100)
        o = overround(p, p)
        assert math.isclose(o, 0.0, abs_tol=1e-10)

    def test_warns_on_negative(self, caplog):
        # Force a scenario where overround < 0 (under-round)
        # p_over=0.4, p_under=0.4 → overround = -0.2
        with caplog.at_level(logging.WARNING, logger="pipeline.utils"):
            o = overround(0.4, 0.4)
        assert o < 0
        assert "Negative overround" in caplog.text
