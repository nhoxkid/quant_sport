"""Tests for pipeline.validate — feed known-good and known-bad data."""
import random

import numpy as np
import pandas as pd
import pytest

from pipeline.validate import validate
from pipeline.config import EXPECTED_COLUMNS


def _make_good_df(n: int = 260) -> pd.DataFrame:
    """Create a synthetic DataFrame that passes all validation checks.

    Produces n games for season 2023 with realistic-ish total points and
    correlated L_close values so cross-checks pass.
    """
    rng = random.Random(42)
    rows = []
    for i in range(n):
        is_dome = i % 4 == 0
        gid = f"2023_{i:04d}_HOM_AWY"
        # Realistic score generation: mean total ~45
        base_total = rng.gauss(45, 10)
        base_total = max(14, min(110, base_total))
        home_s = int(max(0, base_total * rng.uniform(0.4, 0.6)))
        away_s = int(max(0, base_total - home_s))
        total = home_s + away_s

        # L_close correlated with total_points (r > 0.3)
        l_close = total + rng.gauss(0, 5)
        l_close = max(30, min(65, round(l_close * 2) / 2))  # half-point snap

        # Generate valid day within month range
        day = 1 + (i % 28)
        month = 9 + (i % 4)  # Sep, Oct, Nov, Dec
        if month > 12:
            month = 12
        gameday = f"2023-{month:02d}-{day:02d}"

        # Temperature: warm in Sep, cold in Dec
        if is_dome:
            temp = np.nan
            wind = np.nan
            precip = np.nan
            t_norm = np.nan
            t_prime = np.nan
        else:
            base_temp = {9: 24.0, 10: 15.0, 11: 5.0, 12: -2.0}.get(month, 15.0)
            temp = base_temp + rng.gauss(0, 3)
            temp = max(-35, min(42, temp))
            wind = max(0, min(100, rng.gauss(15, 8)))
            precip = max(0, min(50, rng.expovariate(1) * 2))
            t_norm = base_temp
            t_prime = temp - t_norm

        rows.append({
            "game_id": gid,
            "season": 2023,
            "week": (i % 18) + 1,
            "game_type": "REG",
            "home_team": "KC" if not is_dome else "ATL",
            "away_team": "DET",
            "home_score": home_s,
            "away_score": away_s,
            "total_points": total,
            "gameday": gameday,
            "gametime": "13:00",
            "kickoff_utc": f"{gameday}T17:00:00+00:00",
            "stadium": "Stadium",
            "roof": "dome" if is_dome else "outdoors",
            "dome_indicator": 1 if is_dome else 0,
            "temperature": temp,
            "wind_speed": wind,
            "precipitation": precip,
            "T_norm": t_norm,
            "T_prime": t_prime,
            "E_W": 0,
            "E_T": 0 if is_dome else (1 if t_prime is not np.nan and abs(t_prime) >= 8 else 0),
            "L_close": l_close,
            "L_open": l_close - rng.choice([-0.5, 0, 0.5]),
            "over_odds": -110.0,
            "under_odds": -110.0,
            "over_implied_prob": 110 / 210,
            "under_implied_prob": 110 / 210,
            "overround": 10 / 210,
            "p_over_vigfree": 0.5,
            "p_under_vigfree": 0.5,
            "weather_missing": 0,
            "odds_missing": 0,
        })
    return pd.DataFrame(rows)


class TestKnownGoodData:
    def test_all_checks_pass(self):
        df = _make_good_df(260)
        report, passed, failed = validate(df=df)
        assert failed == 0, f"Expected 0 failures:\n{report}"
        assert passed > 0


class TestDuplicateGameId:
    def test_duplicate_detected(self):
        df = _make_good_df(260)
        df.loc[1, "game_id"] = df.loc[0, "game_id"]
        report, passed, failed = validate(df=df)
        assert failed > 0
        assert "duplicate" in report.lower() or "dupes" in report.lower()


class TestTotalPointsInconsistency:
    def test_mismatch_detected(self):
        df = _make_good_df(260)
        df.loc[0, "total_points"] = 999
        report, passed, failed = validate(df=df)
        assert failed > 0


class TestLcloseOutOfRange:
    def test_out_of_range_detected(self):
        df = _make_good_df(260)
        df.loc[0, "L_close"] = 150.0
        report, passed, failed = validate(df=df)
        assert failed > 0


class TestDomeWeatherNotNan:
    def test_dome_with_temperature_detected(self):
        df = _make_good_df(260)
        dome_idx = df[df["dome_indicator"] == 1].index[0]
        df.loc[dome_idx, "temperature"] = 25.0
        report, passed, failed = validate(df=df)
        assert failed > 0
