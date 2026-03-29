"""
Shared pytest fixtures for the NFL totals pipeline tests.

Provides synthetic DataFrames that mirror real data structure without
requiring API access.
"""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_games_df():
    """10 games: mix of dome/outdoor, different weeks/seasons, one with missing score."""
    data = [
        # Outdoor games
        {"game_id": "2023_01_KC_DET", "season": 2023, "week": 1, "game_type": "REG",
         "home_team": "KC", "away_team": "DET",
         "home_score": 21, "away_score": 20, "total_points": 41,
         "gameday": "2023-09-07", "gametime": "20:20",
         "kickoff_utc": pd.Timestamp("2023-09-08 00:20", tz="UTC"),
         "stadium": "GEHA Field", "roof": "outdoors", "dome_indicator": 0},
        {"game_id": "2023_02_BUF_NYJ", "season": 2023, "week": 2, "game_type": "REG",
         "home_team": "BUF", "away_team": "NYJ",
         "home_score": 22, "away_score": 16, "total_points": 38,
         "gameday": "2023-09-11", "gametime": "20:15",
         "kickoff_utc": pd.Timestamp("2023-09-12 00:15", tz="UTC"),
         "stadium": "Highmark Stadium", "roof": "outdoors", "dome_indicator": 0},
        {"game_id": "2023_03_GB_CHI", "season": 2023, "week": 3, "game_type": "REG",
         "home_team": "GB", "away_team": "CHI",
         "home_score": 38, "away_score": 20, "total_points": 58,
         "gameday": "2023-09-17", "gametime": "13:00",
         "kickoff_utc": pd.Timestamp("2023-09-17 17:00", tz="UTC"),
         "stadium": "Lambeau Field", "roof": "outdoors", "dome_indicator": 0},
        {"game_id": "2023_10_NE_MIA", "season": 2023, "week": 10, "game_type": "REG",
         "home_team": "MIA", "away_team": "NE",
         "home_score": 31, "away_score": 17, "total_points": 48,
         "gameday": "2023-11-12", "gametime": "13:00",
         "kickoff_utc": pd.Timestamp("2023-11-12 18:00", tz="UTC"),
         "stadium": "Hard Rock Stadium", "roof": "outdoors", "dome_indicator": 0},
        {"game_id": "2022_18_BUF_CIN", "season": 2022, "week": 18, "game_type": "REG",
         "home_team": "BUF", "away_team": "CIN",
         "home_score": 35, "away_score": 31, "total_points": 66,
         "gameday": "2023-01-08", "gametime": "16:30",
         "kickoff_utc": pd.Timestamp("2023-01-08 21:30", tz="UTC"),
         "stadium": "Highmark Stadium", "roof": "outdoors", "dome_indicator": 0},
        # Dome games
        {"game_id": "2023_05_DAL_ATL", "season": 2023, "week": 5, "game_type": "REG",
         "home_team": "ATL", "away_team": "DAL",
         "home_score": 24, "away_score": 31, "total_points": 55,
         "gameday": "2023-10-08", "gametime": "13:00",
         "kickoff_utc": pd.Timestamp("2023-10-08 17:00", tz="UTC"),
         "stadium": "Mercedes-Benz Stadium", "roof": "dome", "dome_indicator": 1},
        {"game_id": "2023_06_IND_JAX", "season": 2023, "week": 6, "game_type": "REG",
         "home_team": "IND", "away_team": "JAX",
         "home_score": 31, "away_score": 21, "total_points": 52,
         "gameday": "2023-10-15", "gametime": "13:00",
         "kickoff_utc": pd.Timestamp("2023-10-15 17:00", tz="UTC"),
         "stadium": "Lucas Oil Stadium", "roof": "dome", "dome_indicator": 1},
        {"game_id": "2023_07_MIN_CHI", "season": 2023, "week": 7, "game_type": "REG",
         "home_team": "MIN", "away_team": "CHI",
         "home_score": 19, "away_score": 13, "total_points": 32,
         "gameday": "2023-10-22", "gametime": "20:20",
         "kickoff_utc": pd.Timestamp("2023-10-23 00:20", tz="UTC"),
         "stadium": "US Bank Stadium", "roof": "dome", "dome_indicator": 1},
        # Playoff game
        {"game_id": "2023_19_KC_BAL", "season": 2023, "week": 19, "game_type": "POST",
         "home_team": "BAL", "away_team": "KC",
         "home_score": 10, "away_score": 17, "total_points": 27,
         "gameday": "2024-01-28", "gametime": "15:00",
         "kickoff_utc": pd.Timestamp("2024-01-28 20:00", tz="UTC"),
         "stadium": "M&T Bank Stadium", "roof": "outdoors", "dome_indicator": 0},
        # Game with missing score (not yet played)
        {"game_id": "2024_01_KC_BAL", "season": 2024, "week": 1, "game_type": "REG",
         "home_team": "BAL", "away_team": "KC",
         "home_score": np.nan, "away_score": np.nan, "total_points": np.nan,
         "gameday": "2024-09-05", "gametime": "20:20",
         "kickoff_utc": pd.Timestamp("2024-09-06 00:20", tz="UTC"),
         "stadium": "M&T Bank Stadium", "roof": "outdoors", "dome_indicator": 0},
    ]
    return pd.DataFrame(data)


@pytest.fixture
def sample_weather_df():
    """Weather data matching the outdoor games in sample_games_df."""
    data = [
        {"game_id": "2023_01_KC_DET", "temperature": 28.5, "wind_speed": 12.0, "precipitation": 0.0},
        {"game_id": "2023_02_BUF_NYJ", "temperature": 22.0, "wind_speed": 18.5, "precipitation": 0.0},
        {"game_id": "2023_03_GB_CHI", "temperature": 24.0, "wind_speed": 10.0, "precipitation": 0.5},
        {"game_id": "2023_10_NE_MIA", "temperature": 30.0, "wind_speed": 15.0, "precipitation": 0.0},
        {"game_id": "2022_18_BUF_CIN", "temperature": -2.0, "wind_speed": 25.0, "precipitation": 1.0},
        {"game_id": "2023_05_DAL_ATL", "temperature": np.nan, "wind_speed": np.nan, "precipitation": np.nan},
        {"game_id": "2023_06_IND_JAX", "temperature": np.nan, "wind_speed": np.nan, "precipitation": np.nan},
        {"game_id": "2023_07_MIN_CHI", "temperature": np.nan, "wind_speed": np.nan, "precipitation": np.nan},
        {"game_id": "2023_19_KC_BAL", "temperature": 5.0, "wind_speed": 22.0, "precipitation": 0.0},
        {"game_id": "2024_01_KC_BAL", "temperature": np.nan, "wind_speed": np.nan, "precipitation": np.nan},
    ]
    return pd.DataFrame(data)


@pytest.fixture
def sample_odds_df():
    """Odds for 8 of the 10 games (to test partial coverage)."""
    data = [
        {"game_id": "2023_01_KC_DET", "L_close": 53.5, "L_open": 52.0,
         "over_odds": -110, "under_odds": -110},
        {"game_id": "2023_02_BUF_NYJ", "L_close": 42.5, "L_open": 43.0,
         "over_odds": -105, "under_odds": -115},
        {"game_id": "2023_03_GB_CHI", "L_close": 44.0, "L_open": 44.5,
         "over_odds": -110, "under_odds": -110},
        {"game_id": "2023_10_NE_MIA", "L_close": 46.0, "L_open": 45.5,
         "over_odds": +100, "under_odds": -120},
        {"game_id": "2022_18_BUF_CIN", "L_close": 49.0, "L_open": 48.5,
         "over_odds": -110, "under_odds": -110},
        {"game_id": "2023_05_DAL_ATL", "L_close": 47.5, "L_open": 47.0,
         "over_odds": -110, "under_odds": -110},
        {"game_id": "2023_06_IND_JAX", "L_close": 44.5, "L_open": 44.0,
         "over_odds": -115, "under_odds": -105},
        {"game_id": "2023_19_KC_BAL", "L_close": 44.0, "L_open": 44.5,
         "over_odds": -110, "under_odds": -110},
        # game 2023_07_MIN_CHI and 2024_01_KC_BAL intentionally missing
    ]
    df = pd.DataFrame(data)
    # Compute derived columns
    from pipeline.utils import implied_probability, vig_free_probabilities, overround
    df["over_implied_prob"] = np.nan
    df["under_implied_prob"] = np.nan
    df["overround"] = np.nan
    df["p_over_vigfree"] = np.nan
    df["p_under_vigfree"] = np.nan

    for idx, row in df.iterrows():
        if pd.notna(row["over_odds"]) and pd.notna(row["under_odds"]):
            p_o = implied_probability(int(row["over_odds"]))
            p_u = implied_probability(int(row["under_odds"]))
            df.at[idx, "over_implied_prob"] = p_o
            df.at[idx, "under_implied_prob"] = p_u
            df.at[idx, "overround"] = overround(p_o, p_u)
            vf_o, vf_u = vig_free_probabilities(p_o, p_u)
            df.at[idx, "p_over_vigfree"] = vf_o
            df.at[idx, "p_under_vigfree"] = vf_u

    return df


@pytest.fixture
def sample_stadium_coords():
    """Subset of 5 stadiums with known coords/timezones."""
    return {
        "GB":  {"lat": 44.5013, "lon": -88.0622, "timezone": "America/Chicago",     "roof_type": "outdoors"},
        "KC":  {"lat": 39.0489, "lon": -94.4839, "timezone": "America/Chicago",     "roof_type": "outdoors"},
        "ATL": {"lat": 33.7554, "lon": -84.4010, "timezone": "America/New_York",    "roof_type": "dome"},
        "MIA": {"lat": 25.9580, "lon": -80.2389, "timezone": "America/New_York",    "roof_type": "outdoors"},
        "SF":  {"lat": 37.4033, "lon": -121.9694,"timezone": "America/Los_Angeles",  "roof_type": "outdoors"},
    }
