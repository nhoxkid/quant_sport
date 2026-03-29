"""
Configuration for NFL Totals data pipeline.

All API keys, file paths, season ranges, and tunable constants live here.
"""
import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_GAMES_DIR = RAW_DIR / "games"
RAW_WEATHER_DIR = RAW_DIR / "weather"
RAW_ODDS_DIR = RAW_DIR / "odds"

OUTPUT_CSV = PROCESSED_DIR / "nfl_totals_weather.csv"
VALIDATION_REPORT = PROCESSED_DIR / "validation_report.txt"

# ── Seasons ────────────────────────────────────────────────────────────
DEFAULT_SEASONS = list(range(2018, 2025))  # 2018–2024 inclusive

# ── API keys ───────────────────────────────────────────────────────────
ODDS_API_KEY = os.environ.get("ODDS_API_KEY", "")

# ── NFL game data source ───────────────────────────────────────────────
NFLVERSE_SCHEDULES_URL = (
    "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
)

# ── Open-Meteo ─────────────────────────────────────────────────────────
OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_API_SLEEP = 1.0  # seconds between requests

# ── The Odds API ───────────────────────────────────────────────────────
ODDS_API_BASE = "https://api.the-odds-api.com"
ODDS_API_SPORT = "americanfootball_nfl"

# ── Derived-variable thresholds ────────────────────────────────────────
E_T_THRESHOLD = 8.0   # °C — |T_prime| >= this → E_T = 1
E_W_PERCENTILE = 90   # wind percentile for E_W flag

# ── Units (documented here for clarity) ────────────────────────────────
# temperature: °C  (from Open-Meteo)
# wind_speed:  km/h (from Open-Meteo)
# precipitation: mm (from Open-Meteo)
# L_close: total points (half-points possible, e.g. 45.5)
# odds: American format
# T_prime: °C
# E_T threshold: 8°C

# ── Expected output columns ───────────────────────────────────────────
EXPECTED_COLUMNS = [
    "game_id", "season", "week", "game_type",
    "home_team", "away_team",
    "home_score", "away_score", "total_points",
    "gameday", "gametime", "kickoff_utc",
    "stadium", "roof", "dome_indicator",
    "temperature", "wind_speed", "precipitation",
    "T_norm", "T_prime", "E_W", "E_T",
    "L_close", "L_open",
    "over_odds", "under_odds",
    "over_implied_prob", "under_implied_prob",
    "overround", "p_over_vigfree", "p_under_vigfree",
    "weather_missing", "odds_missing",
]
