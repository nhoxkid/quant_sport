"""
Collect weather data at kickoff for outdoor NFL games using Open-Meteo.

Source: Open-Meteo Historical Weather API (free, no key).
Returns hourly temperature (°C), wind_speed (km/h), precipitation (mm).

Strategy: batch requests by stadium — one request per stadium covers the full
date range for all games at that stadium in a season.  This minimizes API calls.

Units (from Open-Meteo, NOT converted):
  temperature_2m: °C
  wind_speed_10m: km/h
  precipitation:  mm
"""
import json
import logging
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from pipeline.config import (
    OPEN_METEO_URL,
    RAW_WEATHER_DIR,
    WEATHER_API_SLEEP,
)
from pipeline.stadium_coords import get_stadium_info, STADIUM_COORDS

logger = logging.getLogger(__name__)

# ── Spot-check games for sanity verification ──────────────────────────
# (game_id, expected description for log)
SPOT_CHECKS = [
    # January Packers home game — should be cold
    {"month": 1, "team": "GB", "expect": "cold (< 5°C)"},
    # September Dolphins home game — should be warm
    {"month": 9, "team": "MIA", "expect": "warm (> 20°C)"},
    # December Bills home game — should be cold
    {"month": 12, "team": "BUF", "expect": "cold (< 10°C)"},
]


def _cache_path(team: str, season: int) -> Path:
    """Return the cache file path for a team-season weather response."""
    RAW_WEATHER_DIR.mkdir(parents=True, exist_ok=True)
    return RAW_WEATHER_DIR / f"weather_{team}_{season}.json"


def _fetch_weather_batch(
    lat: float, lon: float, start_date: str, end_date: str
) -> dict | None:
    """Fetch hourly weather from Open-Meteo for a date range.

    Returns the parsed JSON response, or None on failure.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,wind_speed_10m,precipitation",
        "timezone": "UTC",  # request data in UTC to match our kickoff_utc
    }
    try:
        resp = requests.get(OPEN_METEO_URL, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.warning("Open-Meteo request failed (lat=%.2f, lon=%.2f, %s–%s): %s",
                       lat, lon, start_date, end_date, exc)
        return None


def _round_to_nearest_hour(ts: pd.Timestamp) -> pd.Timestamp:
    """Round a timestamp to the nearest hour.

    13:22 → 13:00,  13:42 → 14:00,  13:30 → 14:00 (round half up)
    """
    if pd.isna(ts):
        return pd.NaT
    # Add 30 minutes then floor to hour
    rounded = (ts + pd.Timedelta(minutes=30)).floor("h")
    return rounded


def _extract_hour_weather(weather_json: dict, target_utc: pd.Timestamp) -> dict:
    """Extract weather values for a specific UTC hour from a batch response.

    Returns dict with temperature, wind_speed, precipitation (or NaNs on failure).
    """
    nan_result = {"temperature": float("nan"), "wind_speed": float("nan"), "precipitation": float("nan")}

    if weather_json is None:
        return nan_result

    hourly = weather_json.get("hourly", {})
    times = hourly.get("time", [])
    temps = hourly.get("temperature_2m", [])
    winds = hourly.get("wind_speed_10m", [])
    precips = hourly.get("precipitation", [])

    if not times:
        return nan_result

    # Target hour string in ISO format matching Open-Meteo output
    target_str = target_utc.strftime("%Y-%m-%dT%H:%M")

    try:
        idx = times.index(target_str)
    except ValueError:
        logger.warning("Hour %s not found in weather data (available: %s ... %s)",
                       target_str, times[0] if times else "?", times[-1] if times else "?")
        return nan_result

    return {
        "temperature": temps[idx] if idx < len(temps) and temps[idx] is not None else float("nan"),
        "wind_speed": winds[idx] if idx < len(winds) and winds[idx] is not None else float("nan"),
        "precipitation": precips[idx] if idx < len(precips) and precips[idx] is not None else float("nan"),
    }


def _validate_kickoff_local_hour(kickoff_utc: pd.Timestamp, timezone_str: str, game_id: str) -> None:
    """Warn if the local kickoff hour is unreasonable (outside 11am-11pm)."""
    if pd.isna(kickoff_utc):
        return
    local_tz = ZoneInfo(timezone_str)
    local_time = kickoff_utc.astimezone(local_tz)
    local_hour = local_time.hour
    if local_hour < 11 or local_hour > 23:
        logger.warning(
            "Suspicious local kickoff time for %s: %s local (%s). "
            "Expected 11am-11pm. Possible timezone bug.",
            game_id, local_time.strftime("%H:%M"), timezone_str,
        )


def collect_weather(games_df: pd.DataFrame) -> pd.DataFrame:
    """Collect weather data for all outdoor games in the DataFrame.

    For dome games (dome_indicator == 1), weather columns are set to NaN.
    Uses batched requests per stadium-season and caches raw JSON.

    Parameters
    ----------
    games_df : pd.DataFrame
        Must contain: game_id, home_team, season, dome_indicator, kickoff_utc, gameday

    Returns
    -------
    pd.DataFrame
        Columns: game_id, temperature, wind_speed, precipitation
    """
    results = []
    outdoor = games_df[games_df["dome_indicator"] == 0].copy()
    dome = games_df[games_df["dome_indicator"] == 1].copy()

    logger.info("Weather collection: %d outdoor games, %d dome games (skipped)",
                len(outdoor), len(dome))

    # Add NaN rows for dome games
    for _, row in dome.iterrows():
        results.append({
            "game_id": row["game_id"],
            "temperature": float("nan"),
            "wind_speed": float("nan"),
            "precipitation": float("nan"),
        })

    if outdoor.empty:
        return pd.DataFrame(results)

    # ── Batch by (home_team, season) ──────────────────────────────────
    grouped = outdoor.groupby(["home_team", "season"])
    failures = 0

    for (team, season), group in grouped:
        cache_file = _cache_path(team, season)

        # Try to get stadium info; handle relocated teams
        try:
            info = get_stadium_info(team, season)
        except KeyError:
            logger.warning("No stadium data for team %s season %d — skipping %d games",
                           team, season, len(group))
            for _, row in group.iterrows():
                results.append({
                    "game_id": row["game_id"],
                    "temperature": float("nan"),
                    "wind_speed": float("nan"),
                    "precipitation": float("nan"),
                })
                failures += 1
            continue

        lat, lon = info["lat"], info["lon"]
        tz_str = info["timezone"]

        # Check cache
        if cache_file.exists():
            logger.debug("Using cached weather for %s %d", team, season)
            with open(cache_file, "r") as f:
                weather_json = json.load(f)
        else:
            # Determine date range for this group
            dates = pd.to_datetime(group["gameday"])
            start_date = dates.min().strftime("%Y-%m-%d")
            end_date = dates.max().strftime("%Y-%m-%d")

            logger.info("Fetching weather for %s %d (%s to %s, %d games)",
                        team, season, start_date, end_date, len(group))

            weather_json = _fetch_weather_batch(lat, lon, start_date, end_date)
            if weather_json is not None:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_file, "w") as f:
                    json.dump(weather_json, f)
            time.sleep(WEATHER_API_SLEEP)

        # Extract per-game weather
        for _, row in group.iterrows():
            kickoff = row["kickoff_utc"]
            game_id = row["game_id"]

            if pd.isna(kickoff):
                logger.warning("No kickoff_utc for outdoor game %s — weather will be NaN", game_id)
                results.append({
                    "game_id": game_id,
                    "temperature": float("nan"),
                    "wind_speed": float("nan"),
                    "precipitation": float("nan"),
                })
                failures += 1
                continue

            # Validate local kickoff hour
            _validate_kickoff_local_hour(kickoff, tz_str, game_id)

            # Round to nearest hour
            rounded = _round_to_nearest_hour(kickoff)
            wx = _extract_hour_weather(weather_json, rounded)

            # Range checks
            t = wx["temperature"]
            w = wx["wind_speed"]
            p = wx["precipitation"]
            if not pd.isna(t) and (t < -40 or t > 45):
                logger.warning("Game %s: temperature %.1f°C out of range [-40, 45]", game_id, t)
            if not pd.isna(w) and (w < 0 or w > 120):
                logger.warning("Game %s: wind_speed %.1f km/h out of range [0, 120]", game_id, w)
            if not pd.isna(p) and (p < 0 or p > 100):
                logger.warning("Game %s: precipitation %.1f mm out of range [0, 100]", game_id, p)

            if pd.isna(t) and pd.isna(w) and pd.isna(p):
                failures += 1

            results.append({"game_id": game_id, **wx})

    weather_df = pd.DataFrame(results)

    # ── Spot checks ───────────────────────────────────────────────────
    _run_spot_checks(weather_df, games_df)

    outdoor_with_wx = weather_df.merge(
        games_df[["game_id", "dome_indicator"]], on="game_id"
    )
    outdoor_with_wx = outdoor_with_wx[outdoor_with_wx["dome_indicator"] == 0]
    wx_success = outdoor_with_wx["temperature"].notna().sum()
    wx_total = len(outdoor_with_wx)

    logger.info(
        "Weather collection complete. %d/%d outdoor games with weather data. %d failures.",
        wx_success, wx_total, failures,
    )

    return weather_df[["game_id", "temperature", "wind_speed", "precipitation"]]


def _run_spot_checks(weather_df: pd.DataFrame, games_df: pd.DataFrame) -> None:
    """Run spot checks on known game types for sanity."""
    merged = weather_df.merge(games_df[["game_id", "home_team", "gameday", "dome_indicator"]], on="game_id")

    for check in SPOT_CHECKS:
        month = check["month"]
        team = check["team"]
        expect = check["expect"]

        mask = (
            (merged["home_team"] == team)
            & (merged["dome_indicator"] == 0)
            & (pd.to_datetime(merged["gameday"]).dt.month == month)
            & merged["temperature"].notna()
        )
        subset = merged[mask]
        if subset.empty:
            logger.info("Spot-check: no %s home games in month %d with weather data", team, month)
            continue

        mean_temp = subset["temperature"].mean()
        logger.info(
            "SPOT-CHECK: %s home games in month %d — mean temp %.1f°C (expected %s)",
            team, month, mean_temp, expect,
        )
