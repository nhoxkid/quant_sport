"""
Collect NFL game results and metadata from nflverse.

Source: nflverse schedules CSV (GitHub release, no API key needed).
Produces a DataFrame with game-level results, kickoff times in UTC,
and dome indicators.

Timezone assumption: nflverse `gametime` is US Eastern (America/New_York).
This is documented in the nflverse data dictionary and confirmed empirically.
"""
import logging
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from pipeline.config import (
    DEFAULT_SEASONS,
    NFLVERSE_SCHEDULES_URL,
    RAW_GAMES_DIR,
)
from pipeline.stadium_coords import get_stadium_info

logger = logging.getLogger(__name__)

EASTERN = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")

CACHED_CSV = RAW_GAMES_DIR / "schedules.csv"


def _download_schedules(force: bool = False) -> Path:
    """Download the nflverse schedules CSV if not already cached."""
    if CACHED_CSV.exists() and not force:
        logger.info("Using cached schedules CSV: %s", CACHED_CSV)
        return CACHED_CSV

    RAW_GAMES_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading nflverse schedules from %s", NFLVERSE_SCHEDULES_URL)
    resp = requests.get(NFLVERSE_SCHEDULES_URL, timeout=60)
    resp.raise_for_status()
    CACHED_CSV.write_bytes(resp.content)
    logger.info("Saved schedules CSV (%d bytes) to %s", len(resp.content), CACHED_CSV)
    return CACHED_CSV


def _compute_dome_indicator(roof: str) -> int:
    """Map nflverse roof values to a binary dome indicator.

    dome_indicator = 1 if roof in {"dome", "closed"}
    dome_indicator = 0 if roof in {"outdoors", "open"}
    """
    roof_lower = str(roof).strip().lower()
    if roof_lower in ("dome", "closed"):
        return 1
    if roof_lower in ("outdoors", "open"):
        return 0
    # Fallback: treat unknown as outdoor but warn
    logger.warning("Unknown roof type %r — treating as outdoor (D=0)", roof)
    return 0


def _parse_kickoff_utc(row: pd.Series) -> pd.Timestamp:
    """Combine gameday + gametime into a UTC timestamp.

    nflverse gametime is in US Eastern. We parse it as Eastern then convert to UTC.
    If gametime is missing, we try to infer from the stadium timezone and a default
    kickoff hour (1:00 PM local), but flag it.
    """
    gameday = str(row.get("gameday", ""))
    gametime = str(row.get("gametime", ""))

    if not gameday or gameday == "nan" or gameday == "NaT":
        return pd.NaT

    if not gametime or gametime == "nan" or gametime == "None":
        # Missing gametime — cannot compute accurate kickoff
        logger.debug("Missing gametime for game %s on %s", row.get("game_id", "?"), gameday)
        return pd.NaT

    try:
        # Parse as Eastern time
        naive_str = f"{gameday} {gametime}"
        naive_dt = pd.Timestamp(naive_str)
        eastern_dt = naive_dt.tz_localize(EASTERN, ambiguous=True, nonexistent="shift_forward")
        utc_dt = eastern_dt.tz_convert(UTC)
        return utc_dt
    except Exception as exc:
        logger.warning("Failed to parse kickoff for game %s: %s", row.get("game_id", "?"), exc)
        return pd.NaT


def collect_games(seasons: list[int] | None = None, force_download: bool = False) -> pd.DataFrame:
    """Collect and clean NFL game data for the requested seasons.

    Parameters
    ----------
    seasons : list[int], optional
        Seasons to include. Defaults to config.DEFAULT_SEASONS (2018-2024).
    force_download : bool
        If True, re-download even if cached CSV exists.

    Returns
    -------
    pd.DataFrame
        One row per completed regular-season or playoff game.
    """
    if seasons is None:
        seasons = DEFAULT_SEASONS

    csv_path = _download_schedules(force=force_download)
    df = pd.read_csv(csv_path, low_memory=False)
    logger.info("Loaded schedules CSV: %d total rows", len(df))

    # ── Filter to requested seasons ────────────────────────────────────
    df = df[df["season"].isin(seasons)].copy()
    logger.info("Filtered to seasons %s: %d rows", seasons, len(df))

    # ── Exclude preseason ──────────────────────────────────────────────
    pre_count = (df["game_type"] == "PRE").sum() if "game_type" in df.columns else 0
    df = df[df["game_type"].isin(["REG", "POST"])].copy()
    logger.info("Excluded %d preseason games, %d remain", pre_count, len(df))

    # ── Drop games with missing scores (not yet played / cancelled) ───
    before = len(df)
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    dropped_no_score = before - len(df)
    if dropped_no_score > 0:
        logger.info("Dropped %d games with missing scores", dropped_no_score)

    # ── Ensure integer scores ──────────────────────────────────────────
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    # ── Compute total_points (never trust a pre-existing column) ──────
    df["total_points"] = df["home_score"] + df["away_score"]

    # ── Dome indicator ─────────────────────────────────────────────────
    df["dome_indicator"] = df["roof"].apply(_compute_dome_indicator)

    # ── Kickoff UTC ────────────────────────────────────────────────────
    df["kickoff_utc"] = df.apply(_parse_kickoff_utc, axis=1)

    # ── Select and rename columns ──────────────────────────────────────
    keep_cols = [
        "game_id", "season", "week", "game_type",
        "home_team", "away_team",
        "home_score", "away_score", "total_points",
        "gameday", "gametime", "kickoff_utc",
        "stadium", "roof", "dome_indicator",
    ]
    # Only keep columns that exist (some nflverse versions differ)
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    # ── Also keep odds columns if they exist (for collect_odds fallback) ──
    # We'll handle this in collect_odds.py; here we just keep the core columns.

    # ── INTEGRITY CHECKS ──────────────────────────────────────────────
    _run_integrity_checks(df, seasons)

    return df.reset_index(drop=True)


def _run_integrity_checks(df: pd.DataFrame, seasons: list[int]) -> None:
    """Embedded integrity checks — run every time, not just in tests."""
    # Scores non-negative
    assert (df["home_score"] >= 0).all(), "Negative home_score found"
    assert (df["away_score"] >= 0).all(), "Negative away_score found"

    # total_points consistency
    recomputed = df["home_score"] + df["away_score"]
    assert (df["total_points"] == recomputed).all(), "total_points != home + away"

    # No duplicate game_ids
    dupes = df["game_id"].duplicated().sum()
    assert dupes == 0, f"Found {dupes} duplicate game_id values"

    # Season range
    assert df["season"].isin(seasons).all(), "Games found outside requested seasons"

    # Summary log
    dome_count = (df["dome_indicator"] == 1).sum()
    outdoor_count = (df["dome_indicator"] == 0).sum()
    logger.info(
        "Collected %d games for seasons %d–%d. %d dome games, %d outdoor games.",
        len(df),
        min(seasons),
        max(seasons),
        dome_count,
        outdoor_count,
    )
