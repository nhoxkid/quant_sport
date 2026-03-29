"""
Collect betting odds (totals) for NFL games.

Primary path: use nflverse schedules CSV which often includes total_line
and related odds columns.  This is the simplest and most reliable source.

Secondary path: The Odds API (requires paid key).  Falls back to nflverse
if the key is empty or the request fails.

Fields produced per game:
  L_close      — closing total line (REQUIRED for analysis)
  L_open       — opening total line (if available)
  over_odds    — American odds for over at close
  under_odds   — American odds for under at close
  over_implied_prob, under_implied_prob — raw implied probs
  overround    — vig
  p_over_vigfree, p_under_vigfree — vig-free probabilities
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from pipeline.config import (
    ODDS_API_KEY,
    ODDS_API_BASE,
    ODDS_API_SPORT,
    RAW_ODDS_DIR,
    RAW_GAMES_DIR,
)
from pipeline.utils import implied_probability, vig_free_probabilities, overround

logger = logging.getLogger(__name__)


def _load_nflverse_odds() -> pd.DataFrame:
    """Load odds data from the cached nflverse schedules CSV.

    nflverse schedules may include columns like:
      total_line, under_odds, over_odds, spread_line, etc.
    Column names vary by version; we handle the common variants.
    """
    cached = RAW_GAMES_DIR / "schedules.csv"
    if not cached.exists():
        logger.warning("No cached schedules CSV for odds fallback at %s", cached)
        return pd.DataFrame()

    df = pd.read_csv(cached, low_memory=False)

    # Map possible column names to our standard names
    col_map = {}
    # Total line (closing)
    for c in ("total_line", "total", "over_under_line"):
        if c in df.columns:
            col_map[c] = "L_close"
            break

    # Over odds
    for c in ("over_odds", "over_line"):
        if c in df.columns:
            col_map[c] = "over_odds_raw"
            break

    # Under odds
    for c in ("under_odds", "under_line"):
        if c in df.columns:
            col_map[c] = "under_odds_raw"
            break

    if "L_close" not in col_map.values():
        logger.warning("nflverse CSV has no recognizable total line column. "
                       "Available columns: %s", list(df.columns))
        return pd.DataFrame()

    keep = ["game_id"] + list(col_map.keys())
    keep = [c for c in keep if c in df.columns]
    out = df[keep].rename(columns=col_map).copy()

    logger.info("Loaded nflverse odds: %d rows, L_close non-null: %d",
                len(out), out["L_close"].notna().sum())
    return out


def _try_odds_api(seasons: list[int]) -> pd.DataFrame:
    """Attempt to fetch odds from The Odds API.

    Requires ODDS_API_KEY to be set.  Returns empty DataFrame on failure.
    The historical endpoint needs a paid plan, so this is best-effort.
    """
    if not ODDS_API_KEY:
        logger.info("No ODDS_API_KEY configured — skipping The Odds API.")
        return pd.DataFrame()

    logger.info("Attempting to fetch odds from The Odds API...")
    # The Odds API historical endpoint structure (simplified)
    # In practice this requires event IDs and a paid plan.
    # We attempt a basic request and fall back gracefully.
    try:
        url = f"{ODDS_API_BASE}/v4/sports/{ODDS_API_SPORT}/odds"
        params = {
            "apiKey": ODDS_API_KEY,
            "regions": "us",
            "markets": "totals",
            "oddsFormat": "american",
        }
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        logger.info("Odds API returned %d events", len(data))
        # Parse would go here for live data; for historical we'd need
        # the historical endpoint which requires a paid plan.
        # Return empty for now — the nflverse fallback handles this.
        return pd.DataFrame()
    except Exception as exc:
        logger.warning("Odds API request failed: %s — falling back to nflverse", exc)
        return pd.DataFrame()


def _compute_odds_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Compute implied probabilities, vig-free probs, and overround.

    Operates in-place on columns: over_odds_raw, under_odds_raw → derived cols.
    """
    df = df.copy()

    # Initialize derived columns
    df["over_implied_prob"] = np.nan
    df["under_implied_prob"] = np.nan
    df["overround"] = np.nan
    df["p_over_vigfree"] = np.nan
    df["p_under_vigfree"] = np.nan
    df["over_odds"] = np.nan
    df["under_odds"] = np.nan

    # Copy raw odds to standard names
    if "over_odds_raw" in df.columns:
        df["over_odds"] = df["over_odds_raw"]
    if "under_odds_raw" in df.columns:
        df["under_odds"] = df["under_odds_raw"]

    # Compute derived values row by row for those with valid odds
    has_both = df["over_odds"].notna() & df["under_odds"].notna()
    for idx in df[has_both].index:
        o_over = df.at[idx, "over_odds"]
        o_under = df.at[idx, "under_odds"]

        # Skip if odds are zero (invalid)
        if o_over == 0 or o_under == 0:
            continue

        try:
            p_over = implied_probability(int(o_over))
            p_under = implied_probability(int(o_under))
            df.at[idx, "over_implied_prob"] = p_over
            df.at[idx, "under_implied_prob"] = p_under
            df.at[idx, "overround"] = overround(p_over, p_under)
            vf_over, vf_under = vig_free_probabilities(p_over, p_under)
            df.at[idx, "p_over_vigfree"] = vf_over
            df.at[idx, "p_under_vigfree"] = vf_under
        except (ValueError, AssertionError) as exc:
            logger.warning("Odds computation error at index %s (over=%s, under=%s): %s",
                           idx, o_over, o_under, exc)

    # Drop intermediate columns
    for col in ("over_odds_raw", "under_odds_raw"):
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    return df


def collect_odds(games_df: pd.DataFrame) -> pd.DataFrame:
    """Collect and compute odds data for all games.

    Parameters
    ----------
    games_df : pd.DataFrame
        Must contain: game_id, season

    Returns
    -------
    pd.DataFrame
        Columns: game_id, L_close, L_open, over_odds, under_odds,
                 over_implied_prob, under_implied_prob, overround,
                 p_over_vigfree, p_under_vigfree
    """
    seasons = sorted(games_df["season"].unique().tolist())

    # Try The Odds API first
    api_df = _try_odds_api(seasons)

    # Fall back to nflverse
    nfl_df = _load_nflverse_odds()

    # Merge: prefer API data if available, otherwise nflverse
    if not api_df.empty and "game_id" in api_df.columns:
        odds_df = api_df.copy()
        # Fill gaps from nflverse
        if not nfl_df.empty:
            missing = ~odds_df["game_id"].isin(nfl_df["game_id"])
            if missing.any():
                odds_df = pd.concat([odds_df, nfl_df[nfl_df["game_id"].isin(
                    games_df.loc[missing, "game_id"]
                )]], ignore_index=True)
    elif not nfl_df.empty:
        odds_df = nfl_df.copy()
    else:
        logger.warning("No odds data available from any source!")
        # Return a DataFrame with all NaN odds
        return pd.DataFrame({
            "game_id": games_df["game_id"],
            "L_close": np.nan, "L_open": np.nan,
            "over_odds": np.nan, "under_odds": np.nan,
            "over_implied_prob": np.nan, "under_implied_prob": np.nan,
            "overround": np.nan,
            "p_over_vigfree": np.nan, "p_under_vigfree": np.nan,
        })

    # Only keep games in our dataset
    odds_df = odds_df[odds_df["game_id"].isin(games_df["game_id"])].copy()

    # Ensure L_open exists
    if "L_open" not in odds_df.columns:
        odds_df["L_open"] = np.nan

    # Compute derived odds columns
    odds_df = _compute_odds_derived(odds_df)

    # ── INTEGRITY CHECKS ──────────────────────────────────────────────
    _run_integrity_checks(odds_df, len(games_df))

    # Ensure all game_ids are present (fill missing with NaN)
    all_ids = games_df[["game_id"]].copy()
    odds_df = all_ids.merge(odds_df, on="game_id", how="left")

    output_cols = [
        "game_id", "L_close", "L_open",
        "over_odds", "under_odds",
        "over_implied_prob", "under_implied_prob",
        "overround", "p_over_vigfree", "p_under_vigfree",
    ]
    for col in output_cols:
        if col not in odds_df.columns:
            odds_df[col] = np.nan

    return odds_df[output_cols]


def _run_integrity_checks(odds_df: pd.DataFrame, total_games: int) -> None:
    """Embedded integrity checks for odds data."""
    if "L_close" in odds_df.columns:
        valid = odds_df["L_close"].dropna()
        if len(valid) > 0:
            out_of_range = valid[(valid < 28) | (valid > 70)]
            if len(out_of_range) > 0:
                logger.warning(
                    "%d L_close values outside [28, 70]: min=%.1f, max=%.1f",
                    len(out_of_range), out_of_range.min(), out_of_range.max(),
                )

    if "overround" in odds_df.columns:
        valid_or = odds_df["overround"].dropna()
        if len(valid_or) > 0:
            bad = valid_or[(valid_or < 0) | (valid_or > 0.15)]
            if len(bad) > 0:
                logger.warning(
                    "%d overround values outside [0, 0.15]: min=%.4f, max=%.4f",
                    len(bad), bad.min(), bad.max(),
                )

    if "p_over_vigfree" in odds_df.columns and "p_under_vigfree" in odds_df.columns:
        both = odds_df[["p_over_vigfree", "p_under_vigfree"]].dropna()
        if len(both) > 0:
            sums = both["p_over_vigfree"] + both["p_under_vigfree"]
            bad_sum = sums[~sums.between(1.0 - 1e-10, 1.0 + 1e-10)]
            if len(bad_sum) > 0:
                logger.warning(
                    "%d vig-free probability pairs don't sum to 1.0",
                    len(bad_sum),
                )

    n_with_odds = odds_df["L_close"].notna().sum() if "L_close" in odds_df.columns else 0
    pct = 100.0 * n_with_odds / total_games if total_games > 0 else 0.0
    logger.info(
        "Odds data available for %d/%d games (%.1f%%).",
        n_with_odds, total_games, pct,
    )
