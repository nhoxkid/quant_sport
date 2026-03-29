"""
Merge games, weather, and odds DataFrames into a single game-level dataset.

Computes derived variables:
  T_norm   — mean temperature for each (stadium, month) across all seasons
  T_prime  — temperature anomaly: T - T_norm
  E_W      — extreme wind flag: 1 if W >= 90th percentile for stadium-month
  E_T      — extreme temperature flag: 1 if |T_prime| >= threshold (default 8°C)

Cleaning rules (each logged with before/after row counts):
  1. Drop games with missing scores
  2. Drop preseason games
  3. Dome games: weather = NaN, E_W = 0, E_T = 0
  4. Flag (don't drop) outdoor games where weather retrieval failed
  5. Flag (don't drop) games where L_close is missing

Output: data/processed/nfl_totals_weather.csv
"""
import logging

import numpy as np
import pandas as pd

from pipeline.config import E_T_THRESHOLD, E_W_PERCENTILE, OUTPUT_CSV, EXPECTED_COLUMNS

logger = logging.getLogger(__name__)


def _log_step(label: str, before: int, after: int) -> None:
    """Log a cleaning step with row counts."""
    logger.info("CLEAN [%s]: %d → %d rows (Δ %d)", label, before, after, after - before)


def _compute_climate_normals(df: pd.DataFrame) -> pd.DataFrame:
    """Compute T_norm(stadium, month) = mean temperature per stadium-month.

    Only uses outdoor games with valid temperature data.
    Merges T_norm back onto the DataFrame and computes T_prime = T - T_norm.
    """
    df = df.copy()

    # Extract month from gameday
    df["_month"] = pd.to_datetime(df["gameday"]).dt.month

    # Compute normals from outdoor games only
    outdoor_valid = df[(df["dome_indicator"] == 0) & df["temperature"].notna()]
    if outdoor_valid.empty:
        logger.warning("No outdoor games with temperature — cannot compute T_norm")
        df["T_norm"] = np.nan
        df["T_prime"] = np.nan
        df.drop(columns=["_month"], inplace=True)
        return df

    normals = (
        outdoor_valid
        .groupby(["home_team", "_month"])["temperature"]
        .mean()
        .reset_index()
        .rename(columns={"temperature": "T_norm"})
    )
    logger.info("Computed climate normals for %d (stadium, month) pairs", len(normals))

    # Merge normals back
    df = df.merge(normals, on=["home_team", "_month"], how="left")
    df["T_prime"] = df["temperature"] - df["T_norm"]

    df.drop(columns=["_month"], inplace=True)
    return df


def _compute_extreme_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Compute E_W and E_T flags.

    E_W = 1 if wind_speed >= 90th percentile for that stadium-month (outdoor only)
    E_T = 1 if |T_prime| >= E_T_THRESHOLD (outdoor only)
    Dome games always get E_W = 0, E_T = 0.
    """
    df = df.copy()
    df["E_W"] = 0
    df["E_T"] = 0

    # Extract month
    df["_month"] = pd.to_datetime(df["gameday"]).dt.month

    # Compute 90th percentile wind per stadium-month from outdoor games
    outdoor_valid = df[(df["dome_indicator"] == 0) & df["wind_speed"].notna()]
    if not outdoor_valid.empty:
        wind_pcts = (
            outdoor_valid
            .groupby(["home_team", "_month"])["wind_speed"]
            .quantile(E_W_PERCENTILE / 100.0)
            .reset_index()
            .rename(columns={"wind_speed": "_wind_p90"})
        )
        df = df.merge(wind_pcts, on=["home_team", "_month"], how="left")

        # E_W flag: outdoor games with wind >= 90th percentile
        outdoor_mask = (df["dome_indicator"] == 0) & df["wind_speed"].notna() & df["_wind_p90"].notna()
        df.loc[outdoor_mask, "E_W"] = (
            (df.loc[outdoor_mask, "wind_speed"] >= df.loc[outdoor_mask, "_wind_p90"]).astype(int)
        )
        df.drop(columns=["_wind_p90"], inplace=True)

    # E_T flag: outdoor games with |T_prime| >= threshold
    outdoor_t_mask = (df["dome_indicator"] == 0) & df["T_prime"].notna()
    df.loc[outdoor_t_mask, "E_T"] = (
        (df.loc[outdoor_t_mask, "T_prime"].abs() >= E_T_THRESHOLD).astype(int)
    )

    # Ensure dome games are 0
    dome_mask = df["dome_indicator"] == 1
    df.loc[dome_mask, "E_W"] = 0
    df.loc[dome_mask, "E_T"] = 0

    df.drop(columns=["_month"], inplace=True)
    return df


def merge_and_clean(
    games_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    odds_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge all data sources and produce the final cleaned dataset.

    Parameters
    ----------
    games_df : pd.DataFrame
        From collect_games()
    weather_df : pd.DataFrame
        From collect_weather() — columns: game_id, temperature, wind_speed, precipitation
    odds_df : pd.DataFrame
        From collect_odds() — columns: game_id, L_close, L_open, over_odds, ...

    Returns
    -------
    pd.DataFrame
        Final merged, cleaned dataset.
    """
    # ── Pre-merge checks ──────────────────────────────────────────────
    logger.info("Sample game_ids from games_df:  %s", list(games_df["game_id"].head(3)))
    logger.info("Sample game_ids from weather_df: %s", list(weather_df["game_id"].head(3)))
    logger.info("Sample game_ids from odds_df:    %s", list(odds_df["game_id"].head(3)))

    n_games = len(games_df)
    logger.info("Pre-merge: games=%d, weather=%d, odds=%d", n_games, len(weather_df), len(odds_df))

    # Verify game_id types match
    games_id_dtype = games_df["game_id"].dtype
    weather_id_dtype = weather_df["game_id"].dtype
    odds_id_dtype = odds_df["game_id"].dtype
    logger.info("game_id dtypes: games=%s, weather=%s, odds=%s",
                games_id_dtype, weather_id_dtype, odds_id_dtype)

    # Coerce all to string to ensure matching
    games_df = games_df.copy()
    weather_df = weather_df.copy()
    odds_df = odds_df.copy()
    games_df["game_id"] = games_df["game_id"].astype(str)
    weather_df["game_id"] = weather_df["game_id"].astype(str)
    odds_df["game_id"] = odds_df["game_id"].astype(str)

    # ── Merge games + weather ─────────────────────────────────────────
    df = games_df.merge(weather_df, on="game_id", how="left")
    assert len(df) == n_games, (
        f"Row count changed after weather merge: {n_games} → {len(df)}. "
        "Check for duplicate game_ids in weather_df."
    )

    # ── Merge + odds ──────────────────────────────────────────────────
    df = df.merge(odds_df, on="game_id", how="left")
    assert len(df) == n_games, (
        f"Row count changed after odds merge: {n_games} → {len(df)}. "
        "Check for duplicate game_ids in odds_df."
    )

    logger.info("Post-merge: %d rows (expected %d)", len(df), n_games)

    # ── Cleaning step 1: drop missing scores ──────────────────────────
    before = len(df)
    df = df.dropna(subset=["home_score", "away_score"]).copy()
    _log_step("drop missing scores", before, len(df))

    # ── Cleaning step 2: drop preseason ───────────────────────────────
    before = len(df)
    if "game_type" in df.columns:
        df = df[df["game_type"].isin(["REG", "POST"])].copy()
    _log_step("drop preseason", before, len(df))

    # ── Cleaning step 3: dome games — weather = NaN, extremes = 0 ────
    dome_mask = df["dome_indicator"] == 1
    dome_count = dome_mask.sum()
    df.loc[dome_mask, ["temperature", "wind_speed", "precipitation"]] = np.nan
    logger.info("CLEAN [dome weather NaN]: %d dome games set to NaN weather", dome_count)

    # ── Compute climate normals and T_prime ───────────────────────────
    df = _compute_climate_normals(df)

    # ── Compute extreme flags ─────────────────────────────────────────
    df = _compute_extreme_flags(df)

    # ── Cleaning step 4: flag outdoor games with missing weather ──────
    outdoor_mask = df["dome_indicator"] == 0
    df["weather_missing"] = 0
    weather_cols = ["temperature", "wind_speed", "precipitation"]
    outdoor_no_wx = outdoor_mask & df[weather_cols].isna().all(axis=1)
    df.loc[outdoor_no_wx, "weather_missing"] = 1
    logger.info("CLEAN [weather_missing flag]: %d outdoor games missing weather", outdoor_no_wx.sum())

    # ── Cleaning step 5: flag games with missing odds ─────────────────
    df["odds_missing"] = 0
    if "L_close" in df.columns:
        df.loc[df["L_close"].isna(), "odds_missing"] = 1
    else:
        df["odds_missing"] = 1
    odds_missing_count = df["odds_missing"].sum()
    logger.info("CLEAN [odds_missing flag]: %d games missing L_close", odds_missing_count)

    # ── Ensure dome games have E_W=0 and E_T=0 (redundant safety) ────
    df.loc[dome_mask, "E_W"] = 0
    df.loc[dome_mask, "E_T"] = 0
    # Also set T_norm and T_prime to NaN for dome games
    df.loc[dome_mask, "T_norm"] = np.nan
    df.loc[dome_mask, "T_prime"] = np.nan

    # ── Ensure all expected columns exist ─────────────────────────────
    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
            logger.warning("Added missing column %r (filled with NaN)", col)

    # ── Cast types ────────────────────────────────────────────────────
    int_cols = ["dome_indicator", "E_W", "E_T", "weather_missing", "odds_missing"]
    for col in int_cols:
        df[col] = df[col].astype(int)

    # ── Final summary ─────────────────────────────────────────────────
    outdoor_count = (df["dome_indicator"] == 0).sum()
    dome_final = (df["dome_indicator"] == 1).sum()
    with_weather = outdoor_count - df["weather_missing"].sum()
    with_odds = len(df) - df["odds_missing"].sum()
    logger.info(
        "Final dataset: %d games. %d outdoor, %d dome. "
        "%d with weather data. %d with odds data.",
        len(df), outdoor_count, dome_final, with_weather, with_odds,
    )

    # ── Save output ───────────────────────────────────────────────────
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df[EXPECTED_COLUMNS].to_csv(OUTPUT_CSV, index=False)
    logger.info("Saved final CSV to %s", OUTPUT_CSV)

    return df[EXPECTED_COLUMNS]
