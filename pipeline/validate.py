"""
Standalone validation module for the NFL totals-weather dataset.

Reads the final CSV and produces a comprehensive data quality report.
Runnable as:  python -m pipeline.validate
          or: python pipeline/validate.py

The report prints to console AND saves to data/processed/validation_report.txt.
"""
import sys
import logging
from datetime import datetime
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running as script or as module
try:
    from pipeline.config import OUTPUT_CSV, VALIDATION_REPORT, EXPECTED_COLUMNS
except ImportError:
    # Running as standalone script
    _this = Path(__file__).resolve().parent
    sys.path.insert(0, str(_this.parent))
    from pipeline.config import OUTPUT_CSV, VALIDATION_REPORT, EXPECTED_COLUMNS

logger = logging.getLogger(__name__)

# ── Report output helper ──────────────────────────────────────────────

class ReportWriter:
    """Accumulates report text and tracks pass/fail counts."""

    def __init__(self):
        self.buf = StringIO()
        self.passed = 0
        self.failed = 0
        self.warnings = 0

    def line(self, text: str = "") -> None:
        self.buf.write(text + "\n")

    def check_pass(self, label: str) -> None:
        self.buf.write(f"  {label}  \u2705\n")
        self.passed += 1

    def check_fail(self, label: str) -> None:
        self.buf.write(f"  {label}  \u274c\n")
        self.failed += 1

    def check_warn(self, label: str) -> None:
        self.buf.write(f"  {label}  \u26a0\ufe0f\n")
        self.warnings += 1

    def check(self, condition: bool, label: str) -> bool:
        if condition:
            self.check_pass(label)
        else:
            self.check_fail(label)
        return condition

    def text(self) -> str:
        return self.buf.getvalue()


def validate(csv_path: Path | str | None = None, df: pd.DataFrame | None = None) -> tuple:
    """Run all validation checks and return (report_text, passed, failed).

    Provide either csv_path or df.  If both are None, uses OUTPUT_CSV.
    """
    if df is None:
        csv_path = Path(csv_path) if csv_path else OUTPUT_CSV
        if not csv_path.exists():
            msg = f"Output CSV not found: {csv_path}"
            logger.error(msg)
            return msg, 0, 1
        df = pd.read_csv(csv_path, low_memory=False)

    rpt = ReportWriter()
    rpt.line("=" * 52)
    rpt.line("DATA QUALITY REPORT — nfl_totals_weather.csv")
    rpt.line(f"Generated: {datetime.now().isoformat(sep=' ', timespec='seconds')}")
    rpt.line("=" * 52)
    rpt.line()

    # ── SCHEMA CHECKS ─────────────────────────────────────────────────
    rpt.line("SCHEMA:")
    present = set(df.columns)
    expected = set(EXPECTED_COLUMNS)
    missing_cols = expected - present
    extra_cols = present - expected
    rpt.check(not missing_cols, f"All {len(expected)} expected columns present"
              + (f" (missing: {missing_cols})" if missing_cols else ""))
    if extra_cols:
        rpt.check_warn(f"Unexpected columns: {extra_cols}")
    else:
        rpt.check_pass("No unexpected columns")
    rpt.line()

    # ── DTYPE CHECKS ──────────────────────────────────────────────────
    rpt.line("DTYPES:")
    dtype_ok = True
    # game_id should be string-like (object)
    if "game_id" in df.columns and df["game_id"].dtype != object:
        dtype_ok = False
    # season should be numeric
    if "season" in df.columns and not pd.api.types.is_numeric_dtype(df["season"]):
        dtype_ok = False
    rpt.check(dtype_ok, "Key dtypes correct (game_id=str, season=numeric)")
    rpt.line()

    # ── COMPLETENESS CHECKS ───────────────────────────────────────────
    rpt.line("COMPLETENESS:")
    n = len(df)

    def completeness(col: str, subset_label: str = "", subset_mask=None):
        if col not in df.columns:
            rpt.check_fail(f"{col}: column missing")
            return
        if subset_mask is not None:
            series = df.loc[subset_mask, col]
            total = subset_mask.sum()
        else:
            series = df[col]
            total = n
        nonnull = series.notna().sum()
        pct = 100.0 * nonnull / total if total > 0 else 0.0
        label = f"{col}: {pct:.1f}% non-null"
        if subset_label:
            label += f" ({subset_label}: {100.0 * nonnull / total:.1f}%)"
        if pct < 95:
            rpt.check_warn(label)
        else:
            rpt.check_pass(label)

    completeness("total_points")
    outdoor_mask = df["dome_indicator"] == 0 if "dome_indicator" in df.columns else pd.Series([False] * n)
    completeness("temperature", "outdoor only", outdoor_mask)
    completeness("wind_speed", "outdoor only", outdoor_mask)
    completeness("precipitation", "outdoor only", outdoor_mask)
    completeness("L_close")
    rpt.line()

    # ── RANGE CHECKS ──────────────────────────────────────────────────
    rpt.line("RANGE CHECKS:")

    def range_check(col: str, lo, hi, subset_mask=None, subset_label=""):
        if col not in df.columns:
            rpt.check_fail(f"{col}: column missing")
            return
        s = df.loc[subset_mask, col].dropna() if subset_mask is not None else df[col].dropna()
        if len(s) == 0:
            rpt.check_warn(f"{col}: no data to check")
            return
        vmin, vmax = s.min(), s.max()
        ok = vmin >= lo and vmax <= hi
        label = f"{col}: min={vmin}, max={vmax} [{lo}, {hi}]"
        if subset_label:
            label += f" ({subset_label})"
        rpt.check(ok, label)

    range_check("total_points", 0, 120)
    range_check("L_close", 28, 70)
    range_check("temperature", -40, 45, outdoor_mask, "outdoor")
    range_check("wind_speed", 0, 120, outdoor_mask, "outdoor")
    range_check("precipitation", 0, 100, outdoor_mask, "outdoor")

    # Binary flags
    for col in ("dome_indicator", "E_W", "E_T"):
        if col in df.columns:
            vals = set(df[col].dropna().unique())
            rpt.check(vals <= {0, 1}, f"{col} in {{0, 1}} (found {vals})")

    # Probabilities
    if "p_over_vigfree" in df.columns:
        valid_p = df["p_over_vigfree"].dropna()
        if len(valid_p) > 0:
            rpt.check(valid_p.min() > 0 and valid_p.max() < 1,
                       f"p_over_vigfree in (0,1): min={valid_p.min():.4f}, max={valid_p.max():.4f}")

    if "overround" in df.columns:
        valid_or = df["overround"].dropna()
        if len(valid_or) > 0:
            rpt.check(valid_or.min() >= 0 and valid_or.max() <= 0.15,
                       f"overround in [0, 0.15]: min={valid_or.min():.4f}, max={valid_or.max():.4f}")
    rpt.line()

    # ── CONSISTENCY CHECKS ────────────────────────────────────────────
    rpt.line("CONSISTENCY:")

    # total_points == home + away
    if all(c in df.columns for c in ("total_points", "home_score", "away_score")):
        recomp = df["home_score"] + df["away_score"]
        match_mask = df["total_points"] == recomp
        rpt.check(match_mask.all(),
                   f"total_points == home + away: {'all' if match_mask.all() else match_mask.sum()}/{n} rows")

    # Dome games have NaN weather
    if "dome_indicator" in df.columns and "temperature" in df.columns:
        dome = df[df["dome_indicator"] == 1]
        dome_wx_nan = dome["temperature"].isna().all() if len(dome) > 0 else True
        rpt.check(dome_wx_nan, "Dome games have NaN weather")

    # Outdoor games mostly have weather
    if outdoor_mask.any() and "temperature" in df.columns:
        outdoor_has_wx = df.loc[outdoor_mask, "temperature"].notna().sum()
        outdoor_total = outdoor_mask.sum()
        pct = 100.0 * outdoor_has_wx / outdoor_total if outdoor_total > 0 else 0
        rpt.check(pct > 80, f"Outdoor games with weather: {outdoor_has_wx}/{outdoor_total} ({pct:.1f}%)")

    # E_W and E_T == 0 for dome games
    if "dome_indicator" in df.columns:
        dome = df[df["dome_indicator"] == 1]
        for flag in ("E_W", "E_T"):
            if flag in df.columns and len(dome) > 0:
                rpt.check((dome[flag] == 0).all(), f"{flag} == 0 for all dome games")

    # No duplicate game_ids
    if "game_id" in df.columns:
        dupes = df["game_id"].duplicated().sum()
        rpt.check(dupes == 0, f"No duplicate game_ids (found {dupes} dupes)")

    # Games per season
    if "season" in df.columns:
        season_counts = df["season"].value_counts().sort_index()
        all_ok = True
        for s, cnt in season_counts.items():
            if cnt < 200:
                rpt.check_fail(f"Season {s}: {cnt} games (< 200 — possible data loss)")
                all_ok = False
        if all_ok:
            rpt.check_pass(f"Games per season >= 200 for all {len(season_counts)} seasons")
    rpt.line()

    # ── DISTRIBUTION SUMMARY ──────────────────────────────────────────
    rpt.line("DISTRIBUTION SUMMARY:")
    for col in ("total_points", "L_close", "temperature", "wind_speed"):
        if col in df.columns:
            s = df[col].dropna()
            if len(s) > 0:
                rpt.line(f"  {col}: mean={s.mean():.1f}, std={s.std():.1f}, "
                         f"min={s.min():.1f}, max={s.max():.1f}, median={s.median():.1f}")
    rpt.line()

    if "season" in df.columns:
        rpt.line("  Season counts:")
        for s, cnt in df["season"].value_counts().sort_index().items():
            rpt.line(f"    {int(s)}: {cnt}")
    rpt.line()

    for col in ("dome_indicator", "E_W", "E_T"):
        if col in df.columns:
            rpt.line(f"  {col} value_counts:")
            for val, cnt in df[col].value_counts().sort_index().items():
                rpt.line(f"    {int(val)}: {cnt}")
    rpt.line()

    # ── CROSS-VARIABLE CHECKS ─────────────────────────────────────────
    rpt.line("CROSS-CHECKS:")

    # Correlation between total_points and L_close
    if "total_points" in df.columns and "L_close" in df.columns:
        both = df[["total_points", "L_close"]].dropna()
        if len(both) > 10:
            corr = both["total_points"].corr(both["L_close"])
            rpt.check(corr > 0.3, f"corr(total_points, L_close) = {corr:.3f} (> 0.3)")

    # Mean temperature: January outdoor < September outdoor
    if "temperature" in df.columns and "gameday" in df.columns and "dome_indicator" in df.columns:
        df_temp = df.copy()
        df_temp["_month"] = pd.to_datetime(df_temp["gameday"], errors="coerce").dt.month
        jan_outdoor = df_temp[(df_temp["_month"] == 1) & (df_temp["dome_indicator"] == 0)]["temperature"].dropna()
        sep_outdoor = df_temp[(df_temp["_month"] == 9) & (df_temp["dome_indicator"] == 0)]["temperature"].dropna()
        if len(jan_outdoor) > 0 and len(sep_outdoor) > 0:
            jan_mean = jan_outdoor.mean()
            sep_mean = sep_outdoor.mean()
            rpt.check(jan_mean < sep_mean,
                       f"Mean Jan outdoor temp ({jan_mean:.1f}°C) < Sep ({sep_mean:.1f}°C)")
        else:
            rpt.check_warn("Insufficient data for Jan/Sep temperature comparison")

    # Mean total_points sanity
    if "total_points" in df.columns:
        mean_tp = df["total_points"].mean()
        rpt.check(43 <= mean_tp <= 50,
                   f"Mean total_points = {mean_tp:.1f} (expected ~43-50)")

    # Dome mean temperature should be NaN
    if "dome_indicator" in df.columns and "temperature" in df.columns:
        dome_temp = df[df["dome_indicator"] == 1]["temperature"]
        rpt.check(dome_temp.isna().all() if len(dome_temp) > 0 else True,
                   "Mean temperature for dome games is NaN")

    rpt.line()

    # ── OVERALL SUMMARY ───────────────────────────────────────────────
    total_checks = rpt.passed + rpt.failed
    status = "\u2705" if rpt.failed == 0 else "\u274c"
    rpt.line(f"OVERALL: {status} {rpt.passed}/{total_checks} checks passed"
             f" ({rpt.warnings} warnings)")
    rpt.line("=" * 52)

    report_text = rpt.text()
    return report_text, rpt.passed, rpt.failed


def main(csv_path: str | None = None) -> int:
    """Run validation and print/save the report.  Returns 0 on success, 1 on failure."""
    report_text, passed, failed = validate(csv_path)
    print(report_text)

    # Save report
    try:
        VALIDATION_REPORT.parent.mkdir(parents=True, exist_ok=True)
        VALIDATION_REPORT.write_text(report_text, encoding="utf-8")
        print(f"Report saved to {VALIDATION_REPORT}")
    except Exception as exc:
        print(f"Warning: could not save report file: {exc}")

    return 1 if failed > 0 else 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    path = sys.argv[1] if len(sys.argv) > 1 else None
    sys.exit(main(path))
