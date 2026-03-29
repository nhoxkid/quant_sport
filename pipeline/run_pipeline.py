"""
Pipeline orchestrator — runs all collection, merge, and validation steps.

Usage:
    python -m pipeline.run_pipeline --seasons 2022 2023 2024
    python pipeline/run_pipeline.py                           # default: 2018-2024
"""
import argparse
import logging
import sys
from pathlib import Path

# Allow running as script
_this = Path(__file__).resolve().parent
if str(_this.parent) not in sys.path:
    sys.path.insert(0, str(_this.parent))

from pipeline.config import DEFAULT_SEASONS
from pipeline.collect_games import collect_games
from pipeline.collect_weather import collect_weather
from pipeline.collect_odds import collect_odds
from pipeline.merge_and_clean import merge_and_clean
from pipeline.validate import main as validate_main

logger = logging.getLogger(__name__)


def run(seasons: list[int] | None = None) -> int:
    """Execute the full pipeline.  Returns 0 on success, 1 on validation failure."""
    if seasons is None:
        seasons = DEFAULT_SEASONS

    logger.info("=" * 60)
    logger.info("NFL Totals Data Pipeline — seasons %s", seasons)
    logger.info("=" * 60)

    # Step 1: Collect games
    logger.info("STEP 1/4: Collecting game data...")
    games_df = collect_games(seasons=seasons)
    logger.info("Games collected: %d rows", len(games_df))

    # Step 2: Collect weather
    logger.info("STEP 2/4: Collecting weather data...")
    weather_df = collect_weather(games_df)
    logger.info("Weather collected: %d rows", len(weather_df))

    # Step 3: Collect odds
    logger.info("STEP 3/4: Collecting odds data...")
    odds_df = collect_odds(games_df)
    logger.info("Odds collected: %d rows", len(odds_df))

    # Step 4: Merge and clean
    logger.info("STEP 4/4: Merging and cleaning...")
    final_df = merge_and_clean(games_df, weather_df, odds_df)
    logger.info("Final dataset: %d rows", len(final_df))

    # Step 5: Validate
    logger.info("Running validation...")
    exit_code = validate_main()

    return exit_code


def main():
    parser = argparse.ArgumentParser(description="NFL Totals Data Pipeline")
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=None,
        help=f"Seasons to collect (default: {DEFAULT_SEASONS[0]}-{DEFAULT_SEASONS[-1]})",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    exit_code = run(seasons=args.seasons)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
