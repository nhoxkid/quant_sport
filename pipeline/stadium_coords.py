"""
Hardcoded NFL stadium coordinates, timezones, and roof types.

Covers all 32 current teams for the 2018-2024 window.
Accounts for relocations:
  - Raiders: Oakland (2018-2019) -> Las Vegas (2020+)
  - Chargers: already in LA by 2017, so LA for entire window
  - Rams: already in LA by 2016, so LA for entire window

Each entry: team_abbr -> {lat, lon, timezone, roof_type, stadium_name}

Timezone strings are IANA identifiers usable by zoneinfo / pytz.
"""
import logging
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

# fmt: off
STADIUM_COORDS: dict[str, dict] = {
    "ARI": {"lat": 33.5276, "lon": -112.2626, "timezone": "America/Phoenix",      "roof_type": "dome",        "stadium_name": "State Farm Stadium"},
    "ATL": {"lat": 33.7554, "lon":  -84.4010, "timezone": "America/New_York",     "roof_type": "dome",        "stadium_name": "Mercedes-Benz Stadium"},
    "BAL": {"lat": 39.2780, "lon":  -76.6227, "timezone": "America/New_York",     "roof_type": "outdoors",    "stadium_name": "M&T Bank Stadium"},
    "BUF": {"lat": 42.7738, "lon":  -78.7870, "timezone": "America/New_York",     "roof_type": "outdoors",    "stadium_name": "Highmark Stadium"},
    "CAR": {"lat": 35.2258, "lon":  -80.8528, "timezone": "America/New_York",     "roof_type": "outdoors",    "stadium_name": "Bank of America Stadium"},
    "CHI": {"lat": 41.8623, "lon":  -87.6167, "timezone": "America/Chicago",      "roof_type": "outdoors",    "stadium_name": "Soldier Field"},
    "CIN": {"lat": 39.0955, "lon":  -84.5161, "timezone": "America/New_York",     "roof_type": "outdoors",    "stadium_name": "Paycor Stadium"},
    "CLE": {"lat": 41.5061, "lon":  -81.6995, "timezone": "America/New_York",     "roof_type": "outdoors",    "stadium_name": "Cleveland Browns Stadium"},
    "DAL": {"lat": 32.7473, "lon":  -97.0945, "timezone": "America/Chicago",      "roof_type": "dome",        "stadium_name": "AT&T Stadium"},
    "DEN": {"lat": 39.7439, "lon": -105.0201, "timezone": "America/Denver",       "roof_type": "outdoors",    "stadium_name": "Empower Field at Mile High"},
    "DET": {"lat": 42.3400, "lon":  -83.0456, "timezone": "America/Detroit",      "roof_type": "dome",        "stadium_name": "Ford Field"},
    "GB":  {"lat": 44.5013, "lon":  -88.0622, "timezone": "America/Chicago",      "roof_type": "outdoors",    "stadium_name": "Lambeau Field"},
    "HOU": {"lat": 29.6847, "lon":  -95.4107, "timezone": "America/Chicago",      "roof_type": "dome",        "stadium_name": "NRG Stadium"},
    "IND": {"lat": 39.7601, "lon":  -86.1639, "timezone": "America/Indiana/Indianapolis", "roof_type": "dome", "stadium_name": "Lucas Oil Stadium"},
    "JAX": {"lat": 30.3239, "lon":  -81.6373, "timezone": "America/New_York",     "roof_type": "outdoors",    "stadium_name": "EverBank Stadium"},
    "KC":  {"lat": 39.0489, "lon":  -94.4839, "timezone": "America/Chicago",      "roof_type": "outdoors",    "stadium_name": "GEHA Field at Arrowhead Stadium"},
    "LV":  {"lat": 36.0908, "lon": -115.1833, "timezone": "America/Los_Angeles",  "roof_type": "dome",        "stadium_name": "Allegiant Stadium"},
    "LAC": {"lat": 33.9535, "lon": -118.3390, "timezone": "America/Los_Angeles",  "roof_type": "dome",        "stadium_name": "SoFi Stadium"},
    "LAR": {"lat": 33.9535, "lon": -118.3390, "timezone": "America/Los_Angeles",  "roof_type": "dome",        "stadium_name": "SoFi Stadium"},
    "MIA": {"lat": 25.9580, "lon":  -80.2389, "timezone": "America/New_York",     "roof_type": "outdoors",    "stadium_name": "Hard Rock Stadium"},
    "MIN": {"lat": 44.9736, "lon":  -93.2575, "timezone": "America/Chicago",      "roof_type": "dome",        "stadium_name": "U.S. Bank Stadium"},
    "NE":  {"lat": 42.0909, "lon":  -71.2643, "timezone": "America/New_York",     "roof_type": "outdoors",    "stadium_name": "Gillette Stadium"},
    "NO":  {"lat": 29.9511, "lon":  -90.0812, "timezone": "America/Chicago",      "roof_type": "dome",        "stadium_name": "Caesars Superdome"},
    "NYG": {"lat": 40.8128, "lon":  -74.0742, "timezone": "America/New_York",     "roof_type": "outdoors",    "stadium_name": "MetLife Stadium"},
    "NYJ": {"lat": 40.8128, "lon":  -74.0742, "timezone": "America/New_York",     "roof_type": "outdoors",    "stadium_name": "MetLife Stadium"},
    "OAK": {"lat": 37.7516, "lon": -122.2005, "timezone": "America/Los_Angeles",  "roof_type": "outdoors",    "stadium_name": "Oakland-Alameda County Coliseum"},
    "PHI": {"lat": 39.9008, "lon":  -75.1675, "timezone": "America/New_York",     "roof_type": "outdoors",    "stadium_name": "Lincoln Financial Field"},
    "PIT": {"lat": 40.4468, "lon":  -80.0158, "timezone": "America/New_York",     "roof_type": "outdoors",    "stadium_name": "Acrisure Stadium"},
    "SEA": {"lat": 47.5952, "lon": -122.3316, "timezone": "America/Los_Angeles",  "roof_type": "outdoors",    "stadium_name": "Lumen Field"},
    "SF":  {"lat": 37.4033, "lon": -121.9694, "timezone": "America/Los_Angeles",  "roof_type": "outdoors",    "stadium_name": "Levi's Stadium"},
    "TB":  {"lat": 27.9759, "lon":  -82.5033, "timezone": "America/New_York",     "roof_type": "outdoors",    "stadium_name": "Raymond James Stadium"},
    "TEN": {"lat": 36.1665, "lon":  -86.7713, "timezone": "America/Chicago",      "roof_type": "outdoors",    "stadium_name": "Nissan Stadium"},
    "WAS": {"lat": 38.9076, "lon":  -76.8645, "timezone": "America/New_York",     "roof_type": "outdoors",    "stadium_name": "Northwest Stadium"},
}
# fmt: on

# The 32 *current* NFL team abbreviations (2024 season).
# OAK is kept for historical games (2018-2019); LV covers 2020+.
CURRENT_TEAMS = {
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE",
    "DAL", "DEN", "DET", "GB",  "HOU", "IND", "JAX", "KC",
    "LAC", "LAR", "LV",  "MIA", "MIN", "NE",  "NO",  "NYG",
    "NYJ", "PHI", "PIT", "SEA", "SF",  "TB",  "TEN", "WAS",
}


def get_stadium_info(team: str, season: int | None = None) -> dict:
    """Return stadium info for a team, handling relocations.

    For the Raiders: use OAK for season <= 2019, LV for >= 2020.
    """
    if team == "OAK" and season is not None and season >= 2020:
        team = "LV"
    elif team == "LV" and season is not None and season <= 2019:
        team = "OAK"

    if team not in STADIUM_COORDS:
        raise KeyError(f"Unknown team abbreviation: {team!r}")
    return STADIUM_COORDS[team]


def validate_stadium_data() -> None:
    """Run integrity checks on stadium coordinates."""
    # All 32 current teams present
    missing = CURRENT_TEAMS - set(STADIUM_COORDS.keys())
    assert not missing, f"Missing teams in STADIUM_COORDS: {missing}"

    for team, info in STADIUM_COORDS.items():
        lat, lon = info["lat"], info["lon"]
        tz_str = info["timezone"]

        assert 25.0 <= lat <= 49.0, (
            f"{team}: latitude {lat} outside continental US range [25, 49]"
        )
        assert -125.0 <= lon <= -70.0, (
            f"{team}: longitude {lon} outside continental US range [-125, -70]"
        )
        # Validate timezone string is loadable
        try:
            ZoneInfo(tz_str)
        except Exception as exc:
            raise AssertionError(
                f"{team}: invalid timezone string {tz_str!r}: {exc}"
            ) from exc

    logger.info("Stadium data validated: %d entries, all checks passed.", len(STADIUM_COORDS))


# Run validation on import so bad data is caught immediately.
validate_stadium_data()
