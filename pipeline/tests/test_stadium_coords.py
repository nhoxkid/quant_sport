"""Tests for pipeline.stadium_coords."""
import math
from zoneinfo import ZoneInfo

from pipeline.stadium_coords import (
    STADIUM_COORDS,
    CURRENT_TEAMS,
    get_stadium_info,
    validate_stadium_data,
)


class TestStadiumData:
    def test_all_32_current_teams_present(self):
        assert len(CURRENT_TEAMS) == 32
        missing = CURRENT_TEAMS - set(STADIUM_COORDS.keys())
        assert not missing, f"Missing: {missing}"

    def test_latitude_range(self):
        for team, info in STADIUM_COORDS.items():
            assert 25.0 <= info["lat"] <= 49.0, f"{team}: lat {info['lat']}"

    def test_longitude_range(self):
        for team, info in STADIUM_COORDS.items():
            assert -125.0 <= info["lon"] <= -70.0, f"{team}: lon {info['lon']}"

    def test_timezone_strings_valid(self):
        for team, info in STADIUM_COORDS.items():
            tz = ZoneInfo(info["timezone"])
            assert tz is not None, f"{team}: invalid timezone"

    def test_lambeau_field_coordinates(self):
        gb = STADIUM_COORDS["GB"]
        # Lambeau Field is roughly (44.50, -88.06)
        assert math.isclose(gb["lat"], 44.50, abs_tol=0.1)
        assert math.isclose(gb["lon"], -88.06, abs_tol=0.1)

    def test_get_stadium_info_raiders_relocation(self):
        # Pre-2020: should get Oakland
        oak = get_stadium_info("OAK", season=2019)
        assert oak["stadium_name"] == "Oakland-Alameda County Coliseum"

        # Post-2020: OAK query should redirect to LV
        lv = get_stadium_info("OAK", season=2020)
        assert lv["stadium_name"] == "Allegiant Stadium"

        # LV query for 2019 should redirect to OAK
        oak2 = get_stadium_info("LV", season=2019)
        assert oak2["stadium_name"] == "Oakland-Alameda County Coliseum"

    def test_validate_passes(self):
        # Should not raise
        validate_stadium_data()

    def test_all_entries_have_required_fields(self):
        required = {"lat", "lon", "timezone", "roof_type", "stadium_name"}
        for team, info in STADIUM_COORDS.items():
            missing = required - set(info.keys())
            assert not missing, f"{team} missing fields: {missing}"
