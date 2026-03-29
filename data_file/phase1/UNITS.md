# Units Documentation — nfl_totals_weather.csv

| Column        | Unit / Format                                    |
|---------------|--------------------------------------------------|
| temperature   | °C (Celsius) — from Open-Meteo, NOT converted    |
| wind_speed    | km/h — from Open-Meteo, NOT converted             |
| precipitation | mm (millimeters per hour) — from Open-Meteo       |
| L_close       | Total points line (half-points possible, e.g. 45.5)|
| L_open        | Total points line (half-points possible)           |
| over_odds     | American format (e.g. -110, +150)                  |
| under_odds    | American format (e.g. -110, +150)                  |
| T_norm        | °C — mean temperature for (stadium, month) pair    |
| T_prime       | °C — temperature anomaly (T - T_norm)              |
| E_T threshold | 8°C (configurable in config.py as E_T_THRESHOLD)  |
| E_W percentile| 90th (configurable in config.py as E_W_PERCENTILE) |
| kickoff_utc   | ISO 8601 UTC datetime                              |
| gameday       | YYYY-MM-DD date string                             |
| gametime      | HH:MM (US Eastern time as reported by nflverse)    |

## Coordinate Reference
- Stadium latitudes: decimal degrees (WGS84)
- Stadium longitudes: decimal degrees (WGS84, negative = West)

## Timezone Assumption
nflverse `gametime` values are in US Eastern time (America/New_York).
All kickoff times are converted to UTC for weather API queries.
