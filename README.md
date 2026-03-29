# NFL Totals ODE Model — Weather, Betting Lines & Market Efficiency

## What Is This Project?

This project models the NFL over/under (totals) betting market as a **2-D coupled ordinary differential equation (ODE) dynamical system** influenced by weather. It asks two questions:

1. **Does weather affect NFL scoring?** (Yes — wind and precipitation depress totals)
2. **Does the betting market correctly price weather into the closing line?** (Not fully — the market captures only ~32% of the wind effect, creating a systematic bias)

The project is structured in four phases, each building on the previous:

```
Phase 1: Data Pipeline     → Collect NFL games, weather, odds (2018-2024)
Phase 2: Statistical Model → Regression analysis of weather effects + market efficiency
Phase 3: ODE Model         → Dynamical systems theory of how the betting line evolves
Phase 4: Simulation        → Monte Carlo simulation of line paths under different conditions
```

---

## Quick Start

```bash
# Install dependencies
pip install pandas requests numpy pytz pytest statsmodels seaborn scipy

# Run everything (downloads data on first run, ~5 min; cached runs ~2 min)
python run.py

# Or run individual phases
python -m pipeline.run_pipeline          # Phase 1 only
python phase2_statistical_modeling.py    # Phase 2 only
python phase3_ode_model.py              # Phase 3 only
python phase4_simulation.py             # Phase 4 only
```

All outputs are saved to `data_file/`:
```
data_file/
  phase1/   — CSV dataset, validation report
  phase2/   — 10 regression & efficiency figures
  phase3/   — 9 ODE calibration & theory figures
  phase4/   — 9 Monte Carlo simulation figures
```

---

## Phase 1 — Data Pipeline

**What it does:** Collects 1,855 NFL games (2018-2024) with weather at kickoff and betting lines, merges them into a single analysis-ready CSV.

**Data sources:**
- **Game results:** nflverse (GitHub CSV, free, no API key)
- **Weather:** Open-Meteo Historical API (free, no key) — temperature (°C), wind speed (km/h), precipitation (mm) at kickoff for outdoor games
- **Betting odds:** nflverse schedules CSV — closing totals line (L_close), American odds

**Key design decisions:**
- Timezone handling: nflverse gametime is US Eastern, converted to UTC for weather API calls. Each stadium has a hardcoded IANA timezone for local-hour validation.
- Dome games get NaN weather (no weather effect indoors). 548 dome games, 1,265 outdoor with weather.
- All raw API responses are cached to `pipeline/data/raw/` so subsequent runs skip downloads.

**Output:** `nfl_totals_weather.csv` — 1,855 rows, 33 columns. Validated by 29 automated checks (schema, ranges, consistency, cross-variable correlations).

**Files:**
```
pipeline/
  config.py            — API URLs, file paths, thresholds
  stadium_coords.py    — All 32 NFL stadiums with lat/lon/timezone/roof
  collect_games.py     — Download and clean game data
  collect_weather.py   — Fetch weather from Open-Meteo (batched by stadium-season)
  collect_odds.py      — Extract betting lines from nflverse
  merge_and_clean.py   — Join all sources, compute derived variables
  validate.py          — 29-check data quality report
  run_pipeline.py      — Orchestrator (CLI: --seasons 2022 2023 2024)
  utils.py             — Implied probability, vig-free, overround formulas
  tests/               — 65 pytest tests
```

---

## Phase 2 — Statistical Modeling

**What it does:** Quantifies how weather affects scoring and tests whether the betting market prices it correctly.

### The Core Models

**Model 1b (baseline):** Simple OLS regression of total points on weather:
```
Y = β₀ + β₁·wind_speed + β₂·precipitation + β₃·T_prime + β₅·week + ε
```
Result: wind significantly depresses scoring (β₁ = -0.174 pts/km/h, p < 0.001).

**Model 2b (enhanced):** Adds team-strength controls and piecewise wind:
```
Y = β₀ + β₁·wind + β₁ₚ·max(0, wind-15) + β₂·precip + β₃·T' + β₅·week
    + β₆·home_pts_avg + β₇·away_pts_avg + ε
```
Finding: wind below 15 km/h barely matters (β₁ ≈ 0), but above 15 km/h each extra km/h costs ~0.33 points.

### Market Efficiency Test

The key insight uses a mathematical identity. Since Y = L + (Y - L) exactly:

```
Regress Y on X:    coefficients = β    (actual weather effect)
Regress L on X:    coefficients = γ    (how market adjusts for weather)
Regress Y-L on X:  coefficients = α    (what the market misses)
```

The decomposition **β = γ + α** holds exactly (proven algebraically and verified numerically). If α ≈ 0, the market is efficient. If α ≠ 0, the market misprices.

**Results (consistent-specification decomposition):**

| Variable | β (actual effect) | γ (market adjustment) | α (gap) | p-value | Interpretation |
|---|---|---|---|---|---|
| wind_speed | -0.174 | -0.056 | -0.118 | 0.016 | Market captures only 32% of wind effect |
| precipitation | -2.137 | -0.701 | -1.436 | 0.029 | Market captures only 33% of rain effect |
| T_prime | +0.167 | +0.024 | +0.143 | 0.070 | Market barely adjusts for temperature |

The market systematically sets totals too high in bad weather.

**Files:** `phase2_statistical_modeling.py` (single self-contained script)

**Figures produced:** 10 PNGs including scatter plots, LOWESS curves, coefficient comparisons, binned wind analysis, residual diagnostics.

---

## Phase 3 — ODE Model (Theory & Calibration)

**What it does:** Models the betting line's evolution as a damped dynamical system and tests whether extreme weather changes the market's dynamics.

### The 2-D ODE System

Two coupled equations describe how the betting line L(t) and the bookmaker's exposure q(t) evolve toward fair value μ:

```
dq/dt = k·(μ - L) - c·q          (5.2) Exposure dynamics
dL/dt = a·q + η·(μ - L)          (5.3) Line dynamics
```

**Parameter meanings:**
- **k** — How aggressively bettors exploit perceived mispricing
- **c** — How fast one-sided exposure dissipates
- **a** — How much the bookmaker moves the line per unit exposure
- **η** — How fast the bookmaker independently tracks fair value

**Equilibrium:** L* = μ, q* = 0 (line converges to fair value with balanced book). Proven stable for all positive parameters.

### Second-Order Reduction

Eliminating q(t) gives a damped harmonic oscillator equation:
```
L'' + (c + η)·L' + (cη + ak)·(L - μ) = 0
```
This has an exact analytic solution (no numerical solver needed):
- **Overdamped** ((c-η)² > 4ak): monotone exponential convergence
- **Underdamped** ((c-η)² < 4ak): damped oscillations around μ
- **Critical** ((c-η)² = 4ak): fastest convergence without oscillation

### Key Design Decision: Composite Parameters Only

The full system has 4 parameters (k, c, a, η), but since q(t) is never observed (bookmaker exposure is private), only two **composite parameters** are identifiable from line data:
- **γ = c + η** (effective damping)
- **ω² = cη + ak** (effective stiffness)

Synthetic validation proved individual parameters are unrecoverable (60-900% errors). The composite approach gives 30-55% errors — imperfect but real.

### Key Finding

Extreme weather games have **lower damping** (γ) than normal games:
- Normal: median γ = 0.35
- Extreme: median γ = 0.19 (p = 0.009)
- Interpretation: the market converges to fair value more slowly in extreme weather, because genuine uncertainty makes bookmakers hesitate.

**Files:**
```
src/
  ode_model.py      — ODE system, Jacobian, eigenvalues, analytic solution
  calibration.py    — 3-parameter composite fit (analytic closed-form, no solver)
  baselines.py      — Linear interpolation, exponential smoothing, random walk
  diagnostics.py    — Synthetic validation, residual tests
  simulation.py     — Line path simulation with noise
phase3_ode_model.py — Main script
```

**Figures produced:** 9 PNGs including ODE fits, phase portraits, parameter sensitivity, extreme-vs-normal boxplots.

---

## Phase 4 — Simulation Module

**What it does:** Runs Monte Carlo simulations of the full stochastic system to predict closing-line accuracy under different weather scenarios.

### The SDE Extension

Adding noise to the ODE gives a stochastic differential equation:
```
dq = [k(μ - L) - c·q]·dt + σ_q·dW₁     (additive noise, NOT multiplicative like GBM)
dL = [a·q + η(μ - L)]·dt + σ_L·dW₂
```

Discretised via Euler-Maruyama, vectorised over 1,000 paths simultaneously (mirroring the GBM Monte Carlo Studio architecture).

### Three Scenarios

| Scenario | μ | η | Key difference | Closing error |
|---|---|---|---|---|
| **Baseline** | 45.5 | 0.40 | Normal conditions | P(<0.5pt) = 100% |
| **Extreme Wind** | 42.5 | 0.15 | Wind known from open; η reduced (book tracks slowly under uncertainty) | P(<0.5pt) = 91% |
| **Late Shock** | 45.5→42.5 at t=-8h | 0.15 | Weather forecast arrives late; only 8 hours to adjust | P(<0.5pt) = 50% |

The late-shock scenario is the hardest: the line must drop 3 points in 8 hours under slow-tracking dynamics. Half the time, it doesn't make it — leaving the closing line systematically too high.

**Files:** `phase4_simulation.py` (single self-contained script)

**Figures produced:** 9 PNGs including cross-validation, single stochastic paths, fan charts, closing-error distributions (the key Figure 7), and trajectory comparisons.

---

## The Story in Three Sentences

1. Wind above 15 km/h significantly depresses NFL scoring (~0.33 points per km/h), but the betting market only adjusts ~32% of this effect.

2. The ODE model shows the market converges to fair value via damped oscillations, but extreme weather slows this convergence (lower damping), explaining why closing lines are less accurate in bad weather.

3. Monte Carlo simulation confirms: under normal conditions, the closing line is within 0.5 points of fair value 100% of the time, but when extreme weather arrives late, this drops to ~50%.

---

## Units Reference

| Variable | Unit | Source |
|---|---|---|
| temperature, T_prime | °C | Open-Meteo |
| wind_speed | km/h | Open-Meteo |
| precipitation | mm | Open-Meteo |
| L_close, total_points | points | nflverse |
| odds | American format | nflverse |
| time (ODE) | hours before kickoff | t=0 at kickoff |
| gamma (damping) | hr⁻¹ | composite: c + η |
| omega_sq (stiffness) | hr⁻² | composite: cη + ak |

---

## Known Limitations

1. **q(t) is unobservable.** Bookmaker exposure is private data. Only composite parameters (γ, ω²) are identifiable from public line data. Individual ODE parameters should not be over-interpreted.

2. **μ = L_close is partly circular.** Fitting a damped path to a known endpoint will always look decent. The test is intermediate snapshot accuracy, not endpoint accuracy.

3. **Weather timing mismatch.** Our weather is measured at kickoff; the betting line closes hours earlier. If forecasts changed between line close and kickoff, the efficiency test is biased toward finding "underreaction."

4. **No real line snapshots.** Phase 3 calibration uses synthetic data. The methodology is validated but not yet applied to real timestamped line movements.

5. **Low R².** Weather explains ~6% of scoring variance. Team quality, matchups, and randomness dominate. But even small systematic mispricings can be exploitable if consistent.

---

## Project Structure

```
C:\Code\ODE_coupled\
  run.py                         — One-click runner for all 4 phases
  phase1.md / phase2.md / ...    — Specification documents
  requirements.txt               — Python dependencies

  pipeline/                      — Phase 1: Data collection & cleaning
    config.py, collect_*.py, merge_and_clean.py, validate.py, ...
    data/raw/                    — Cached API responses
    data/processed/              — Final CSV + validation report
    tests/                       — 65 pytest tests

  src/                           — Phase 3: ODE model library
    ode_model.py                 — Core ODE math
    calibration.py               — Composite parameter fitting
    baselines.py                 — Baseline comparison models
    diagnostics.py               — Synthetic validation & residual tests
    simulation.py                — Line path simulation

  phase2_statistical_modeling.py — Phase 2: Regressions & efficiency
  phase3_ode_model.py            — Phase 3: ODE calibration runner
  phase4_simulation.py           — Phase 4: SDE Monte Carlo runner

  data_file/                     — ALL OUTPUT FILES
    phase1/                      — CSV, validation report
    phase2/                      — 10 regression figures
    phase3/                      — 9 ODE model figures
    phase4/                      — 9 simulation figures
```

---

## Dependencies

```
pandas >= 1.5.0
requests >= 2.28.0
numpy >= 1.23.0
scipy >= 1.9.0
statsmodels >= 0.13.0
seaborn >= 0.12.0
matplotlib >= 3.6.0
pytz >= 2022.1
pytest >= 7.0.0
```

Install: `pip install -r requirements.txt`

---

## Test Suite

```bash
python -m pytest pipeline/tests/ -v    # 65 tests covering data pipeline
```

Covers: implied probability math, stadium coordinates, game collection, weather API mocking, merge/join logic, validation checks.
