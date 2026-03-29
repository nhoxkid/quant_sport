# NFL Totals ODE Model

Modeling the NFL over/under betting market as a coupled differential equation system. Turns out Vegas doesn't price wind correctly — the market only captures about 32% of the actual wind effect on scoring.

---

## what this is

I wanted to know two things:
1. Does weather actually affect NFL scoring in a measurable way?
2. If so, does the betting market account for it properly?

Short answers: yes, and not really. Wind above 15 km/h costs about 0.33 points per km/h, and the closing line barely adjusts for it. I built a full pipeline to prove this — data collection, regression analysis, ODE modeling, and Monte Carlo simulation.

Four phases, each builds on the last:

```
Phase 1: Data Pipeline     → 1,855 NFL games (2018-2024) with weather and odds
Phase 2: Statistical Model → regression analysis + market efficiency decomposition
Phase 3: ODE Model         → betting line dynamics as a damped harmonic oscillator
Phase 4: Simulation        → stochastic Monte Carlo across 3 weather scenarios
```

## how to run

```bash
pip install pandas requests numpy pytz pytest statsmodels seaborn scipy

python run.py    # runs all 4 phases, first run ~5min (downloads data), cached ~2min
```

Or individually:
```bash
python -m pipeline.run_pipeline          # Phase 1
python phase2_statistical_modeling.py    # Phase 2
python phase3_ode_model.py              # Phase 3
python phase4_simulation.py             # Phase 4
```

Everything saves to `data_file/`:
```
data_file/
  phase1/   — CSV dataset + validation report
  phase2/   — 10 regression/efficiency figures
  phase3/   — 9 ODE calibration figures
  phase4/   — 9 Monte Carlo figures
```

---

## Phase 1 — Data Pipeline

Scrapes NFL game results from nflverse, pulls hourly weather from Open-Meteo for every outdoor stadium, grabs closing totals lines. Merges it all into one CSV.

The timezone handling was painful — NFL posts gametime in Eastern, the weather API wants UTC, and stadiums span 4 time zones. All 32 stadiums are hardcoded with lat/lon/timezone/roof type. 29 automated validation checks run at the end because I got burned by bad joins early on. 65 unit tests.

1,855 games total. 548 dome (NaN weather), 1,265 outdoor with real weather data. All API responses cached so you don't re-download every run.

## Phase 2 — Statistical Modeling

The interesting part. Baseline regression shows wind depresses scoring at -0.174 pts/km/h (p < 0.001). Piecewise model shows it's nonlinear — below 15 km/h basically nothing happens, above 15 each km/h costs ~0.33 points.

The market efficiency test uses a neat algebraic identity: since Y = L + (Y - L), regressing each piece on the same weather variables gives a decomposition where β = γ + α exactly (verified numerically to 10 decimal places). β is what weather actually does, γ is what the market thinks it does, α is the gap.

| Variable | Actual effect (β) | Market adjustment (γ) | Gap (α) | p-value |
|---|---|---|---|---|
| wind_speed | -0.174 | -0.056 | -0.118 | 0.016 |
| precipitation | -2.137 | -0.701 | -1.436 | 0.029 |
| temp anomaly | +0.167 | +0.024 | +0.143 | 0.070 |

Market captures about a third of the weather effect. Totals are systematically too high in bad weather.

## Phase 3 — ODE Model

This is where it gets fun. The betting line is modeled as a 2-D coupled system:

```
dq/dt = k·(μ - L) - c·q          (exposure dynamics)
dL/dt = a·q + η·(μ - L)          (line dynamics)
```

Eliminate q and you get a damped harmonic oscillator: `L'' + (c+η)·L' + (cη+ak)·(L-μ) = 0`. The line overshoots fair value, generates reverse flow, oscillates, decays. Analytic solution matches the numerical solver to 10⁻¹¹.

Tried fitting all 4 parameters individually — synthetic validation showed 60-900% errors because q(t) is unobservable (bookmaker exposure is private). Switched to fitting only the composite parameters (γ = c+η, ω² = cη+ak) which are the only things that actually appear in the L(t) equation. Much more reasonable.

Main finding: extreme weather games have significantly lower damping (γ: 0.35 normal vs 0.19 extreme, p = 0.009). The market converges to fair value more slowly when weather introduces uncertainty.

## Phase 4 — Simulation

Added stochastic noise to the ODE (additive, not multiplicative — this isn't GBM) and ran 1,000-path Monte Carlo simulations. Architecture adapted from my [GBM repo](https://github.com/nhoxkid/GBM) — Euler-Maruyama stepping, vectorized over all paths.

Three scenarios:

| Scenario | P(closing line within 0.5 pts) |
|---|---|
| Baseline (normal weather) | 100% |
| Extreme wind (known from open, slow tracking) | 91% |
| Late shock (forecast arrives 8h before kickoff) | 50% |

The late shock case is the mechanism behind the Phase 2 efficiency gap. The line needs to drop 3 points in 8 hours under slow dynamics, and half the time it doesn't finish converging.

---

## limitations

- **q(t) is unobservable** — only composite parameters are identifiable from public data
- **μ = L_close is partly circular** — fitting a damped path to a known endpoint always looks okay; intermediate accuracy is the real test
- **weather timing** — our weather is at kickoff, the line closes hours earlier
- **no real line snapshots** — Phase 3 calibration is synthetic; need Odds API historical tier for real data
- **R² = 6%** — weather is a small factor vs team quality and matchups, but even small mispricings matter if they're systematic

## project structure

```
run.py                         — one-click runner for all phases
pipeline/                      — Phase 1 (data collection, 65 tests)
src/                           — Phase 3 ODE library (model, calibration, diagnostics)
phase2_statistical_modeling.py — Phase 2 script
phase3_ode_model.py            — Phase 3 script
phase4_simulation.py           — Phase 4 script
data_file/                     — all output (figures + data)
```

## deps & tests

```bash
pip install -r requirements.txt
python -m pytest pipeline/tests/ -v    # 65 tests
```
