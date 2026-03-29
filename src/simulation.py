"""
Simulation: generate synthetic line paths and Monte Carlo analysis.
====================================================================

Three simulation modes:

1. DETERMINISTIC SCENARIOS: Generate clean ODE paths for normal vs
   extreme conditions. Used to visualise how parameters affect dynamics.

2. NOISY SCENARIOS: Add observation noise to simulate what real
   snapshot data looks like. Used to test calibration pipeline.

3. MONTE CARLO: Run many simulations to estimate distributional
   properties of closing-line errors and over/under hit rates.

SCENARIO DESIGN (from Section 4 of spec):
- Normal: typical parameters, mu=45.5, L_open=44.0
- Extreme outcome: same dynamics but mu shifts down (wind depresses scoring)
- Extreme dynamics: mu shifts down AND market params change
  (more reactive, less damped — reflecting genuine uncertainty)
"""

import numpy as np
from scipy.integrate import solve_ivp

from src.ode_model import ode_system, solve_ode


def simulate_line_path(k: float, c: float, a: float, eta: float,
                       mu: float, L_open: float, q0: float,
                       t_start: float, t_end: float,
                       n_snapshots: int = 15, sigma_L: float = 0.2,
                       seed: int | None = None) -> dict:
    """Simulate a line path with optional observation noise.

    The ODE is solved deterministically (the dynamics are deterministic).
    Noise is added to simulate measurement/observation uncertainty in
    the snapshot data — the line isn't perfectly observed.

    Parameters
    ----------
    k, c, a, eta : ODE parameters
    mu : fundamental value
    L_open : opening line (initial condition for L)
    q0 : initial exposure (usually 0)
    t_start : market open time (hours before kickoff, negative)
    t_end : kickoff time (usually 0)
    n_snapshots : number of observation times
    sigma_L : std dev of observation noise (in points)
    seed : RNG seed for reproducibility

    Returns
    -------
    dict with dense solution, snapshot observations, and realised outcome.
    """
    rng = np.random.default_rng(seed)

    # Dense solution for smooth plotting (500 points)
    sol_dense = solve_ode(k, c, a, eta, mu, L_open, q0, t_start, t_end, n_dense=500)

    # Snapshot times (what you'd actually observe from an odds API)
    t_snap = np.linspace(t_start, t_end, n_snapshots)
    sol_snap = solve_ode(k, c, a, eta, mu, L_open, q0, t_start, t_end, t_eval=t_snap)

    L_clean = sol_snap["L"]
    L_obs = L_clean + sigma_L * rng.standard_normal(n_snapshots)

    # Simulate realised total points
    # NFL totals have std ~ 13.9 (from Phase 2 descriptive stats)
    sigma_Y = 13.9
    Y = mu + sigma_Y * rng.standard_normal()

    return {
        "t_dense": sol_dense["t"],
        "L_dense": sol_dense["L"],
        "q_dense": sol_dense["q"],
        "t_snap": t_snap,
        "L_clean": L_clean,
        "L_obs": L_obs,
        "q_snap": sol_snap["q"],
        "Y": float(Y),
        "mu": mu,
        "params": {"k": k, "c": c, "a": a, "eta": eta},
    }


def monte_carlo_paths(k: float, c: float, a: float, eta: float,
                      mu: float, L_open: float, q0: float,
                      t_start: float, t_end: float,
                      sigma_L: float, sigma_Y: float,
                      n_sims: int = 1000, seed: int = 0) -> dict:
    """Monte Carlo simulation of closing-line errors and realised outcomes.

    The ODE deterministic path is solved once. Then:
    - Observation noise is added to the closing line: L_close_obs = L_close_true + noise
    - Realised total is drawn: Y ~ N(mu, sigma_Y^2)

    This lets us estimate:
    - What fraction of closing lines land within 1 or 2 points of mu?
    - What's the over/under hit rate?
    - Is there systematic bias?

    Note: the ODE path itself is deterministic. All randomness comes
    from observation noise and the inherent unpredictability of Y.
    """
    rng = np.random.default_rng(seed)

    # Solve ODE once to get the deterministic closing line
    sol = solve_ivp(
        ode_system, (t_start, t_end), [q0, L_open],
        args=(k, c, a, eta, mu),
        method="RK45", rtol=1e-10, atol=1e-12,
        t_eval=[t_end],
    )
    L_close_true = float(sol.y[1][0]) if sol.success else L_open

    # Monte Carlo draws
    closing_noise = sigma_L * rng.standard_normal(n_sims)
    L_close_obs = L_close_true + closing_noise
    Y_sims = mu + sigma_Y * rng.standard_normal(n_sims)

    return {
        "L_close_true": L_close_true,
        "L_close_obs": L_close_obs,
        "Y_sims": Y_sims,
        "closing_errors": L_close_obs - mu,
        "pct_within_1pt": float(np.mean(np.abs(L_close_obs - mu) < 1.0)),
        "pct_within_2pt": float(np.mean(np.abs(L_close_obs - mu) < 2.0)),
        "mean_closing_error": float(np.mean(L_close_obs - mu)),
        "over_hit_rate": float(np.mean(Y_sims > L_close_obs)),
        "n_sims": n_sims,
    }
