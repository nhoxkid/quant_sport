"""
Diagnostics: residual tests and synthetic parameter recovery.
==============================================================

Two roles:

1. SYNTHETIC VALIDATION (Step 0 — do this BEFORE real data):
   Generate data from known parameters, add noise, calibrate,
   check if the optimizer recovers the truth. If it can't recover
   known params from clean synthetic data, it won't work on real
   data either. This is the single most important quality check.

2. RESIDUAL DIAGNOSTICS (after calibration):
   Check whether ODE residuals look like white noise (i.i.d.).
   If residuals show autocorrelation, the ODE is missing dynamics.
   - Ljung-Box test: formal test for autocorrelation at multiple lags
   - Runs test: non-parametric test for randomness of sign pattern
   - Lag-1 autocorrelation: simplest check
"""

import numpy as np
from scipy.integrate import solve_ivp

from src.ode_model import ode_system, solve_ode
from src.calibration import calibrate_game


def synthetic_validation(k_true: float, c_true: float, a_true: float,
                         eta_true: float, mu: float, L_open: float,
                         q0: float, t_start: float, t_end: float,
                         n_snapshots: int, sigma: float,
                         n_trials: int = 100, seed: int = 0) -> dict:
    """Run Monte Carlo parameter recovery experiment.

    Protocol (from Section 2.1 of the spec):
    1. Simulate the true ODE to get L_true(t) at n_snapshots times.
    2. Add Gaussian noise: L_obs = L_true + sigma * Z.
    3. Calibrate and record estimated parameters.
    4. Repeat n_trials times with different noise seeds.
    5. Report relative errors: |param_est - param_true| / param_true.

    This tells us:
    - Can the optimizer find the true parameters AT ALL?
    - Which parameters are well-identified vs. poorly identified?
    - How does noise level (sigma) and snapshot count affect recovery?
    - Are composite params (gamma, omega_sq) more robust than individuals?

    Returns
    -------
    dict with arrays of estimated params, relative errors, and summary stats.
    """
    rng = np.random.default_rng(seed)
    t_snap = np.linspace(t_start, t_end, n_snapshots)

    # Generate true L(t) at snapshot times (no noise)
    sol_true = solve_ivp(
        ode_system, (t_start, t_end), [q0, L_open],
        args=(k_true, c_true, a_true, eta_true, mu),
        t_eval=t_snap, method="RK45", rtol=1e-10, atol=1e-12,
    )
    L_true = sol_true.y[1]

    # True composite parameters
    gamma_true = c_true + eta_true
    omega_sq_true = c_true * eta_true + a_true * k_true

    results = {
        "k": [], "c": [], "a": [], "eta": [],
        "gamma": [], "omega_sq": [],
        "rmse": [], "success": [],
    }

    for trial in range(n_trials):
        noise = sigma * rng.standard_normal(n_snapshots)
        L_obs = L_true + noise

        try:
            cal = calibrate_game(t_snap, L_obs, mu, seed=seed + trial)
            results["k"].append(cal["k"])
            results["c"].append(cal["c"])
            results["a"].append(cal["a"])
            results["eta"].append(cal["eta"])
            results["gamma"].append(cal["gamma"])
            results["omega_sq"].append(cal["omega_sq"])
            results["rmse"].append(cal["rmse"])
            results["success"].append(cal["success"])
        except Exception:
            for key in results:
                results[key].append(np.nan)

    # Convert to arrays
    for key in results:
        results[key] = np.array(results[key])

    # Compute relative errors (ignoring failed trials)
    def _rel_err(est, true_val):
        valid = np.isfinite(est)
        if not valid.any():
            return np.full_like(est, np.nan)
        re = np.full_like(est, np.nan)
        re[valid] = np.abs(est[valid] - true_val) / abs(true_val) if true_val != 0 else np.abs(est[valid])
        return re

    results["k_relerr"] = _rel_err(results["k"], k_true)
    results["c_relerr"] = _rel_err(results["c"], c_true)
    results["a_relerr"] = _rel_err(results["a"], a_true)
    results["eta_relerr"] = _rel_err(results["eta"], eta_true)
    results["gamma_relerr"] = _rel_err(results["gamma"], gamma_true)
    results["omega_sq_relerr"] = _rel_err(results["omega_sq"], omega_sq_true)

    # Summary statistics
    def _summary(arr):
        valid = arr[np.isfinite(arr)]
        if len(valid) == 0:
            return {"median": np.nan, "iqr_lo": np.nan, "iqr_hi": np.nan}
        return {
            "median": float(np.median(valid)),
            "iqr_lo": float(np.percentile(valid, 25)),
            "iqr_hi": float(np.percentile(valid, 75)),
        }

    results["summary"] = {
        "k": _summary(results["k_relerr"]),
        "c": _summary(results["c_relerr"]),
        "a": _summary(results["a_relerr"]),
        "eta": _summary(results["eta_relerr"]),
        "gamma": _summary(results["gamma_relerr"]),
        "omega_sq": _summary(results["omega_sq_relerr"]),
    }

    results["true_params"] = {
        "k": k_true, "c": c_true, "a": a_true, "eta": eta_true,
        "gamma": gamma_true, "omega_sq": omega_sq_true,
    }
    results["config"] = {
        "n_snapshots": n_snapshots, "sigma": sigma, "n_trials": n_trials,
    }

    return results


def diagnose_residuals(residuals: np.ndarray, t_obs: np.ndarray) -> dict:
    """Check whether ODE fit residuals are consistent with white noise.

    Three tests:
    1. Ljung-Box: formal test for autocorrelation up to lag 3.
       H0: residuals are uncorrelated. Low p => significant autocorrelation.

    2. Runs test: counts how many times the residual sign flips.
       If residuals are random, the sign should flip roughly half the time.
       Too few runs => systematic pattern (model missing dynamics).

    3. Lag-1 autocorrelation: simplest check. |r1| > 0.3 is concerning.
    """
    results = {}
    n = len(residuals)

    # ── Ljung-Box test ────────────────────────────────────────────────
    if n >= 8:
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            max_lag = min(3, n // 2 - 1)
            if max_lag >= 1:
                lb = acorr_ljungbox(residuals, lags=max_lag, return_df=True)
                results["ljung_box_p"] = lb["lb_pvalue"].values.tolist()
                results["ljung_box_significant"] = any(p < 0.05 for p in results["ljung_box_p"])
        except ImportError:
            results["ljung_box_p"] = None
            results["ljung_box_significant"] = None

    # ── Runs test ─────────────────────────────────────────────────────
    signs = np.sign(residuals)
    signs = signs[signs != 0]  # remove exact zeros
    if len(signs) >= 3:
        runs = 1 + int(np.sum(np.abs(np.diff(signs)) > 0))
        n_pos = int(np.sum(signs > 0))
        n_neg = int(np.sum(signs < 0))
        if n_pos > 0 and n_neg > 0:
            expected_runs = 1 + 2 * n_pos * n_neg / (n_pos + n_neg)
            results["runs"] = runs
            results["expected_runs"] = float(expected_runs)
            results["runs_ratio"] = runs / expected_runs  # near 1 = good

    # ── Lag-1 autocorrelation ─────────────────────────────────────────
    if n >= 4:
        r1 = float(np.corrcoef(residuals[:-1], residuals[1:])[0, 1])
        results["lag1_autocorr"] = r1
        results["lag1_concerning"] = abs(r1) > 0.3

    return results
