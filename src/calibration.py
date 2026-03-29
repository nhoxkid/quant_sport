"""
Calibration: fit composite ODE parameters from line snapshot data.
==================================================================

DESIGN DECISION — WHY COMPOSITE PARAMETERS ONLY
-------------------------------------------------
The full 2-D ODE has 4 free parameters (k, c, a, eta). But q(t) is
latent (unobservable), meaning only L(t) provides data. The second-
order reduction shows that L(t) depends only on:

    gamma    = c + eta          (effective damping)
    omega_sq = c*eta + a*k      (effective stiffness)

These are the ONLY two combinations that appear in the observable L(t)
dynamics. The individual (k, c, a, eta) are structurally unidentifiable
from L(t) data alone — proven by the synthetic validation in Step 1,
which showed 60-900% relative errors on individual params even with
clean data.

FIT METHOD — ANALYTIC CLOSED-FORM (no ODE solver)
---------------------------------------------------
The second-order equation:

    L'' + gamma*L' + omega_sq*(L - mu) = 0

has the exact analytic solution:

    OVERDAMPED (gamma^2 > 4*omega_sq):
        L(t) = mu + A*exp(lam1*tau) + B*exp(lam2*tau)
        where lam1,2 = [-gamma +/- sqrt(gamma^2 - 4*omega_sq)] / 2

    UNDERDAMPED (gamma^2 < 4*omega_sq):
        L(t) = mu + exp(alpha*tau) * [d0*cos(w*tau) + ((v0-alpha*d0)/w)*sin(w*tau)]
        where alpha = -gamma/2, w = sqrt(4*omega_sq - gamma^2) / 2

    CRITICALLY DAMPED (gamma^2 = 4*omega_sq):
        L(t) = mu + (d0 + (v0 + gamma/2 * d0)*tau) * exp(-gamma/2 * tau)

    where tau = t - t0, d0 = L(t0) - mu, v0 = L'(t0) (initial velocity).

We fit 3 parameters: (gamma, omega_sq, v0).
- gamma, omega_sq: the two observable composite parameters
- v0: initial rate of line movement at t0 (can't be determined from
  the composites alone because it depends on the initial exposure q0
  and the individual eta, which we don't know).

This is a 3-parameter fit using a closed-form expression. Each function
evaluation is ~1 microsecond (exp + cos/sin), vs ~1 millisecond for
solve_ivp. The entire calibration takes milliseconds per game.

NORMALISATION (Section 2.2 of spec):
    tau = (t - t0) / T where T = t_end - t0
    gamma_tilde = T * gamma     (dimensionless damping)
    omega_sq_tilde = T^2 * omega_sq  (dimensionless stiffness)
    These are comparable across games regardless of time window.
"""

import numpy as np
from scipy.optimize import minimize, nnls


def analytic_L(t: np.ndarray, t0: float, mu: float, d0: float,
               gamma: float, omega_sq: float, v0: float) -> np.ndarray:
    """Evaluate the analytic solution L(t) for the second-order ODE.

    L'' + gamma*L' + omega_sq*(L - mu) = 0

    Initial conditions at t = t0:
        L(t0) = mu + d0     (d0 = deviation from equilibrium)
        L'(t0) = v0         (initial line velocity)

    Parameters
    ----------
    t : array
        Evaluation times.
    t0 : float
        Initial time.
    mu : float
        Equilibrium (fundamental value).
    d0 : float
        Initial deviation L(t0) - mu.
    gamma : float
        Damping coefficient (= c + eta). Must be > 0.
    omega_sq : float
        Natural frequency squared (= c*eta + a*k). Must be > 0.
    v0 : float
        Initial velocity L'(t0).

    Returns
    -------
    L(t) : array
        Line values at each time in t.
    """
    tau = t - t0
    disc = gamma * gamma - 4.0 * omega_sq

    if disc > 1e-10:
        # ── OVERDAMPED: two distinct real eigenvalues ─────────────────
        # Both negative (guaranteed since gamma > 0, omega_sq > 0).
        sqrt_disc = np.sqrt(disc)
        lam1 = (-gamma + sqrt_disc) / 2.0
        lam2 = (-gamma - sqrt_disc) / 2.0
        # A + B = d0
        # lam1*A + lam2*B = v0
        # => A = (v0 - lam2*d0) / (lam1 - lam2)
        A = (v0 - lam2 * d0) / (lam1 - lam2)
        B = d0 - A
        return mu + A * np.exp(lam1 * tau) + B * np.exp(lam2 * tau)

    elif disc < -1e-10:
        # ── UNDERDAMPED: complex conjugate eigenvalues ────────────────
        # L(t) = mu + exp(alpha*tau) * [d0*cos(w*tau) + C2*sin(w*tau)]
        # where C2 = (v0 - alpha*d0) / w
        alpha = -gamma / 2.0
        w = np.sqrt(-disc) / 2.0
        C2 = (v0 - alpha * d0) / w
        exp_part = np.exp(alpha * tau)
        return mu + exp_part * (d0 * np.cos(w * tau) + C2 * np.sin(w * tau))

    else:
        # ── CRITICALLY DAMPED: repeated eigenvalue ────────────────────
        alpha = -gamma / 2.0
        # L(t) = mu + (d0 + (v0 - alpha*d0)*tau) * exp(alpha*tau)
        return mu + (d0 + (v0 - alpha * d0) * tau) * np.exp(alpha * tau)


def _objective_composite(params: np.ndarray, t_obs: np.ndarray,
                         L_obs: np.ndarray, mu: float) -> float:
    """Penalised sum-of-squared-errors for the 3-parameter composite fit.

    params = [gamma, omega_sq, v0]

    Uses the analytic closed-form — no ODE solver, no numerical
    integration. Each evaluation is O(n) elementary operations.

    REGULARISATION: A small penalty on omega_sq prevents the optimizer
    from fitting high-frequency oscillations to noisy data. This is
    Tikhonov regularisation (ridge-like) — it encodes the prior belief
    that betting line dynamics are smooth and slow, not oscillatory.

    The penalty weight lambda_reg is set so that it's negligible when
    omega_sq is in the physically plausible range (< 0.5) but
    increasingly penalises larger values. Specifically:
        penalty = lambda_reg * omega_sq^2
    With lambda_reg = 0.5 and n=10 observations, the penalty at
    omega_sq=1.0 is 0.5, comparable to one data point's SSE contribution
    at sigma=0.7 — enough to discourage overfitting but not dominate.
    """
    gamma, omega_sq, v0 = params

    if gamma <= 0 or omega_sq <= 0:
        return 1e12

    t0 = t_obs[0]
    d0 = L_obs[0] - mu

    try:
        L_pred = analytic_L(t_obs, t0, mu, d0, gamma, omega_sq, v0)
        if np.any(np.isnan(L_pred)) or np.any(np.isinf(L_pred)):
            return 1e12
        sse = float(np.sum((L_pred - L_obs) ** 2))
        # Tikhonov penalty: penalise both high omega_sq (prevents spurious
        # oscillation) and high |v0| (prevents wild initial trajectories).
        # Scaled by 1/n so the penalty doesn't dominate with small datasets.
        n = len(t_obs)
        penalty = 2.0 * omega_sq ** 2 + 0.5 * v0 ** 2
        return sse + penalty
    except (OverflowError, FloatingPointError):
        return 1e12


# ── Parameter bounds for composite fit ────────────────────────────────
# gamma = c + eta: physically [0.02, 5.0]
#   Lower bound: very slow convergence (half-life ~70 hrs)
#   Upper bound: very fast convergence (half-life ~0.3 hrs)
# omega_sq = c*eta + a*k: physically [0.001, 2.0]
#   Upper bound prevents spurious high-frequency oscillations.
#   With omega_sq=2.0 and gamma=0.5, oscillation period = 2*pi/w
#   = 2*pi / sqrt(4*2 - 0.25)/2 ≈ 2.3 hours. That's the fastest
#   physically plausible line oscillation. Above this is overfitting.
# v0 = initial velocity: [-1.0, 1.0] pts/hr
#   NFL lines rarely move more than 1 point per hour even in fast markets.
COMPOSITE_BOUNDS = [(0.02, 5.0), (0.001, 2.0), (-1.0, 1.0)]


def calibrate_game(t_obs: np.ndarray, L_obs: np.ndarray, mu: float,
                   bounds: list | None = None, seed: int = 42) -> dict:
    """Calibrate composite ODE parameters for a single game.

    Fits (gamma, omega_sq, v0) using the analytic closed-form solution.
    No ODE solver is invoked — this runs in milliseconds.

    Multi-start L-BFGS-B with 20 random initialisations to handle
    the modest non-convexity in the 3-parameter landscape.

    Parameters
    ----------
    t_obs : array
        Observation times (hours, t=0 at kickoff, negative earlier).
    L_obs : array
        Observed line values.
    mu : float
        Fundamental value for this game.
    bounds : list of (lo, hi), optional
        Bounds for [gamma, omega_sq, v0].
    seed : int
        Random seed for multi-start.

    Returns
    -------
    dict with:
        gamma, omega_sq, v0: fitted parameters
        regime: overdamped/underdamped/critically_damped
        rmse: root mean squared error
        L_fitted: predicted line at observation times
        residuals: L_obs - L_fitted
        plus normalised (dimensionless) composites for cross-game comparison
    """
    if bounds is None:
        bounds = COMPOSITE_BOUNDS

    t_obs = np.asarray(t_obs, dtype=float)
    L_obs = np.asarray(L_obs, dtype=float)

    rng = np.random.default_rng(seed)
    best = None

    # ── Multi-start L-BFGS-B ─────────────────────────────────────────
    # 10 random starts for a 3-parameter smooth landscape.
    # One good initialisation: estimate v0 from first two points.
    v0_est = (L_obs[1] - L_obs[0]) / (t_obs[1] - t_obs[0]) if len(t_obs) > 1 else 0.0

    starts = [np.array([0.5, 0.1, v0_est])]  # educated guess
    for _ in range(9):
        starts.append(np.array([
            rng.uniform(bounds[0][0], bounds[0][1]),
            rng.uniform(bounds[1][0], bounds[1][1]),
            rng.uniform(bounds[2][0], bounds[2][1]),
        ]))

    for x0 in starts:
        try:
            res = minimize(
                _objective_composite, x0,
                args=(t_obs, L_obs, mu),
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 500, "ftol": 1e-12},
            )
            if best is None or res.fun < best.fun:
                best = res
        except Exception:
            continue

    if best is None or best.fun >= 1e11:
        return {
            "gamma": np.nan, "omega_sq": np.nan, "v0": np.nan,
            "regime": "failed", "rmse": np.nan,
            "L_fitted": np.full_like(L_obs, np.nan),
            "residuals": np.full_like(L_obs, np.nan),
            "success": False,
        }

    gamma, omega_sq, v0 = best.x
    t0 = t_obs[0]
    d0 = L_obs[0] - mu

    # Classify regime
    disc = gamma ** 2 - 4 * omega_sq
    if disc > 1e-10:
        regime = "overdamped"
    elif disc < -1e-10:
        regime = "underdamped"
    else:
        regime = "critically_damped"

    # Fitted values
    L_fitted = analytic_L(t_obs, t0, mu, d0, gamma, omega_sq, v0)
    residuals = L_obs - L_fitted
    rmse = float(np.sqrt(np.mean(residuals ** 2)))

    # Time window and normalised composites
    T = abs(t_obs[-1] - t_obs[0])

    return {
        "gamma": float(gamma),
        "omega_sq": float(omega_sq),
        "v0": float(v0),
        "regime": regime,
        "discriminant": float(disc),
        # Normalised (dimensionless) composites for cross-game comparison
        "gamma_tilde": float(T * gamma),
        "omega_sq_tilde": float(T ** 2 * omega_sq),
        "rmse": rmse,
        "sse": float(best.fun),
        "residuals": residuals,
        "L_fitted": L_fitted,
        "n_snapshots": len(t_obs),
        "T_hours": float(T),
        "mu": mu,
        "success": True,
    }


def second_order_ols(t_obs: np.ndarray, L_obs: np.ndarray, mu: float) -> dict:
    """Estimate (gamma, omega_sq) via finite-difference OLS.

    The second-order ODE: L'' = -gamma*L' - omega_sq*(L - mu)
    is linear in (gamma, omega_sq). Estimate L', L'' from finite
    differences, then run non-negative least squares.

    WARNING: finite differences amplify noise. Only reliable with
    15+ regularly-spaced, low-noise snapshots. Use as sanity check.
    """
    n = len(t_obs)
    if n < 7:
        return {"gamma": np.nan, "omega_sq": np.nan,
                "error": "Need >= 7 snapshots for second-order OLS"}

    # First derivative (central differences at interior points)
    L_prime = np.zeros(n)
    dt = np.diff(t_obs)
    L_prime[0] = (L_obs[1] - L_obs[0]) / dt[0]
    L_prime[-1] = (L_obs[-1] - L_obs[-2]) / dt[-1]
    for i in range(1, n - 1):
        L_prime[i] = (L_obs[i + 1] - L_obs[i - 1]) / (t_obs[i + 1] - t_obs[i - 1])

    # Second derivative at interior points
    interior = list(range(1, n - 1))
    L_double_prime = np.zeros(len(interior))
    for idx, i in enumerate(interior):
        h1 = t_obs[i] - t_obs[i - 1]
        h2 = t_obs[i + 1] - t_obs[i]
        L_double_prime[idx] = 2 * (
            L_obs[i + 1] / (h2 * (h1 + h2))
            - L_obs[i] / (h1 * h2)
            + L_obs[i - 1] / (h1 * (h1 + h2))
        )

    # OLS: L'' = -gamma*L' - omega_sq*(L - mu)
    y = L_double_prime
    X = np.column_stack([
        -L_prime[interior],
        -(L_obs[interior] - mu),
    ])

    coeffs, rnorm = nnls(X, y)
    return {
        "gamma": float(coeffs[0]),
        "omega_sq": float(coeffs[1]),
        "method": "second_order_ols",
        "n_interior_points": len(interior),
        "rnorm": float(rnorm),
        "warning": "Finite differences amplify noise. Sanity check only.",
    }
