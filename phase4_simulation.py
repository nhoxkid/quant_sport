#!/usr/bin/env python3
"""
Phase 4 — Simulation Module: NFL Totals 2-D ODE/SDE + Monte Carlo
===================================================================

Architecture mirrors the GBM Monte Carlo Studio (C:/git/GBM):
  - Frozen dataclasses for configs (like RegimeConfig)
  - Vectorised batch paths (like GBMSimulator)
  - Stats aggregation into immutable dataclasses (like MonteCarloSummary)
  - Single-file layout (like gbm_simulation.py)

KEY DIFFERENCE FROM GBM: Additive noise, NOT multiplicative.
  GBM:       dS = mu*S*dt + sigma*S*dW        (multiplicative/Itô)
  This model: dL = [drift]*dt + sigma_L*dW     (additive)
  The sqrt(dt)*Z term is NOT multiplied by the state variable.

MATHEMATICAL FOUNDATION
-----------------------
The 2-D ODE system (Eqs 5.2–5.3 from the project PDF):

    dq/dt = k*(mu(t) - L(t)) - c*q(t)       ... (5.2)
    dL/dt = a*q(t) + eta*(mu(t) - L(t))      ... (5.3)

SDE extension (additive noise):
    dq = [k*(mu - L) - c*q]*dt + sigma_q*dW1
    dL = [a*q + eta*(mu - L)]*dt + sigma_L*dW2
    where W1, W2 are independent Wiener processes.

Euler-Maruyama discretisation:
    q_{n+1} = q_n + [k*(mu_n - L_n) - c*q_n]*dt + sigma_q*sqrt(dt)*Z1
    L_{n+1} = L_n + [a*q_n + eta*(mu_n - L_n)]*dt + sigma_L*sqrt(dt)*Z2
    where Z1, Z2 ~ N(0,1) iid.

Observation noise (Eq 7.1):
    L_obs(t_i) = L(t_i) + sigma_obs*Z_i

Realised total (Eq 7.2):
    Y = mu_final + sigma_Y*Z

Second-order reduction (constant mu only):
    L'' + (c+eta)*L' + (c*eta+a*k)*(L-mu) = 0

Eigenvalues:
    lambda = [-(c+eta) ± sqrt((c-eta)^2 - 4*a*k)] / 2

Discriminant: Delta = (c-eta)^2 - 4*a*k
    Delta < 0: underdamped (oscillatory)
    Delta > 0: overdamped (monotone)
    Delta = 0: critically damped
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Constants ─────────────────────────────────────────────────────────
FIG_DIR = Path("data_file/phase4")
FIG_DPI = 200

# Color scheme (spec-defined)
C_BASE = "#2563EB"     # blue
C_BASE_FAN = "#93C5FD"
C_EXTREME = "#DC2626"  # red
C_EXTREME_FAN = "#FCA5A5"
C_SHOCK = "#7C3AED"    # purple
C_SHOCK_FAN = "#C4B5FD"


# =====================================================================
# 1. ODEParams — immutable configuration dataclass
# =====================================================================

@dataclass(frozen=True)
class ODEParams:
    """Immutable configuration for a single simulation scenario.

    Mirrors GBM's RegimeConfig: frozen dataclass with derived properties
    computed at access time, plus post-init validation.

    Parameters
    ----------
    k : float > 0
        Betting-pressure sensitivity. How fast bettors respond to
        perceived mispricing (mu - L). [exposure * hr^-1 * pt^-1]
    c : float > 0
        Exposure decay rate. Mean-reversion of bookmaker liability.
        Half-life = ln(2)/c hours. [hr^-1]
    a : float > 0
        Line-adjustment rate. How aggressively the book moves L in
        response to net exposure q. [pt * hr^-1 * exposure^-1]
    eta : float > 0
        Information-tracking speed. The line drifts toward mu even
        without betting flow (bookmaker's own model). [hr^-1]
    mu_base : float
        Fundamental total (pre-shock value).
    L_open : float
        Opening line value at t_start.
    q_init : float
        Initial bookmaker exposure (usually 0 = balanced book).
    sigma_q : float >= 0
        Diffusion coefficient for exposure process (additive noise).
    sigma_L : float >= 0
        Diffusion coefficient for line process (additive noise).
    sigma_obs : float >= 0
        Observation noise std dev (Eq 7.1).
    sigma_Y : float > 0
        Realised total noise std dev (Eq 7.2). ~13.9 from Phase 2.
    t_start : float
        Start time in hours (negative, e.g. -48).
    t_end : float
        End time (kickoff), usually 0.
    n_steps : int
        Number of Euler-Maruyama time steps.
    label : str
        Human-readable scenario name.
    mu_schedule : tuple
        Sequence of (t_switch, mu_new) for time-varying mu.
        Empty tuple = constant mu. Applied in chronological order.
        Example: ((-8.0, 42.5),) means mu drops to 42.5 at t=-8h.
    """
    k: float
    c: float
    a: float
    eta: float
    mu_base: float
    L_open: float
    q_init: float = 0.0
    sigma_q: float = 0.0
    sigma_L: float = 0.0
    sigma_obs: float = 0.0
    sigma_Y: float = 10.0
    t_start: float = -48.0
    t_end: float = 0.0
    n_steps: int = 960
    label: str = "unnamed"
    mu_schedule: tuple = ()

    def __post_init__(self):
        """Validate all parameters (Proposition 5.1 stability conditions)."""
        assert self.k > 0, f"k must be > 0, got {self.k}"
        assert self.c > 0, f"c must be > 0, got {self.c}"
        assert self.a > 0, f"a must be > 0, got {self.a}"
        assert self.eta > 0, f"eta must be > 0, got {self.eta}"
        assert self.sigma_q >= 0, f"sigma_q must be >= 0"
        assert self.sigma_L >= 0, f"sigma_L must be >= 0"
        assert self.sigma_obs >= 0, f"sigma_obs must be >= 0"
        assert self.sigma_Y > 0, f"sigma_Y must be > 0"
        assert self.t_start < self.t_end, "t_start must be < t_end"
        assert self.n_steps > 0, "n_steps must be > 0"

    # ── Derived properties ────────────────────────────────────────────

    @property
    def dt(self) -> float:
        """Time step size in hours."""
        return (self.t_end - self.t_start) / self.n_steps

    @property
    def t_grid(self) -> np.ndarray:
        """Full time grid from t_start to t_end (n_steps + 1 points)."""
        return np.linspace(self.t_start, self.t_end, self.n_steps + 1)

    @property
    def jacobian(self) -> np.ndarray:
        """Jacobian matrix A of the linearised system at equilibrium.

        A = [[-c, -k],
             [ a, -eta]]

        The state vector is [q, L-mu]. Eigenvalues of A determine
        the stability and oscillation properties.
        """
        return np.array([[-self.c, -self.k],
                         [self.a, -self.eta]])

    @property
    def trace(self) -> float:
        """tr(A) = -(c + eta). Must be < 0 for stability."""
        return -(self.c + self.eta)

    @property
    def determinant(self) -> float:
        """det(A) = c*eta + a*k. Must be > 0 for stability."""
        return self.c * self.eta + self.a * self.k

    @property
    def discriminant(self) -> float:
        """Delta = (c - eta)^2 - 4*a*k.

        Derived from the characteristic equation of the Jacobian:
            lambda^2 + (c+eta)*lambda + (c*eta + a*k) = 0
        Discriminant of this quadratic = (c+eta)^2 - 4*(c*eta+a*k)
            = c^2 + 2*c*eta + eta^2 - 4*c*eta - 4*a*k
            = (c - eta)^2 - 4*a*k

        NOT (c+eta)^2 - 4*det. The simplification matters.
        """
        return (self.c - self.eta) ** 2 - 4 * self.a * self.k

    @property
    def eigenvalues(self) -> tuple[complex, complex]:
        """Eigenvalues of the Jacobian.

        lambda = [-(c+eta) ± sqrt(Delta)] / 2
        Complex when Delta < 0 (oscillatory regime).
        """
        gamma = self.c + self.eta
        d = self.discriminant
        if d >= 0:
            sq = np.sqrt(d)
            return ((-gamma + sq) / 2, (-gamma - sq) / 2)
        else:
            sq = np.sqrt(-d)
            return (complex(-gamma / 2, sq / 2), complex(-gamma / 2, -sq / 2))

    @property
    def is_oscillatory(self) -> bool:
        """True if discriminant < 0 (underdamped, complex eigenvalues)."""
        return self.discriminant < -1e-12

    @property
    def decay_time(self) -> float:
        """Time constant for exponential decay = 1 / |Re(lambda)|.

        Both eigenvalues have the same real part magnitude = (c+eta)/2.
        Decay time = 2 / (c + eta).
        """
        return 2.0 / (self.c + self.eta)

    @property
    def oscillation_period(self) -> float | None:
        """Period of oscillation (hours), or None if overdamped.

        omega = sqrt(-Delta) / 2, period = 2*pi / omega.
        """
        if not self.is_oscillatory:
            return None
        omega = np.sqrt(-self.discriminant) / 2
        return 2 * np.pi / omega if omega > 0 else None

    @property
    def mu_final(self) -> float:
        """mu at kickoff (t=0). Accounts for mu_schedule."""
        mu = self.mu_base
        for t_switch, mu_new in self.mu_schedule:
            if t_switch <= self.t_end:
                mu = mu_new
        return mu

    def mu_at(self, t: float) -> float:
        """Evaluate mu at a specific time, respecting mu_schedule."""
        mu = self.mu_base
        for t_switch, mu_new in self.mu_schedule:
            if t >= t_switch:
                mu = mu_new
        return mu

    def mu_array(self) -> np.ndarray:
        """Vectorised mu over the full time grid (n_steps+1 values).

        For the late-shock scenario, this is a step function.
        """
        t = self.t_grid
        mu = np.full_like(t, self.mu_base)
        for t_switch, mu_new in self.mu_schedule:
            mu[t >= t_switch] = mu_new
        return mu

    def summary(self) -> str:
        """Diagnostic string with all derived quantities."""
        ev = self.eigenvalues
        lines = [
            f"Scenario: {self.label}",
            f"  ODE params: k={self.k}, c={self.c}, a={self.a}, eta={self.eta}",
            f"  mu_base={self.mu_base}, L_open={self.L_open}, q_init={self.q_init}",
            f"  Noise: sigma_q={self.sigma_q}, sigma_L={self.sigma_L}, "
            f"sigma_obs={self.sigma_obs}, sigma_Y={self.sigma_Y}",
            f"  Time: [{self.t_start}, {self.t_end}], n_steps={self.n_steps}, dt={self.dt:.4f}h",
            f"  Jacobian trace = {self.trace:.4f} (< 0: stable)",
            f"  Jacobian det   = {self.determinant:.4f} (> 0: stable)",
            f"  Discriminant   = {self.discriminant:.4f}",
            f"  Eigenvalues    = {ev[0]:.4f}, {ev[1]:.4f}",
            f"  Regime: {'oscillatory' if self.is_oscillatory else 'monotone'}",
            f"  Decay time     = {self.decay_time:.2f} h",
        ]
        if self.is_oscillatory:
            lines.append(f"  Osc. period    = {self.oscillation_period:.2f} h")
        if self.mu_schedule:
            lines.append(f"  mu_schedule    = {self.mu_schedule}")
            lines.append(f"  mu_final       = {self.mu_final}")
        return "\n".join(lines)


# =====================================================================
# 2. ODESolver — deterministic ODE integration + analytic cross-check
# =====================================================================

class ODESolver:
    """Deterministic ODE solver for the 2-D coupled system.

    Two methods:
    1. solve_system(): scipy.integrate.solve_ivp (RK45) — works for
       constant AND time-varying mu.
    2. solve_analytical(): closed-form of the second-order equation —
       constant mu ONLY. Used for cross-validation.

    Mirroring pattern: like GBMSimulator.simulate() but deterministic.
    """

    @staticmethod
    def _rhs(t: float, y: np.ndarray, params: ODEParams) -> list[float]:
        """Right-hand side of the 2-D ODE.

        dq/dt = k*(mu(t) - L) - c*q       ... (Eq 5.2)
        dL/dt = a*q + eta*(mu(t) - L)      ... (Eq 5.3)
        """
        q, L = y
        mu = params.mu_at(t)
        dqdt = params.k * (mu - L) - params.c * q
        dLdt = params.a * q + params.eta * (mu - L)
        return [dqdt, dLdt]

    @staticmethod
    def solve_system(params: ODEParams, t_eval: np.ndarray | None = None) -> dict:
        """Solve the full 2-D ODE numerically via RK45.

        Returns dict with t, q, L arrays.
        """
        if t_eval is None:
            t_eval = params.t_grid

        sol = solve_ivp(
            ODESolver._rhs,
            (params.t_start, params.t_end),
            [params.q_init, params.L_open],
            args=(params,),
            t_eval=t_eval,
            method="RK45",
            rtol=1e-10, atol=1e-12,
        )
        return {"t": sol.t, "q": sol.y[0], "L": sol.y[1], "success": sol.success}

    @staticmethod
    def solve_analytical(params: ODEParams) -> np.ndarray:
        """Analytic solution for L(t) via the second-order equation.

        CONSTANT MU ONLY. For time-varying mu (late-shock), this is
        not applicable — use solve_system() instead.

        Second-order ODE: L'' + (c+eta)*L' + (c*eta+a*k)*(L-mu) = 0

        Initial conditions:
            L(t0) = L_open
            L'(t0) = a*q_init + eta*(mu - L_open)    [from Eq 5.3]

        General solution: L(t) = mu + f(t-t0), where f depends on regime.
        """
        if params.mu_schedule:
            raise ValueError("Analytical solution requires constant mu. "
                             "Use solve_system() for time-varying mu.")

        mu = params.mu_base
        t = params.t_grid
        t0 = params.t_start
        tau = t - t0

        d0 = params.L_open - mu  # initial deviation
        v0 = params.a * params.q_init + params.eta * (mu - params.L_open)

        gamma = params.c + params.eta
        disc = params.discriminant

        if disc > 1e-10:
            # OVERDAMPED: two distinct real eigenvalues
            sq = np.sqrt(disc)
            lam1 = (-gamma + sq) / 2
            lam2 = (-gamma - sq) / 2
            A = (v0 - lam2 * d0) / (lam1 - lam2)
            B = d0 - A
            L = mu + A * np.exp(lam1 * tau) + B * np.exp(lam2 * tau)

        elif disc < -1e-10:
            # UNDERDAMPED: complex conjugate eigenvalues
            alpha = -gamma / 2
            w = np.sqrt(-disc) / 2
            C2 = (v0 - alpha * d0) / w
            L = mu + np.exp(alpha * tau) * (
                d0 * np.cos(w * tau) + C2 * np.sin(w * tau)
            )

        else:
            # CRITICALLY DAMPED
            alpha = -gamma / 2
            L = mu + (d0 + (v0 - alpha * d0) * tau) * np.exp(alpha * tau)

        return L


# =====================================================================
# 3. SDESimulator — Euler-Maruyama stepper (adapted from GBM Itô stepper)
# =====================================================================

class SDESimulator:
    """Vectorised Euler-Maruyama SDE simulator for the 2-D system.

    Adapted from GBM's batch simulation architecture:
    - Pre-allocates arrays of shape (n_paths, n_steps+1) for q and L
    - Pre-draws ALL noise upfront: Z_q, Z_L of shape (n_paths, n_steps)
    - Single vectorised loop over time steps (no loop over paths)

    KEY DIFFERENCE FROM GBM:
        GBM: dS = mu*S*dt + sigma*S*dW        (multiplicative noise)
        Here: dL = [drift]*dt + sigma_L*dW     (additive noise)

    The noise term sigma*sqrt(dt)*Z is NOT multiplied by L(t).
    This is the critical distinction between Itô (GBM) and additive SDEs.

    Euler-Maruyama update (per step n, vectorised over paths):
        q[:, n+1] = q[:, n] + [k*(mu_n - L[:, n]) - c*q[:, n]]*dt
                    + sigma_q * sqrt_dt * Z_q[:, n]
        L[:, n+1] = L[:, n] + [a*q[:, n] + eta*(mu_n - L[:, n])]*dt
                    + sigma_L * sqrt_dt * Z_L[:, n]
    """

    @staticmethod
    def simulate(params: ODEParams, n_paths: int = 1000,
                 seed: int = 42) -> dict:
        """Run the Euler-Maruyama simulation.

        Parameters
        ----------
        params : ODEParams
            Scenario configuration.
        n_paths : int
            Number of Monte Carlo paths (vectorised, no path loop).
        seed : int
            RNG seed for reproducibility.

        Returns
        -------
        dict with:
            t: (n_steps+1,) time grid
            q: (n_paths, n_steps+1) exposure paths
            L: (n_paths, n_steps+1) line paths (true process)
            L_obs: (n_paths, n_obs) observed line at ~30min intervals
            t_obs: (n_obs,) observation times
            Y: (n_paths,) realised totals
        """
        rng = np.random.default_rng(seed)
        ns = params.n_steps
        dt = params.dt
        sqrt_dt = np.sqrt(dt)

        # Pre-allocate (mirrors GBM's torch.empty pattern)
        q = np.empty((n_paths, ns + 1))
        L = np.empty((n_paths, ns + 1))
        q[:, 0] = params.q_init
        L[:, 0] = params.L_open

        # Pre-draw ALL noise upfront (mirrors GBM's batch-noise pattern)
        Z_q = rng.standard_normal((n_paths, ns))
        Z_L = rng.standard_normal((n_paths, ns))

        # Vectorised mu array for time-varying support
        mu_arr = params.mu_array()  # shape (ns+1,)

        # ── Euler-Maruyama stepping (vectorised over paths) ───────────
        for n in range(ns):
            mu_n = mu_arr[n]
            q_n = q[:, n]
            L_n = L[:, n]

            # Drift (Eqs 5.2, 5.3)
            drift_q = params.k * (mu_n - L_n) - params.c * q_n
            drift_L = params.a * q_n + params.eta * (mu_n - L_n)

            # Diffusion (ADDITIVE — not multiplied by state)
            diff_q = params.sigma_q * sqrt_dt * Z_q[:, n]
            diff_L = params.sigma_L * sqrt_dt * Z_L[:, n]

            q[:, n + 1] = q_n + drift_q * dt + diff_q
            L[:, n + 1] = L_n + drift_L * dt + diff_L

        # ── Observation noise (Eq 7.1) ────────────────────────────────
        # Sample at ~30min intervals
        obs_every = max(1, int(0.5 / dt))  # 0.5h / dt steps
        obs_indices = np.arange(0, ns + 1, obs_every)
        t_obs = params.t_grid[obs_indices]
        Z_obs = rng.standard_normal((n_paths, len(obs_indices)))
        L_obs = L[:, obs_indices] + params.sigma_obs * Z_obs

        # ── Realised total (Eq 7.2) ───────────────────────────────────
        # Y = mu_final + sigma_Y * Z
        Y = params.mu_final + params.sigma_Y * rng.standard_normal(n_paths)

        return {
            "t": params.t_grid,
            "q": q, "L": L,
            "L_obs": L_obs, "t_obs": t_obs,
            "Y": Y,
        }


# =====================================================================
# 4. MCStats — summary statistics (mirrors MonteCarloSummary)
# =====================================================================

@dataclass(frozen=True)
class MCStats:
    """Immutable Monte Carlo summary statistics.

    Mirrors GBM's MonteCarloSummary: immutable aggregation of terminal
    distribution properties.

    Error analysis uses L[:, -1] (TRUE process at kickoff), NOT L_obs.
    The closing line the bookmaker posts IS the true process; observation
    noise is the observer's measurement artifact.
    """
    label: str
    n_paths: int
    # Closing line L(0) statistics
    L_close_mean: float
    L_close_std: float
    L_close_p5: float
    L_close_p25: float
    L_close_median: float
    L_close_p75: float
    L_close_p95: float
    # Closing-line error |L(0) - mu_final|
    error_mean: float
    error_std: float
    error_median: float
    # Key probabilities
    p_within_05: float   # P(|error| < 0.5)
    p_within_10: float   # P(|error| < 1.0)
    p_within_15: float   # P(|error| < 1.5)
    # Realised total Y
    Y_mean: float
    Y_std: float
    # Signed error (for bias detection)
    signed_error_mean: float

    def report(self) -> str:
        """Formatted report string (mirrors MonteCarloSummary format)."""
        return (
            f"\n{'─'*50}\n"
            f"MC Report: {self.label}  (N={self.n_paths})\n"
            f"{'─'*50}\n"
            f"  Closing line L(0):\n"
            f"    mean={self.L_close_mean:.3f}, std={self.L_close_std:.3f}\n"
            f"    [p5={self.L_close_p5:.2f}, p25={self.L_close_p25:.2f}, "
            f"med={self.L_close_median:.2f}, p75={self.L_close_p75:.2f}, "
            f"p95={self.L_close_p95:.2f}]\n"
            f"  Closing error |L(0) - mu_final|:\n"
            f"    mean={self.error_mean:.4f}, std={self.error_std:.4f}, "
            f"median={self.error_median:.4f}\n"
            f"  Signed error L(0) - mu_final:\n"
            f"    mean={self.signed_error_mean:.4f} "
            f"({'biased high' if self.signed_error_mean > 0.01 else 'biased low' if self.signed_error_mean < -0.01 else 'unbiased'})\n"
            f"  Key probabilities:\n"
            f"    P(|err| < 0.5pt) = {self.p_within_05*100:.1f}%\n"
            f"    P(|err| < 1.0pt) = {self.p_within_10*100:.1f}%\n"
            f"    P(|err| < 1.5pt) = {self.p_within_15*100:.1f}%\n"
            f"  Realised total Y:\n"
            f"    mean={self.Y_mean:.2f}, std={self.Y_std:.2f}\n"
        )


# =====================================================================
# 5. MonteCarloEngine — runs batch + computes stats
# =====================================================================

class MonteCarloEngine:
    """Runs SDE simulation and computes MCStats.

    Mirrors GBM's summarize_terminal_distribution pattern:
    simulate → extract terminal values → aggregate into MCStats.
    """

    @staticmethod
    def run(params: ODEParams, n_paths: int = 1000,
            seed: int = 42) -> tuple[dict, MCStats]:
        """Run Monte Carlo and return (raw_data, stats)."""
        data = SDESimulator.simulate(params, n_paths=n_paths, seed=seed)

        # Terminal line values (TRUE process, NOT L_obs)
        L_close = data["L"][:, -1]
        mu_f = params.mu_final

        # Errors
        signed_errors = L_close - mu_f
        abs_errors = np.abs(signed_errors)

        stats = MCStats(
            label=params.label,
            n_paths=n_paths,
            L_close_mean=float(np.mean(L_close)),
            L_close_std=float(np.std(L_close)),
            L_close_p5=float(np.percentile(L_close, 5)),
            L_close_p25=float(np.percentile(L_close, 25)),
            L_close_median=float(np.median(L_close)),
            L_close_p75=float(np.percentile(L_close, 75)),
            L_close_p95=float(np.percentile(L_close, 95)),
            error_mean=float(np.mean(abs_errors)),
            error_std=float(np.std(abs_errors)),
            error_median=float(np.median(abs_errors)),
            p_within_05=float(np.mean(abs_errors < 0.5)),
            p_within_10=float(np.mean(abs_errors < 1.0)),
            p_within_15=float(np.mean(abs_errors < 1.5)),
            Y_mean=float(np.mean(data["Y"])),
            Y_std=float(np.std(data["Y"])),
            signed_error_mean=float(np.mean(signed_errors)),
        )

        return data, stats


# =====================================================================
# 6. Visualizer — all Phase 4 figures
# =====================================================================

class Visualizer:
    """Generates all required Phase 4 figures.

    7 figure types, all saved to phase4_figures/ at 200 dpi.
    """

    @staticmethod
    def fig1_cross_validation(params_list: list[ODEParams],
                              num_results: list[dict],
                              ana_results: list[np.ndarray | None]):
        """Figure 1: Deterministic ODE cross-validation (numerical vs analytical).

        2x2 grid: top row L(t), bottom row q(t) for baseline and extreme.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        for col, (p, nr, ar) in enumerate(zip(params_list[:2], num_results[:2], ana_results[:2])):
            # L(t)
            axes[0, col].plot(nr["t"], nr["L"], "b-", lw=2, label="Numerical (RK45)")
            if ar is not None:
                axes[0, col].plot(p.t_grid, ar, "r--", lw=1.5, label="Analytical")
                max_err = np.max(np.abs(nr["L"] - ar[:len(nr["L"])]))
                axes[0, col].set_title(f"{p.label}: L(t)\nmax|num-ana| = {max_err:.2e}")
            else:
                axes[0, col].set_title(f"{p.label}: L(t) [no analytic]")
            axes[0, col].axhline(p.mu_final, ls=":", color="gray", label=f"mu={p.mu_final}")
            axes[0, col].axhline(p.L_open, ls=":", color="lightgray")
            axes[0, col].set_ylabel("L (totals line)")
            axes[0, col].legend(fontsize=8)

            # q(t)
            axes[1, col].plot(nr["t"], nr["q"], "b-", lw=2, label="q(t)")
            axes[1, col].axhline(0, ls=":", color="gray")
            axes[1, col].set_xlabel("Hours before kickoff")
            axes[1, col].set_ylabel("q (exposure)")
            axes[1, col].set_title(f"{p.label}: q(t)")

        fig.suptitle("Figure 1: Deterministic Cross-Validation", fontsize=14)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "fig1_cross_validation.png", dpi=FIG_DPI, bbox_inches="tight")
        plt.close(fig)
        print("  Saved fig1_cross_validation.png")

    @staticmethod
    def fig2_single_path(params: ODEParams, data: dict, det: dict, idx: int = 0):
        """Figure 2: Single stochastic path overlaid on deterministic."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                                        gridspec_kw={"height_ratios": [2, 1]})
        color = C_BASE if "Baseline" in params.label else (C_EXTREME if "Extreme" in params.label else C_SHOCK)

        # L(t)
        ax1.plot(det["t"], det["L"], color="gray", lw=1.5, alpha=0.7, label="Deterministic ODE")
        ax1.plot(data["t"], data["L"][idx], color=color, lw=1, alpha=0.9, label="SDE path")
        ax1.scatter(data["t_obs"], data["L_obs"][idx], s=15, c="black", zorder=5,
                    alpha=0.6, label="L_obs (observed)")
        # mu(t)
        mu_arr = params.mu_array()
        ax1.plot(params.t_grid, mu_arr, "g:", lw=2, label=f"mu(t) [final={params.mu_final}]")
        ax1.set_ylabel("L (totals line)")
        ax1.set_title(f"Figure 2: Single Stochastic Path — {params.label}")
        ax1.legend(fontsize=8)

        # q(t)
        ax2.plot(data["t"], data["q"][idx], color=color, lw=1)
        ax2.axhline(0, ls=":", color="gray")
        ax2.set_xlabel("Hours before kickoff")
        ax2.set_ylabel("q (exposure)")

        fig.tight_layout()
        fig.savefig(FIG_DIR / f"fig2_path_{params.label.replace(' ', '_').lower()}.png",
                    dpi=FIG_DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved fig2_path_{params.label.replace(' ', '_').lower()}.png")

    @staticmethod
    def fig3_fan_chart(params_list: list[ODEParams], data_list: list[dict]):
        """Figure 3: MC fan chart (percentile bands + sample paths)."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        configs = [
            (params_list[0], data_list[0], C_BASE, C_BASE_FAN),
            (params_list[1], data_list[1], C_EXTREME, C_EXTREME_FAN),
        ]
        for ax, (p, d, c_main, c_fan) in zip(axes, configs):
            t = d["t"]
            L = d["L"]
            p5 = np.percentile(L, 5, axis=0)
            p25 = np.percentile(L, 25, axis=0)
            p50 = np.median(L, axis=0)
            p75 = np.percentile(L, 75, axis=0)
            p95 = np.percentile(L, 95, axis=0)

            ax.fill_between(t, p5, p95, color=c_fan, alpha=0.3, label="5th-95th")
            ax.fill_between(t, p25, p75, color=c_fan, alpha=0.5, label="25th-75th")
            ax.plot(t, p50, color=c_main, lw=2, label="Median")
            # 50 sample paths
            for i in range(min(50, L.shape[0])):
                ax.plot(t, L[i], color=c_main, alpha=0.05, lw=0.5)
            ax.axhline(p.mu_final, ls=":", color="gray", label=f"mu={p.mu_final}")
            ax.set_title(p.label)
            ax.set_xlabel("Hours before kickoff")
            ax.set_ylabel("L (totals line)")
            ax.legend(fontsize=8)

        fig.suptitle("Figure 3: Monte Carlo Fan Chart", fontsize=14)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "fig3_fan_chart.png", dpi=FIG_DPI, bbox_inches="tight")
        plt.close(fig)
        print("  Saved fig3_fan_chart.png")

    @staticmethod
    def fig4_closing_error(stats_list: list[MCStats], data_list: list[dict],
                           params_list: list[ODEParams]):
        """Figure 4: Closing-error distribution (histogram + CDF)."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        colors = [C_BASE, C_EXTREME, C_SHOCK]

        for i, (s, d, p, c) in enumerate(zip(stats_list, data_list, params_list, colors)):
            errors = np.abs(d["L"][:, -1] - p.mu_final)
            ax1.hist(errors, bins=50, alpha=0.5, color=c, label=s.label, density=True)
            sorted_e = np.sort(errors)
            cdf = np.arange(1, len(sorted_e) + 1) / len(sorted_e)
            ax2.plot(sorted_e, cdf, color=c, lw=2, label=s.label)

        for v in [0.5, 1.0, 1.5]:
            ax1.axvline(v, ls="--", color="gray", alpha=0.5)
            ax2.axvline(v, ls="--", color="gray", alpha=0.5)

        # Annotate CDF
        for i, (s, c) in enumerate(zip(stats_list, colors)):
            ax2.annotate(
                f"{s.label}:\n  P(<0.5)={s.p_within_05*100:.0f}%\n  P(<1.0)={s.p_within_10*100:.0f}%",
                xy=(0.97, 0.35 - i * 0.25), xycoords="axes fraction",
                ha="right", fontsize=8, color=c,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))

        ax1.set_xlabel("|L(0) - mu_final| (points)")
        ax1.set_ylabel("Density")
        ax1.set_title("Closing-Error Histogram")
        ax1.legend(fontsize=8)
        ax2.set_xlabel("|L(0) - mu_final| (points)")
        ax2.set_ylabel("CDF")
        ax2.set_title("Closing-Error CDF")
        ax2.legend(fontsize=8)

        fig.suptitle("Figure 4: Closing-Error Distribution", fontsize=14)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "fig4_closing_error.png", dpi=FIG_DPI, bbox_inches="tight")
        plt.close(fig)
        print("  Saved fig4_closing_error.png")

    @staticmethod
    def fig5_scatter(stats_list: list[MCStats], data_list: list[dict],
                     params_list: list[ODEParams]):
        """Figure 5: Realised total Y vs closing line L(0)."""
        fig, ax = plt.subplots(figsize=(9, 9))
        colors = [C_BASE, C_EXTREME, C_SHOCK]
        for d, p, c in zip(data_list, params_list, colors):
            L_close = d["L"][:, -1]
            Y = d["Y"]
            ax.scatter(L_close, Y, alpha=0.08, s=10, color=c, label=p.label)
        lims = ax.get_xlim()
        ax.plot(lims, lims, "k--", lw=1, alpha=0.5, label="Y = L(0)")
        ax.set_xlabel("Closing Line L(0)")
        ax.set_ylabel("Realised Total Y")
        ax.set_title("Figure 5: Y vs L(0)")
        ax.legend(fontsize=9)
        fig.savefig(FIG_DIR / "fig5_scatter.png", dpi=FIG_DPI, bbox_inches="tight")
        plt.close(fig)
        print("  Saved fig5_scatter.png")

    @staticmethod
    def fig6_trajectory_comparison(params_list: list[ODEParams], data_list: list[dict]):
        """Figure 6: All 3 scenarios overlaid (10 paths each)."""
        fig, ax = plt.subplots(figsize=(14, 7))
        colors = [C_BASE, C_EXTREME, C_SHOCK]
        for p, d, c in zip(params_list, data_list, colors):
            for i in range(10):
                ax.plot(d["t"], d["L"][i], color=c, alpha=0.3, lw=0.8,
                        label=p.label if i == 0 else "")
            # mu(t)
            mu_arr = p.mu_array()
            ax.plot(p.t_grid, mu_arr, color=c, ls=":", lw=2, alpha=0.8)

        ax.set_xlabel("Hours before kickoff")
        ax.set_ylabel("L (totals line)")
        ax.set_title("Figure 6: Trajectory Comparison (10 paths per scenario)")
        ax.legend(fontsize=9)
        fig.savefig(FIG_DIR / "fig6_trajectories.png", dpi=FIG_DPI, bbox_inches="tight")
        plt.close(fig)
        print("  Saved fig6_trajectories.png")

    @staticmethod
    def fig7_three_scenario_error(stats_list: list[MCStats], data_list: list[dict],
                                  params_list: list[ODEParams]):
        """Figure 7: THREE-SCENARIO closing-error comparison (KEY FIGURE).

        Left: overlaid histograms. Right: overlaid CDFs with annotations.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        colors = [C_BASE, C_EXTREME, C_SHOCK]

        for d, p, s, c in zip(data_list, params_list, stats_list, colors):
            errors = np.abs(d["L"][:, -1] - p.mu_final)
            ax1.hist(errors, bins=60, alpha=0.4, color=c, label=s.label, density=True)
            sorted_e = np.sort(errors)
            cdf = np.arange(1, len(sorted_e) + 1) / len(sorted_e)
            ax2.plot(sorted_e, cdf, color=c, lw=2.5, label=s.label)

        for v in [0.5, 1.0, 1.5]:
            ax1.axvline(v, ls="--", color="gray", alpha=0.4)
            ax2.axvline(v, ls="--", color="gray", alpha=0.4)

        # Annotation box
        box_text = "P(|err| < 0.5pt) / P(|err| < 1.0pt):\n"
        for s in stats_list:
            box_text += f"  {s.label}: {s.p_within_05*100:.0f}% / {s.p_within_10*100:.0f}%\n"
        ax2.annotate(box_text.strip(), xy=(0.97, 0.05), xycoords="axes fraction",
                     ha="right", va="bottom", fontsize=9,
                     bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.9))

        ax1.set_xlabel("|L(0) - mu_final|")
        ax1.set_ylabel("Density")
        ax1.set_title("Closing-Error Histograms")
        ax1.legend()
        ax2.set_xlabel("|L(0) - mu_final|")
        ax2.set_ylabel("CDF")
        ax2.set_title("Closing-Error CDFs")
        ax2.legend()

        fig.suptitle("Figure 7: Three-Scenario Closing-Error Comparison (KEY FIGURE)", fontsize=14)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "fig7_closing_error_3way.png", dpi=FIG_DPI, bbox_inches="tight")
        plt.close(fig)
        print("  Saved fig7_closing_error_3way.png")


# =====================================================================
# MAIN EXECUTION
# =====================================================================

def main():
    FIG_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("PHASE 4: SIMULATION MODULE")
    print("=" * 60)

    # ── 1. Create scenarios ───────────────────────────────────────────
    baseline = ODEParams(
        k=0.5, c=0.3, a=0.2, eta=0.4,
        mu_base=45.5, L_open=44.0, q_init=0.0,
        sigma_q=0.15, sigma_L=0.08, sigma_obs=0.30, sigma_Y=10.0,
        t_start=-48, t_end=0, n_steps=960,
        label="Baseline",
    )
    extreme = ODEParams(
        k=0.8, c=0.3, a=0.2, eta=0.15,
        mu_base=42.5, L_open=44.0, q_init=0.0,
        sigma_q=0.40, sigma_L=0.18, sigma_obs=0.40, sigma_Y=10.0,
        t_start=-48, t_end=0, n_steps=960,
        label="Extreme Wind",
    )
    late_shock = ODEParams(
        k=0.8, c=0.3, a=0.2, eta=0.15,
        mu_base=45.5, L_open=44.0, q_init=0.0,
        sigma_q=0.40, sigma_L=0.18, sigma_obs=0.40, sigma_Y=10.0,
        t_start=-48, t_end=0, n_steps=960,
        label="Late Shock",
        mu_schedule=((-8.0, 42.5),),
    )
    scenarios = [baseline, extreme, late_shock]

    # ── 2. Print diagnostics ──────────────────────────────────────────
    for p in scenarios:
        print(f"\n{p.summary()}")

    # ── 3. Cross-validate: numerical vs analytical ────────────────────
    print("\n" + "=" * 60)
    print("CROSS-VALIDATION: Numerical vs Analytical")
    print("=" * 60)

    num_results = []
    ana_results = []
    for p in scenarios:
        nr = ODESolver.solve_system(p)
        num_results.append(nr)
        assert nr["success"], f"ODE solve failed for {p.label}"

        if not p.mu_schedule:
            ar = ODESolver.solve_analytical(p)
            max_err = np.max(np.abs(nr["L"] - ar[:len(nr["L"])]))
            status = "PASS" if max_err < 1e-4 else "FAIL"
            print(f"  {p.label}: max|L_num - L_ana| = {max_err:.2e} → {status}")
            ana_results.append(ar)
        else:
            print(f"  {p.label}: SKIP analytical (time-varying mu)")
            ana_results.append(None)

    # ── 4. Monte Carlo ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("MONTE CARLO (1000 paths each)")
    print("=" * 60)

    mc_data = []
    mc_stats = []
    for p, seed in zip(scenarios, [42, 43, 44]):
        d, s = MonteCarloEngine.run(p, n_paths=1000, seed=seed)
        mc_data.append(d)
        mc_stats.append(s)

    # ── 5. Print reports ──────────────────────────────────────────────
    for s in mc_stats:
        print(s.report())

    # ── 6. Generate figures ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("GENERATING FIGURES")
    print("=" * 60)

    Visualizer.fig1_cross_validation(scenarios[:2], num_results[:2], ana_results[:2])

    for p, d, nr in zip(scenarios, mc_data, num_results):
        Visualizer.fig2_single_path(p, d, nr)

    Visualizer.fig3_fan_chart(scenarios[:2], mc_data[:2])
    Visualizer.fig4_closing_error(mc_stats, mc_data, scenarios)
    Visualizer.fig5_scatter(mc_stats, mc_data, scenarios)
    Visualizer.fig6_trajectory_comparison(scenarios, mc_data)
    Visualizer.fig7_three_scenario_error(mc_stats, mc_data, scenarios)

    # ── 7. Summary ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 4 COMPLETE")
    print("=" * 60)
    print(f"Figures saved to {FIG_DIR.resolve()}/")
    print(f"  Baseline P(<0.5pt): {mc_stats[0].p_within_05*100:.0f}%  (expect ~100%)")
    print(f"  Extreme  P(<0.5pt): {mc_stats[1].p_within_05*100:.0f}%  (expect ~91%)")
    print(f"  LateShock P(<0.5pt): {mc_stats[2].p_within_05*100:.0f}%  (expect ~50%)")


if __name__ == "__main__":
    main()
