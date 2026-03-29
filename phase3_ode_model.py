"""
Phase 3 — ODE Model: Theory, Calibration, and Simulation
==========================================================

KEY DESIGN DECISION: COMPOSITE PARAMETERS ONLY
-----------------------------------------------
The full 2-D ODE has 4 parameters (k, c, a, eta), but q(t) is latent.
Synthetic validation proved individual parameters are unidentifiable
from L(t) data alone (60–900% relative errors, even with clean data).

The second-order reduction shows L(t) depends only on:
    gamma    = c + eta     (damping)
    omega_sq = c*eta + a*k (stiffness)

We fit these 2 composites + v0 (initial velocity) = 3 parameters total,
using the ANALYTIC CLOSED-FORM solution — no ODE solver, no solve_ivp.
Each fit takes milliseconds.

ANALYTIC SOLUTION
-----------------
The second-order ODE: L'' + gamma*L' + omega_sq*(L - mu) = 0

Eigenvalues: lam = [-gamma ± sqrt(gamma² - 4*omega_sq)] / 2

Overdamped  (gamma² > 4*omega_sq): L(t) = mu + A*exp(lam1*t) + B*exp(lam2*t)
Underdamped (gamma² < 4*omega_sq): L(t) = mu + exp(alpha*t) * [C1*cos(wt) + C2*sin(wt)]
Critical    (gamma² = 4*omega_sq): L(t) = mu + (C1 + C2*t) * exp(alpha*t)

where A, B, C1, C2 are determined by initial conditions L(t0) and L'(t0).

EXECUTION ORDER
---------------
1. Verify analytic solution matches numerical solver
2. Synthetic validation (prove composite recovery works)
3. Per-game calibration on synthetic data + baseline comparison
4. Extreme vs normal comparison
5. Simulation scenarios + Monte Carlo
6. Exercises, diagnostics, limitations
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.ode_model import (
    ode_system, eigenvalues, composite_params,
    solve_ode, analytic_solution,
)
from src.calibration import calibrate_game, analytic_L, second_order_ols
from src.baselines import baseline_linear, baseline_exp_smooth, baseline_random_walk
from src.diagnostics import diagnose_residuals
from src.simulation import simulate_line_path, monte_carlo_paths

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 11, "axes.titlesize": 13, "axes.labelsize": 12,
})

FIG_DIR = Path("data_file/phase3")

# ── Ground-truth parameters (for theory and simulation) ──────────────
K_TRUE, C_TRUE, A_TRUE, ETA_TRUE = 0.05, 0.3, 0.2, 0.4
MU_TRUE = 45.5
L_OPEN_TRUE = 44.0
Q0 = 0.0
T_START = -48.0
T_END = 0.0

# Derived composite ground truth
GAMMA_TRUE = C_TRUE + ETA_TRUE                     # 0.7
OMEGA_SQ_TRUE = C_TRUE * ETA_TRUE + A_TRUE * K_TRUE  # 0.13
V0_TRUE = ETA_TRUE * (MU_TRUE - L_OPEN_TRUE)       # 0.6  (from dL/dt at t0 with q0=0)


# =====================================================================
# STEP 1: VERIFY ANALYTIC SOLUTION
# =====================================================================

def step1_verify_analytic():
    """Prove the analytic closed-form matches the numerical ODE solver.

    This is a necessary pre-check: if analytic != numerical, everything
    downstream is wrong. We compare at 200 time points and require
    agreement to < 1e-6 points.
    """
    print("=" * 70)
    print("STEP 1: VERIFY ANALYTIC SOLUTION")
    print("=" * 70)

    t_test = np.linspace(T_START, T_END, 200)

    # Numerical: full 2-D system via solve_ivp
    sol_num = solve_ode(K_TRUE, C_TRUE, A_TRUE, ETA_TRUE, MU_TRUE,
                        L_OPEN_TRUE, Q0, T_START, T_END, t_eval=t_test)

    # Analytic via composite parameters
    d0 = L_OPEN_TRUE - MU_TRUE
    L_an = analytic_L(t_test, T_START, MU_TRUE, d0,
                      GAMMA_TRUE, OMEGA_SQ_TRUE, V0_TRUE)

    max_diff = np.max(np.abs(sol_num["L"] - L_an))
    print(f"Max |L_numerical - L_analytic| = {max_diff:.2e}")
    assert max_diff < 1e-5, f"FAIL: analytic/numerical mismatch = {max_diff}"
    print("PASS: analytic closed-form matches solve_ivp.\n")

    # Also verify the analytic_solution function in ode_model.py
    L_an2 = analytic_solution(K_TRUE, C_TRUE, A_TRUE, ETA_TRUE,
                               MU_TRUE, L_OPEN_TRUE, Q0, t_test, T_START)
    max_diff2 = np.max(np.abs(sol_num["L"] - L_an2))
    print(f"Max |L_numerical - L_analytic_full| = {max_diff2:.2e}")
    assert max_diff2 < 1e-5, f"FAIL: full analytic mismatch = {max_diff2}"
    print("PASS: both analytic forms agree with numerical solver.\n")

    # Print the regime
    disc = GAMMA_TRUE**2 - 4 * OMEGA_SQ_TRUE
    regime = "overdamped" if disc > 0 else ("underdamped" if disc < 0 else "critical")
    print(f"True composites: gamma={GAMMA_TRUE}, omega_sq={OMEGA_SQ_TRUE}")
    print(f"Discriminant = {disc:.4f} => {regime}")
    print(f"Initial velocity v0 = eta*(mu - L0) = {V0_TRUE}")


# =====================================================================
# STEP 2: SYNTHETIC VALIDATION (composite recovery)
# =====================================================================

def step2_synthetic_validation():
    """Prove the 3-parameter composite fit recovers known parameters.

    Unlike the 4-parameter fit which failed catastrophically (60-900%
    errors), the composite fit should recover gamma and omega_sq with
    much better accuracy because:
    1. Only 3 parameters instead of 4
    2. gamma and omega_sq DIRECTLY control L(t) — they are the
       observable parameters, not derived from unobservable ones
    3. The analytic solution is exact (no numerical integration error)
    """
    print("\n" + "=" * 70)
    print("STEP 2: SYNTHETIC VALIDATION (composite parameters)")
    print("=" * 70)

    configs = [
        (5, 0.1), (5, 0.5),
        (10, 0.1), (10, 0.5),
        (20, 0.1),
        (50, 0.1),
    ]

    n_trials = 25  # 25 trials per config — enough for median/IQR, runs in ~60s
    rng = np.random.default_rng(42)

    # Generate true L(t) from the analytic solution (no solver needed)
    d0 = L_OPEN_TRUE - MU_TRUE

    all_results = []

    print(f"True: gamma={GAMMA_TRUE:.3f}, omega_sq={OMEGA_SQ_TRUE:.4f}, v0={V0_TRUE:.3f}")
    print(f"{'n':>4s} {'sigma':>6s} {'gamma_err%':>11s} {'omgsq_err%':>11s} {'v0_err%':>11s} {'rmse':>8s}")
    print("─" * 55)

    for n_snap, sigma in configs:
        gamma_errs, omgsq_errs, v0_errs, rmses = [], [], [], []

        for trial in range(n_trials):
            t_snap = np.linspace(T_START, T_END, n_snap)
            L_true = analytic_L(t_snap, T_START, MU_TRUE, d0,
                                GAMMA_TRUE, OMEGA_SQ_TRUE, V0_TRUE)
            noise = sigma * rng.standard_normal(n_snap)
            L_obs = L_true + noise

            cal = calibrate_game(t_snap, L_obs, MU_TRUE, seed=trial)

            if cal["success"]:
                gamma_errs.append(abs(cal["gamma"] - GAMMA_TRUE) / GAMMA_TRUE * 100)
                omgsq_errs.append(abs(cal["omega_sq"] - OMEGA_SQ_TRUE) / OMEGA_SQ_TRUE * 100)
                v0_errs.append(abs(cal["v0"] - V0_TRUE) / abs(V0_TRUE) * 100 if V0_TRUE != 0 else 0)
                rmses.append(cal["rmse"])

        med_g = np.median(gamma_errs) if gamma_errs else np.nan
        med_o = np.median(omgsq_errs) if omgsq_errs else np.nan
        med_v = np.median(v0_errs) if v0_errs else np.nan
        med_r = np.median(rmses) if rmses else np.nan

        print(f"{n_snap:4d} {sigma:6.2f} {med_g:10.1f}% {med_o:10.1f}% {med_v:10.1f}% {med_r:8.4f}")
        all_results.append({
            "n": n_snap, "sigma": sigma,
            "gamma_err": gamma_errs, "omgsq_err": omgsq_errs,
            "v0_err": v0_errs, "rmse": rmses,
        })

    # ── Figure: Recovery accuracy ─────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    param_data = [
        ("gamma (damping)", "gamma_err"),
        ("omega_sq (stiffness)", "omgsq_err"),
        ("v0 (initial velocity)", "v0_err"),
    ]
    for ax, (label, key) in zip(axes, param_data):
        for n_snap in [5, 10, 20, 50]:
            sigmas, medians = [], []
            for r in all_results:
                if r["n"] == n_snap and r[key]:
                    sigmas.append(r["sigma"])
                    medians.append(np.median(r[key]))
            if sigmas:
                ax.plot(sigmas, medians, "o-", label=f"n={n_snap}", lw=2)
        ax.set_xlabel("Noise sigma (points)")
        ax.set_ylabel("Median relative error (%)")
        ax.set_title(label)
        ax.legend()
        ax.axhline(20, color="red", ls="--", alpha=0.3)
        ax.set_ylim(bottom=0)

    fig.suptitle("Synthetic Validation: Composite Parameter Recovery", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_synth_recovery.png")
    plt.close(fig)
    print("\n  Saved fig_synth_recovery.png")

    return all_results


# =====================================================================
# STEP 3: PER-GAME CALIBRATION ON SYNTHETIC DATA
# =====================================================================

def step3_calibration_demo():
    """Demonstrate the calibration pipeline on 20 synthetic games.

    10 normal + 10 extreme. Each game has different mu, L_open, and
    slight parameter variation. This demonstrates:
    - The fit works across diverse conditions
    - Comparison to baselines (linear, exp smooth, random walk)
    - Residual diagnostics
    """
    print("\n" + "=" * 70)
    print("STEP 3: PER-GAME CALIBRATION (synthetic data)")
    print("=" * 70)

    rng = np.random.default_rng(123)
    games = []

    for i in range(20):
        is_extreme = i >= 10

        # Game-specific true composites with variation
        gamma_g = 0.7 + rng.normal(0, 0.1) + (-0.2 if is_extreme else 0)
        omega_sq_g = 0.13 + rng.normal(0, 0.02) + (0.05 if is_extreme else 0)
        gamma_g = max(gamma_g, 0.1)
        omega_sq_g = max(omega_sq_g, 0.02)

        mu_g = 45.5 + rng.normal(0, 2) + (-3.0 if is_extreme else 0)
        L_open_g = mu_g + rng.uniform(-2, 0.5)
        d0_g = L_open_g - mu_g
        v0_g = 0.4 * (mu_g - L_open_g) + rng.normal(0, 0.1)  # eta*(mu-L0) + noise

        n_snap = rng.integers(8, 18)
        sigma_g = 0.15 + (0.1 if is_extreme else 0)

        t_snap = np.linspace(T_START, T_END, n_snap)
        L_true = analytic_L(t_snap, T_START, mu_g, d0_g, gamma_g, omega_sq_g, v0_g)
        L_obs = L_true + sigma_g * rng.standard_normal(n_snap)

        games.append({
            "game_id": f"SYN_{i:03d}",
            "t_obs": t_snap, "L_obs": L_obs, "mu": mu_g,
            "extreme_wind": 1 if is_extreme else 0,
            "true_gamma": gamma_g, "true_omega_sq": omega_sq_g,
            "true_v0": v0_g, "L_true": L_true,
        })

    # Calibrate each game
    cal_results = []
    for g in games:
        cal = calibrate_game(g["t_obs"], g["L_obs"], g["mu"])
        bl_lin = baseline_linear(g["t_obs"], g["L_obs"])
        bl_exp = baseline_exp_smooth(g["t_obs"], g["L_obs"])
        bl_rw = baseline_random_walk(g["t_obs"], g["L_obs"])
        diag = diagnose_residuals(cal["residuals"], g["t_obs"]) if cal["success"] else {}

        cal.update({
            "game_id": g["game_id"],
            "extreme_wind": g["extreme_wind"],
            "rmse_linear": bl_lin["rmse"],
            "rmse_exp": bl_exp["rmse"],
            "rmse_rw": bl_rw["rmse"],
            "beats_linear": cal["rmse"] < bl_lin["rmse"],
            "true_gamma": g["true_gamma"],
            "true_omega_sq": g["true_omega_sq"],
            "diagnostics": diag,
        })
        cal_results.append(cal)

        tag = "EX" if g["extreme_wind"] else "NM"
        bl = "BETTER" if cal["beats_linear"] else "worse"
        print(f"  {g['game_id']} [{tag}] n={cal['n_snapshots']:2d}: "
              f"RMSE={cal['rmse']:.4f} (lin={bl_lin['rmse']:.4f} {bl}) "
              f"gamma={cal['gamma']:.3f} regime={cal['regime']}")

    # ── Summary ───────────────────────────────────────────────────────
    df_cal = pd.DataFrame([{
        "game_id": r["game_id"], "extreme": r["extreme_wind"],
        "gamma": r["gamma"], "omega_sq": r["omega_sq"],
        "v0": r["v0"], "regime": r["regime"],
        "rmse_ode": r["rmse"], "rmse_lin": r["rmse_linear"],
        "rmse_exp": r["rmse_exp"], "rmse_rw": r["rmse_rw"],
        "beats_lin": r["beats_linear"],
        "true_gamma": r["true_gamma"], "true_omega_sq": r["true_omega_sq"],
    } for r in cal_results])

    n_beats = df_cal["beats_lin"].sum()
    print(f"\nODE beats linear: {n_beats}/{len(df_cal)} ({100*n_beats/len(df_cal):.0f}%)")

    # Parameter recovery accuracy
    df_cal["gamma_err%"] = (df_cal["gamma"] - df_cal["true_gamma"]).abs() / df_cal["true_gamma"] * 100
    df_cal["omgsq_err%"] = (df_cal["omega_sq"] - df_cal["true_omega_sq"]).abs() / df_cal["true_omega_sq"] * 100
    print(f"Gamma recovery: median error = {df_cal['gamma_err%'].median():.1f}%")
    print(f"Omega_sq recovery: median error = {df_cal['omgsq_err%'].median():.1f}%")

    # ── Figure 1: Fitted L(t) for 6 games ────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for idx, gi in enumerate([0, 3, 5, 10, 13, 17]):
        ax = axes.flat[idx]
        g = games[gi]
        r = cal_results[gi]

        t_dense = np.linspace(g["t_obs"][0], g["t_obs"][-1], 200)
        d0 = g["L_obs"][0] - g["mu"]
        L_fit_dense = analytic_L(t_dense, g["t_obs"][0], g["mu"],
                                 d0, r["gamma"], r["omega_sq"], r["v0"])

        ax.scatter(g["t_obs"], g["L_obs"], c="black", s=30, zorder=5, label="Observed")
        ax.plot(g["t_obs"], g["L_true"], "g--", alpha=0.5, lw=1, label="True (no noise)")
        ax.plot(t_dense, L_fit_dense, "r-", lw=2, label="ODE fit")
        bl = baseline_linear(g["t_obs"], g["L_obs"])
        ax.plot(g["t_obs"], bl["L_pred"], "b:", lw=1.5, label="Linear")
        ax.axhline(g["mu"], color="gray", ls="--", alpha=0.5)

        tag = "EXTREME" if g["extreme_wind"] else "Normal"
        ax.set_title(f"{g['game_id']} ({tag})\nRMSE: ODE={r['rmse']:.3f} Lin={r['rmse_linear']:.3f}")
        ax.set_xlabel("Hours before kickoff")
        ax.set_ylabel("Totals Line")
        if idx == 0:
            ax.legend(fontsize=7)

    fig.suptitle("ODE Fits: Observed vs Predicted L(t)", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_ode_fits.png")
    plt.close(fig)
    print("  Saved fig_ode_fits.png")

    # ── Figure 4: RMSE comparison ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    methods = ["ODE (analytic)", "Linear Interp", "Exp Smooth", "Random Walk"]
    rmse_cols = ["rmse_ode", "rmse_lin", "rmse_exp", "rmse_rw"]
    means = [df_cal[c].mean() for c in rmse_cols]
    stds = [df_cal[c].std() for c in rmse_cols]
    colors = ["steelblue", "gray", "coral", "goldenrod"]
    bars = ax.bar(methods, means, yerr=stds, capsize=5, color=colors, alpha=0.85)
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{val:.4f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylabel("Mean RMSE (points)")
    ax.set_title("Model Comparison: ODE vs Baselines")
    fig.savefig(FIG_DIR / "fig_rmse_comparison.png")
    plt.close(fig)
    print("  Saved fig_rmse_comparison.png")

    return cal_results, games, df_cal


# =====================================================================
# STEP 4: EXTREME vs NORMAL
# =====================================================================

def step4_extreme_analysis(df_cal):
    """Compare composite parameters between normal and extreme games.

    Uses Wilcoxon rank-sum (appropriate for n=10 per group).
    Reports effect sizes, not just p-values.
    """
    print("\n" + "=" * 70)
    print("STEP 4: EXTREME vs NORMAL COMPARISON")
    print("=" * 70)

    normal = df_cal[df_cal["extreme"] == 0]
    extreme = df_cal[df_cal["extreme"] == 1]
    print(f"Normal: n={len(normal)}, Extreme: n={len(extreme)}")
    print("NOTE: With n=10, power is limited. Effect sizes > p-values.\n")

    from scipy.stats import mannwhitneyu
    for param in ["gamma", "omega_sq"]:
        n_v, e_v = normal[param].values, extreme[param].values
        stat, p = mannwhitneyu(n_v, e_v, alternative="two-sided")
        r_rb = 1 - 2 * stat / (len(n_v) * len(e_v))
        print(f"  {param:12s}: Normal med={np.median(n_v):.4f}, "
              f"Extreme med={np.median(e_v):.4f}, p={p:.4f}, r_rb={r_rb:.3f}")

    # ── Figure 3: Boxplots ────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, param, label in zip(axes, ["gamma", "omega_sq"],
                                 ["gamma (damping)", "omega_sq (stiffness)"]):
        data = [df_cal[df_cal["extreme"] == 0][param].values,
                df_cal[df_cal["extreme"] == 1][param].values]
        ax.boxplot(data, labels=["Normal", "Extreme"], showmeans=True,
                   meanprops=dict(marker="D", markerfacecolor="red", markersize=8))
        ax.set_title(label)
        ax.set_ylabel(param)
    fig.suptitle("Composite Parameters: Normal vs Extreme Weather", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_extreme_boxplots.png")
    plt.close(fig)
    print("  Saved fig_extreme_boxplots.png")


# =====================================================================
# STEP 5: SIMULATION MODULE
# =====================================================================

def step5_simulation():
    """Simulation scenarios + Monte Carlo."""
    print("\n" + "=" * 70)
    print("STEP 5: SIMULATION")
    print("=" * 70)

    normal = simulate_line_path(0.05, 0.3, 0.2, 0.4, 45.5, 44.0, 0.0,
                                -48, 0, 15, 0.2, seed=42)
    extreme_out = simulate_line_path(0.05, 0.3, 0.2, 0.4, 42.0, 44.0, 0.0,
                                     -48, 0, 15, 0.2, seed=42)
    extreme_dyn = simulate_line_path(0.08, 0.2, 0.3, 0.25, 42.0, 44.0, 0.0,
                                     -48, 0, 15, 0.3, seed=42)

    # ── Figure 5: Line paths ─────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (title, sim) in zip(axes, [
        ("Normal", normal),
        ("Extreme (outcome only)", extreme_out),
        ("Extreme (dynamics + outcome)", extreme_dyn),
    ]):
        ax.plot(sim["t_dense"], sim["L_dense"], "b-", lw=2, label="True L(t)")
        ax.scatter(sim["t_snap"], sim["L_obs"], c="red", s=25, zorder=5, label="Noisy obs")
        ax.axhline(sim["mu"], color="green", ls="--", alpha=0.7, label=f"mu={sim['mu']}")
        ax.set_title(title); ax.set_xlabel("Hours before kickoff"); ax.set_ylabel("Line")
        ax.legend(fontsize=8)
    fig.suptitle("Simulated Line Paths", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_sim_paths.png"); plt.close(fig)
    print("  Saved fig_sim_paths.png")

    # ── Figure: Phase portraits ───────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (title, sim) in zip(axes, [
        ("Normal", normal), ("Extreme outcome", extreme_out), ("Extreme dynamics", extreme_dyn),
    ]):
        ax.plot(sim["q_dense"], sim["L_dense"], "b-", lw=1.5)
        ax.plot(sim["q_dense"][0], sim["L_dense"][0], "go", ms=8, label="Start")
        ax.plot(sim["q_dense"][-1], sim["L_dense"][-1], "rs", ms=8, label="End")
        ax.plot(0, sim["mu"], "k*", ms=12, label="Equilibrium")
        ax.set_title(title); ax.set_xlabel("q (exposure)"); ax.set_ylabel("L (line)")
        ax.legend(fontsize=8)
    fig.suptitle("Phase Portraits: (q, L) Trajectories", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_phase_portraits.png"); plt.close(fig)
    print("  Saved fig_phase_portraits.png")

    # ── Figure: Parameter sensitivity ─────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    base = {"k": 0.05, "c": 0.3, "a": 0.2, "eta": 0.4}
    for ax, (pname, vals) in zip(axes.flat, {
        "k": [0.01, 0.05, 0.15, 0.30],
        "c": [0.1, 0.3, 0.6, 1.2],
        "a": [0.05, 0.2, 0.5, 1.0],
        "eta": [0.1, 0.4, 0.8, 1.5],
    }.items()):
        for v in vals:
            p = base.copy(); p[pname] = v
            sol = solve_ode(p["k"], p["c"], p["a"], p["eta"],
                            MU_TRUE, L_OPEN_TRUE, Q0, T_START, T_END)
            ax.plot(sol["t"], sol["L"], label=f"{pname}={v}")
        ax.axhline(MU_TRUE, color="gray", ls="--", alpha=0.5)
        ax.set_title(f"Varying {pname}")
        ax.set_xlabel("Hours before kickoff"); ax.set_ylabel("L(t)")
        ax.legend(fontsize=8)
    fig.suptitle("Parameter Sensitivity", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_param_sensitivity.png"); plt.close(fig)
    print("  Saved fig_param_sensitivity.png")

    # ── Monte Carlo ───────────────────────────────────────────────────
    print("\n── Monte Carlo (5000 sims each) ──")
    mc_n = monte_carlo_paths(0.05, 0.3, 0.2, 0.4, 45.5, 44.0, 0.0, -48, 0,
                             0.2, 13.9, 5000, seed=0)
    mc_e = monte_carlo_paths(0.08, 0.2, 0.3, 0.25, 42.0, 44.0, 0.0, -48, 0,
                             0.3, 13.9, 5000, seed=0)
    for label, mc in [("Normal", mc_n), ("Extreme", mc_e)]:
        print(f"  {label}: L_close_true={mc['L_close_true']:.3f}, "
              f"within 1pt={mc['pct_within_1pt']*100:.1f}%, "
              f"over rate={mc['over_hit_rate']*100:.1f}%")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(mc_n["closing_errors"], bins=50, alpha=0.7, color="steelblue",
                 label="Normal", density=True)
    axes[0].hist(mc_e["closing_errors"], bins=50, alpha=0.5, color="coral",
                 label="Extreme", density=True)
    axes[0].axvline(0, color="black", ls="--")
    axes[0].set_title("Closing Error (L_close - mu)"); axes[0].legend()

    axes[1].hist(mc_n["Y_sims"] - mc_n["L_close_obs"], bins=50, alpha=0.7,
                 color="steelblue", label="Normal", density=True)
    axes[1].hist(mc_e["Y_sims"] - mc_e["L_close_obs"], bins=50, alpha=0.5,
                 color="coral", label="Extreme", density=True)
    axes[1].axvline(0, color="black", ls="--")
    axes[1].set_title("Realised Error (Y - L_close)"); axes[1].legend()

    fig.suptitle("Monte Carlo: 5000 Simulations", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_monte_carlo.png"); plt.close(fig)
    print("  Saved fig_monte_carlo.png")


# =====================================================================
# STEP 6: EXERCISES + DIAGNOSTICS + LIMITATIONS
# =====================================================================

def step6_appendix(cal_results, games):
    """Print exercises, residual diagnostics, and limitations."""
    print("\n" + "=" * 70)
    print("EXERCISES (Appendix)")
    print("=" * 70)
    print("""
Ex 1: Derive L'' + (c+eta)L' + (c*eta+a*k)(L-mu) = 0
  From dL/dt = a*q + eta*(mu-L), get q = [L' + eta*L - eta*mu]/a.
  Differentiate: dq/dt = [L'' + eta*L']/a.
  Substitute into dq/dt = k(mu-L) - c*q:
    [L'' + eta*L']/a = k(mu-L) - c*[L' + eta*(L-mu)]/a
    L'' + eta*L' = a*k*(mu-L) - c*L' - c*eta*(L-mu)
    L'' + (c+eta)*L' + (c*eta+a*k)*(L-mu) = 0  QED

Ex 2: Eigenvalues: lam = [-(c+eta) +/- sqrt((c-eta)^2 - 4ak)] / 2
  Oscillation when (c-eta)^2 < 4ak.

Ex 3: Increasing eta increases gamma (damping) and omega_sq (stiffness).
  Line converges to mu faster. Can shift underdamped -> overdamped.

Ex 4: eta = eta0 - eta1*E_W. Extreme wind slows info-tracking.
  Market relies more on betting flow (the a*q channel).

Ex 5: q(t) is unobservable because public data shows handle ($$),
  not signed net position. Only a*k is identifiable from L(t).
""")

    # Residual diagnostics figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for idx in range(min(6, len(cal_results))):
        ax = axes.flat[idx]
        r = cal_results[idx]
        resid = r["residuals"]
        ax.stem(range(len(resid)), resid, linefmt="b-", markerfmt="bo", basefmt="k-")
        ax.axhline(0, color="red", ls="--")
        d = r.get("diagnostics", {})
        r1 = d.get("lag1_autocorr", None)
        title = r["game_id"]
        if r1 is not None:
            title += f"\nr1={r1:.3f}"
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Snapshot"); ax.set_ylabel("Residual")
    fig.suptitle("ODE Fit Residuals", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_residual_diagnostics.png"); plt.close(fig)
    print("  Saved fig_residual_diagnostics.png")

    # Limitations
    print("\n" + "=" * 70)
    print("KNOWN LIMITATIONS")
    print("=" * 70)
    print("""
1. q(t) is latent. Only gamma and omega_sq are identifiable from L(t).
2. mu = L_close is partly circular (fitting path to known endpoint).
3. Small sample (n=10/group) limits extreme-vs-normal comparison power.
4. Constant mu ignores updating weather forecasts.
5. All calibration uses synthetic data (no real line snapshots available).
6. Continuous model applied to a discrete (0.5-pt increment) market.
""")


# =====================================================================
# MAIN
# =====================================================================

def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("PHASE 3: ODE MODEL — COMPOSITE PARAMETER APPROACH")
    print("=" * 70)
    print()

    step1_verify_analytic()
    step2_synthetic_validation()
    cal_results, games, df_cal = step3_calibration_demo()
    step4_extreme_analysis(df_cal)
    step5_simulation()
    step6_appendix(cal_results, games)

    print("\n" + "=" * 70)
    print("Phase 3 complete. All figures saved as PNG.")
    print("=" * 70)


if __name__ == "__main__":
    main()
