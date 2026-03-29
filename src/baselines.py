"""
Baseline models for comparison against the ODE.
=================================================

The ODE must beat these to justify its complexity. If it doesn't beat
linear interpolation on > 60% of games, the model structure isn't
adding much beyond connecting two known endpoints.

Three baselines:

1. Linear interpolation: straight line from L(t_start) to L(t_end).
   This is the simplest possible "model" — just connect the dots.
   If the ODE can't beat this, it means the intermediate dynamics
   the ODE captures are either wrong or drowned out by noise.

2. Exponential smoothing toward L_close: the line asymptotically
   approaches the final value. This captures the empirical fact
   that most line movement occurs early, with convergence later.

3. Random walk: L(t+1) = L(t) + noise. The one-step-ahead RMSE
   measures how "unpredictable" line movements are. If the ODE
   doesn't beat a random walk, the dynamics are purely stochastic.
"""

import numpy as np


def baseline_linear(t_obs: np.ndarray, L_obs: np.ndarray) -> dict:
    """Linear interpolation from first to last observation.

    L_pred(t) = L_obs[0] + (L_obs[-1] - L_obs[0]) * (t - t[0]) / (t[-1] - t[0])

    This uses ONLY the endpoints. Any model that fits intermediate
    points should beat this easily. If it doesn't, the intermediate
    data doesn't contain learnable structure.
    """
    L_pred = np.interp(t_obs, [t_obs[0], t_obs[-1]], [L_obs[0], L_obs[-1]])
    residuals = L_obs - L_pred
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    return {"rmse": rmse, "L_pred": L_pred, "method": "linear_interp"}


def baseline_exp_smooth(t_obs: np.ndarray, L_obs: np.ndarray,
                        alpha: float = 3.0) -> dict:
    """Exponential smoothing toward the final observed value.

    L_pred(t) = L_obs[0] + (L_obs[-1] - L_obs[0]) * (1 - exp(-alpha * tau))

    where tau = (t - t[0]) / (t[-1] - t[0]) in [0, 1].
    alpha controls how fast the line converges: larger alpha = faster.
    At tau=1: L_pred = L_obs[-1] exactly (by construction).

    The choice alpha=3.0 means about 95% of the move happens by tau=1.
    """
    tau = (t_obs - t_obs[0]) / (t_obs[-1] - t_obs[0])
    # Avoid division by zero at tau=1
    L_target = L_obs[-1]
    L_pred = L_obs[0] + (L_target - L_obs[0]) * (1 - np.exp(-alpha * tau))
    # Force exact match at endpoint
    L_pred[-1] = L_target
    residuals = L_obs - L_pred
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    return {"rmse": rmse, "L_pred": L_pred, "method": "exp_smooth"}


def baseline_random_walk(t_obs: np.ndarray, L_obs: np.ndarray) -> dict:
    """Random walk baseline: previous value predicts next value.

    One-step-ahead error: e_i = L(t_{i+1}) - L(t_i)
    RMSE = sqrt(mean(e_i^2))

    This measures the inherent "step size" of line movements.
    An ODE that can't predict the DIRECTION of movement any better
    than assuming no change is not capturing useful dynamics.

    Note: this has N-1 errors (not N), since it's pairwise.
    """
    diffs = np.diff(L_obs)
    rmse = float(np.sqrt(np.mean(diffs ** 2)))
    # Naive prediction: L_pred[i] = L_obs[i-1]
    L_pred = np.empty_like(L_obs)
    L_pred[0] = L_obs[0]
    L_pred[1:] = L_obs[:-1]
    return {"rmse": rmse, "L_pred": L_pred, "method": "random_walk"}
