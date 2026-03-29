"""
ODE Model: 2-D coupled system for NFL totals line dynamics.
=============================================================

MATHEMATICAL FOUNDATION
-----------------------
State variables:
    q(t) = bookmaker net exposure to Over at time t  (latent, unobservable)
    L(t) = totals line (Over/Under) at time t        (observable)

Fundamental value:
    mu = model-implied expected total points (constant per game)

The coupled system (Equations 5.2-5.3):

    dq/dt = k(mu - L) - c*q          ... (1) exposure dynamics
    dL/dt = a*q + eta*(mu - L)        ... (2) line dynamics

PARAMETER MEANINGS (with units, assuming t in hours, L in points):
    k > 0 [exposure * hr^-1 * pt^-1]:
        Betting response rate. How aggressively bettors react to a
        perceived mispricing (mu - L). Larger k = sharper bettors.

    c > 0 [hr^-1]:
        Exposure decay (mean-reversion). The bookmaker absorbs bets
        over time, reducing one-sided liability. Half-life = ln(2)/c.

    a > 0 [pt * hr^-1 * exposure^-1]:
        Line-adjustment sensitivity. How much the bookmaker moves the
        line per unit of net exposure. Larger a = more reactive book.

    eta > 0 [hr^-1]:
        Direct information-tracking rate. The line moves toward mu
        even with zero exposure (the book adjusts based on its own
        models). Larger eta = faster convergence to fair value.

CRITICAL IDENTIFIABILITY NOTE:
    Since q(t) is never observed, its absolute scale is arbitrary.
    From L(t) data alone, only these composites are identifiable:
        gamma   = c + eta        (effective damping)
        omega_sq = c*eta + a*k   (effective stiffness / natural freq^2)
    The individual parameters (k, c, a, eta) require the q(t) scale
    to be pinned, which we do by fixing q(0) = 0.

EQUILIBRIUM (set dq/dt = dL/dt = 0):
    From (1): k(mu - L*) = c*q*
    From (2): a*q* + eta*(mu - L*) = 0  =>  q* = -eta*(mu - L*) / a
    Substituting: k(mu - L*) = -c*eta*(mu - L*)/a
    Since k + c*eta/a > 0, we need mu - L* = 0.
    Therefore: L* = mu, q* = 0.
    The line converges to fair value with balanced exposure.

STABILITY (Jacobian at equilibrium):
    A = [[-c, -k],
         [ a, -eta]]

    trace(A) = -(c + eta) < 0          (always)
    det(A)   = c*eta + a*k > 0         (always, for positive params)
    => Both eigenvalues have negative real parts => stable.

EIGENVALUES:
    lambda = [-(c + eta) +/- sqrt(Delta)] / 2
    Delta  = (c - eta)^2 - 4*a*k

    Delta > 0: overdamped   (monotone exponential decay to mu)
    Delta = 0: critically damped
    Delta < 0: underdamped  (damped oscillations around mu)

    Oscillation occurs when 4*a*k > (c - eta)^2.
    Interpretation: strong betting response (k) and aggressive bookmaker
    adjustment (a) create overshoot — the line passes mu, triggering
    reverse flow, creating oscillation. Damped by c and eta.

SECOND-ORDER REDUCTION (eliminate q):
    Differentiate (2), substitute (1):
        L'' + (c + eta)*L' + (c*eta + a*k)*(L - mu) = 0

    This is a damped harmonic oscillator:
        damping coefficient: gamma = c + eta
        natural frequency^2: omega_sq = c*eta + a*k
        equilibrium: mu

TIME CONVENTION:
    t = 0 at kickoff. t < 0 earlier (e.g., t = -48 is 48 hours before).
    We integrate forward: solve_ivp(fun, [t_start, 0], ...) where t_start < 0.
"""

import numpy as np
from scipy.integrate import solve_ivp


def ode_system(t: float, y: np.ndarray, k: float, c: float,
               a: float, eta: float, mu: float) -> list[float]:
    """Right-hand side of the 2-D ODE system.

    Parameters
    ----------
    t : float
        Current time (hours, t=0 at kickoff).
    y : array [q, L]
        q = net exposure, L = totals line.
    k, c, a, eta : float
        ODE parameters (all strictly positive).
    mu : float
        Fundamental value (constant).

    Returns
    -------
    [dq/dt, dL/dt]
    """
    q, L = y
    dqdt = k * (mu - L) - c * q
    dLdt = a * q + eta * (mu - L)
    return [dqdt, dLdt]


def jacobian(c: float, k: float, a: float, eta: float) -> np.ndarray:
    """Jacobian matrix of the linearised system at equilibrium.

    A = [[-c, -k],
         [ a, -eta]]

    Used for eigenvalue analysis and stability assessment.
    """
    return np.array([[-c, -k],
                     [a, -eta]])


def eigenvalues(c: float, k: float, a: float, eta: float) -> tuple:
    """Compute eigenvalues and classify the dynamical regime.

    Returns
    -------
    lambda1, lambda2 : complex
        The two eigenvalues.
    discriminant : float
        (c - eta)^2 - 4*a*k. Positive = overdamped, negative = underdamped.
    regime : str
        'overdamped', 'critically_damped', or 'underdamped'.
    """
    gamma = c + eta
    omega_sq = c * eta + a * k
    discriminant = (c - eta) ** 2 - 4 * a * k

    if discriminant > 1e-12:
        sqrt_d = np.sqrt(discriminant)
        lam1 = (-gamma + sqrt_d) / 2.0
        lam2 = (-gamma - sqrt_d) / 2.0
        regime = "overdamped"
    elif discriminant < -1e-12:
        sqrt_d = np.sqrt(-discriminant)
        lam1 = complex(-gamma / 2.0, sqrt_d / 2.0)
        lam2 = complex(-gamma / 2.0, -sqrt_d / 2.0)
        regime = "underdamped"
    else:
        lam1 = lam2 = -gamma / 2.0
        regime = "critically_damped"

    return lam1, lam2, discriminant, regime


def composite_params(k: float, c: float, a: float, eta: float) -> dict:
    """Compute the observable composite parameters.

    These are more robustly estimable than individual params because
    they directly appear in the second-order ODE for L(t).

    gamma   = c + eta         (damping coefficient)
    omega_sq = c*eta + a*k    (natural frequency squared)
    """
    gamma = c + eta
    omega_sq = c * eta + a * k
    discriminant = (c - eta) ** 2 - 4 * a * k
    _, _, _, regime = eigenvalues(c, k, a, eta)
    return {
        "gamma": gamma,
        "omega_sq": omega_sq,
        "discriminant": discriminant,
        "regime": regime,
        "half_life_hrs": np.log(2) / (gamma / 2) if gamma > 0 else np.inf,
    }


def solve_ode(k: float, c: float, a: float, eta: float, mu: float,
              L0: float, q0: float, t_start: float, t_end: float,
              t_eval: np.ndarray | None = None,
              n_dense: int = 500) -> dict:
    """Solve the ODE system forward from t_start to t_end.

    Parameters
    ----------
    k, c, a, eta : float
        ODE parameters.
    mu : float
        Fundamental value.
    L0 : float
        Initial line value at t_start.
    q0 : float
        Initial exposure at t_start (usually 0).
    t_start, t_end : float
        Time window (hours). Typically t_start < 0, t_end = 0.
    t_eval : array, optional
        Specific times to evaluate. If None, uses n_dense evenly spaced.
    n_dense : int
        Number of points for dense output (if t_eval is None).

    Returns
    -------
    dict with keys: t, q, L, success
    """
    if t_eval is None:
        t_eval = np.linspace(t_start, t_end, n_dense)

    sol = solve_ivp(
        ode_system,
        (t_start, t_end),
        [q0, L0],
        args=(k, c, a, eta, mu),
        t_eval=t_eval,
        method="RK45",
        rtol=1e-10,
        atol=1e-12,
    )

    return {
        "t": sol.t,
        "q": sol.y[0],
        "L": sol.y[1],
        "success": sol.success,
        "message": sol.message,
    }


def analytic_solution(k: float, c: float, a: float, eta: float,
                      mu: float, L0: float, q0: float,
                      t: np.ndarray, t0: float = 0.0) -> np.ndarray:
    """Analytic solution for L(t) via the second-order reduction.

    L(t) = mu + A*exp(lam1*(t-t0)) + B*exp(lam2*(t-t0))

    where A, B are determined from initial conditions:
        L(t0) = L0  =>  A + B = L0 - mu
        L'(t0) = a*q0 + eta*(mu - L0)  =>  lam1*A + lam2*B = L'(t0)

    This provides an independent verification of the numerical solver.
    """
    lam1, lam2, disc, regime = eigenvalues(c, k, a, eta)

    # Initial velocity from the ODE
    L_prime_0 = a * q0 + eta * (mu - L0)
    delta_0 = L0 - mu  # initial deviation from equilibrium

    dt = t - t0

    if regime == "critically_damped":
        # L(t) = mu + (A + B*(t-t0)) * exp(lam1*(t-t0))
        lam = float(np.real(lam1))
        A = delta_0
        B = L_prime_0 - lam * delta_0
        L_t = mu + (A + B * dt) * np.exp(lam * dt)
    else:
        # General case: L(t) = mu + A*exp(lam1*dt) + B*exp(lam2*dt)
        # Solve: A + B = delta_0
        #        lam1*A + lam2*B = L_prime_0
        if abs(lam1 - lam2) < 1e-15:
            # Numerically degenerate — fallback to critically damped
            lam = float(np.real(lam1))
            A = delta_0
            B = L_prime_0 - lam * delta_0
            L_t = mu + (A + B * dt) * np.exp(lam * dt)
        else:
            B_coef = (L_prime_0 - lam1 * delta_0) / (lam2 - lam1)
            A_coef = delta_0 - B_coef
            L_t = mu + np.real(A_coef * np.exp(lam1 * dt) + B_coef * np.exp(lam2 * dt))

    return L_t
