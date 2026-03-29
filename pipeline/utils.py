"""
Shared utility functions for the NFL Totals pipeline.

Implements implied-probability conversion, vig-free normalization, and overround
calculation.  Every function validates its inputs and raises clear errors.
"""
import logging
import math

logger = logging.getLogger(__name__)


def implied_probability(american_odds: int) -> float:
    """Convert American odds to implied probability.

    p_tilde(O) = 100 / (O + 100)   if O > 0
    p_tilde(O) = -O  / (-O + 100)  if O < 0

    Raises
    ------
    ValueError
        If american_odds == 0 (invalid) or if the result is not in (0, 1).
    """
    if american_odds == 0:
        raise ValueError("American odds of 0 are invalid.")

    if american_odds > 0:
        p = 100.0 / (american_odds + 100.0)
    else:
        p = (-american_odds) / ((-american_odds) + 100.0)

    if not (0.0 < p < 1.0):
        raise ValueError(
            f"Implied probability {p} from odds {american_odds} is outside (0, 1)."
        )
    return p


def vig_free_probabilities(p_over: float, p_under: float) -> tuple:
    """Normalize implied probabilities to remove vig (sum to 1).

    p_O = p_over  / (p_over + p_under)
    p_U = p_under / (p_over + p_under)

    Raises
    ------
    ValueError
        If either input is <= 0 or >= 1.
    """
    if not (0.0 < p_over < 1.0):
        raise ValueError(f"p_over must be in (0, 1), got {p_over}")
    if not (0.0 < p_under < 1.0):
        raise ValueError(f"p_under must be in (0, 1), got {p_under}")

    total = p_over + p_under
    vf_over = p_over / total
    vf_under = p_under / total

    # Sanity: must sum to 1.0 within floating-point tolerance
    assert math.isclose(vf_over + vf_under, 1.0, abs_tol=1e-10), (
        f"Vig-free probabilities sum to {vf_over + vf_under}, expected 1.0"
    )
    return (vf_over, vf_under)


def overround(p_over: float, p_under: float) -> float:
    """Compute the overround (vig) from raw implied probabilities.

    overround = p_over + p_under - 1

    Should be >= 0 for a properly set book.  Logs a warning if negative.
    """
    if not (0.0 < p_over < 1.0):
        raise ValueError(f"p_over must be in (0, 1), got {p_over}")
    if not (0.0 < p_under < 1.0):
        raise ValueError(f"p_under must be in (0, 1), got {p_under}")

    o = p_over + p_under - 1.0
    if o < 0:
        logger.warning(
            "Negative overround %.6f — implies under-round book (bad data?)", o
        )
    return o
