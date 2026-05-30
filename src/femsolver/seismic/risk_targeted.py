"""Risk-targeted Maximum Considered Earthquake (MCE_R) -- ASCE 7-22
Chapter 21.

Given a hazard curve ``lambda(IM > im)`` at a site and a structure's
collapse fragility (log-normal, with median ``theta`` and dispersion
``beta``), the **annual collapse rate** is::

    lambda_C = integral lambda(IM > im) * |d P_C / d im| dim
             = integral P_C(im) * |d lambda / d im| dim   (integ. by parts)

The **risk-targeted MCE_R** is the IM that, when used as the
fragility median ``theta = MCE_R``, yields exactly a 1% probability
of collapse in 50 years. The dispersion is taken as ``beta = 0.6``
per ASCE 7-22 Table 21.2-1.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from femsolver.seismic.psha import HazardCurve


def _normal_cdf(z: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _lognormal_cdf(x: float, theta: float, beta: float) -> float:
    """``Phi(ln(x/theta) / beta)``."""
    if x <= 0:
        return 0.0
    return _normal_cdf(math.log(x / theta) / beta)


def annual_collapse_rate(
    curve: HazardCurve,
    *,
    theta: float,
    beta: float = 0.6,
) -> float:
    """Convolve a hazard curve with a log-normal collapse fragility.

    Parameters
    ----------
    curve : HazardCurve
        Site hazard curve.
    theta : float
        Median collapse capacity (g).
    beta : float, default 0.6
        Log-normal dispersion (ASCE 7-22 Table 21.2-1).

    Returns
    -------
    lambda_C : float
        Mean annual rate of collapse.
    """
    ims = curve.im_levels
    rates = curve.annual_rates
    # |d lambda / d im| via finite differences -- log-log interpolation
    # is more accurate but linear is sufficient for engineering use.
    d_lambda = -np.diff(rates)        # rates are decreasing in im
    im_mids = 0.5 * (ims[1:] + ims[:-1])
    P_C_at_mids = np.array(
        [_lognormal_cdf(im, theta, beta) for im in im_mids]
    )
    return float(np.sum(P_C_at_mids * d_lambda))


def risk_targeted_im(
    curve: HazardCurve,
    *,
    target_collapse_prob: float = 0.01,
    window_years: float = 50.0,
    beta: float = 0.6,
    bracket: tuple[float, float] = (0.01, 5.0),
) -> float:
    """Find the IM such that, used as the collapse median, the structure
    sees ``target_collapse_prob`` collapse probability in ``window_years``
    years (default 1% in 50 yr -> ASCE 7 MCE_R).

    Uses the relation ``P_C = 1 - exp(-lambda_C * window_years)``.
    """
    if not 0 < target_collapse_prob < 1:
        raise ValueError(
            f"target_collapse_prob must be in (0, 1), got {target_collapse_prob}"
        )
    if window_years <= 0:
        raise ValueError(f"window_years must be positive, got {window_years}")
    if beta <= 0:
        raise ValueError(f"beta must be positive, got {beta}")
    target_lambda_C = -math.log(1.0 - target_collapse_prob) / window_years

    def f(theta):
        lc = annual_collapse_rate(curve, theta=theta, beta=beta)
        return lc - target_lambda_C

    # Check brackets
    lo, hi = bracket
    f_lo = f(lo); f_hi = f(hi)
    if f_lo * f_hi > 0:
        # Try expanding the bracket once
        hi *= 4
        f_hi = f(hi)
        if f_lo * f_hi > 0:
            raise RuntimeError(
                f"risk_targeted_im: target not bracketed in "
                f"[{bracket[0]}, {hi}]; f_lo={f_lo:.3e}, f_hi={f_hi:.3e}"
            )
    return float(brentq(f, lo, hi, xtol=1e-6))
