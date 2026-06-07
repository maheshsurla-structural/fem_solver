"""Conditional Mean Spectrum (Baker 2011) and Baker-Jayaram (2008)
inter-period correlation model for ground-motion record selection.

The **Conditional Mean Spectrum (CMS)** is the expected spectral
shape of a ground motion that has ``Sa(T*) = Sa_target`` at the
conditioning period ``T*``, given a GMPE-predicted mean spectrum
``mu_lnSa(T)`` and its log-standard-deviation ``sigma_lnSa(T)``::

    ln CMS(T_i | Sa(T*) = Sa_target) =
        mu_lnSa(T_i)  +  rho(T_i, T*) · epsilon(T*) · sigma_lnSa(T_i)

where::

    epsilon(T*) = ( ln(Sa_target) - mu_lnSa(T*) ) / sigma_lnSa(T*)

and ``rho(T_i, T*)`` is the inter-period correlation of log-Sa from
the Baker-Jayaram (2008) empirical model.

The CMS is the canonical replacement for the *uniform hazard
spectrum* in PBSE record selection: the UHS over-predicts the
spectral shape because it implicitly assumes Sa is at high
percentile across all periods simultaneously, which no single
record can deliver.

References
----------
* Baker, J.W. (2011) "Conditional Mean Spectrum: Tool for Ground-
  Motion Selection." *J. Struct. Eng.*, 137(3), 322-331.
* Baker, J.W. & Jayaram, N. (2008) "Correlation of spectral
  acceleration values from NGA ground motion models." *Earthquake
  Spectra*, 24(1), 299-317.
"""
from __future__ import annotations

import math

import numpy as np


# ============================================================ correlation

def baker_jayaram_correlation(T_i: float, T_j: float) -> float:
    """Inter-period log-Sa correlation rho(T_i, T_j) per Baker &
    Jayaram 2008 Eq. 1-6.

    Symmetric in ``(T_i, T_j)``. Returns 1.0 when ``T_i == T_j``.

    Parameters
    ----------
    T_i, T_j : float
        Two periods (s), strictly positive.

    Returns
    -------
    rho : float
        In [-1, 1]. For closely-spaced periods returns near 1; for
        widely-separated periods returns lower (down to ~ 0.2-0.4
        at 1 decade separation).
    """
    if T_i <= 0.0 or T_j <= 0.0:
        raise ValueError("periods must be positive")
    T_min = min(T_i, T_j)
    T_max = max(T_i, T_j)

    # Equation 1: smooth base correlation.
    C1 = 1.0 - math.cos(
        math.pi / 2.0
        - 0.366 * math.log(T_max / max(T_min, 0.109))
    )

    # Equation 2: short-period correction (T_max < 0.2 s).
    if T_max < 0.2:
        C2 = 1.0 - 0.105 * (
            1.0 - 1.0 / (1.0 + math.exp(100.0 * T_max - 5.0))
        ) * (T_max - T_min) / (T_max - 0.0099)
    else:
        C2 = C1

    # Equation 3: very short period (T_max < 0.109 s) -> use C2.
    if T_max < 0.109:
        C3 = C2
    else:
        C3 = C1

    # Equation 4: smooth bridge from C3 to C1 at T_min < 0.109.
    C4 = C1 + 0.5 * (math.sqrt(C3) - C3) * (
        1.0 + math.cos(math.pi * T_min / 0.109)
    )

    # Branch selection (Eq. 5-6).
    if T_max <= 0.109:
        rho = C2
    elif T_min > 0.109:
        rho = C1
    elif T_max < 0.2:
        rho = min(C2, C4)
    else:
        rho = C4
    # Clip numerical noise.
    return float(max(-1.0, min(1.0, rho)))


# ============================================================ CMS

def compute_epsilon(
    Sa_target: float,
    mu_Sa_at_Tstar: float,
    sigma_lnSa_at_Tstar: float,
) -> float:
    """Compute ``epsilon(T*) = ( ln Sa_target - mu_lnSa(T*) ) / sigma_lnSa(T*)``.

    Parameters
    ----------
    Sa_target : float
        Target spectral acceleration at the conditioning period.
        E.g., the MCE_R or design-level Sa at T_1.
    mu_Sa_at_Tstar : float
        GMPE-predicted MEDIAN Sa at the conditioning period (linear,
        same units as ``Sa_target``).
    sigma_lnSa_at_Tstar : float
        GMPE-predicted log-stddev at the conditioning period.
    """
    if Sa_target <= 0.0:
        raise ValueError("Sa_target must be positive")
    if mu_Sa_at_Tstar <= 0.0:
        raise ValueError("mu_Sa must be positive")
    if sigma_lnSa_at_Tstar <= 0.0:
        raise ValueError("sigma_lnSa must be positive")
    return (math.log(Sa_target) - math.log(mu_Sa_at_Tstar)) / sigma_lnSa_at_Tstar


def conditional_mean_spectrum(
    T_star: float,
    epsilon_star: float,
    periods: np.ndarray,
    mu_lnSa: np.ndarray,
    sigma_lnSa: np.ndarray,
) -> np.ndarray:
    """Compute the Conditional Mean Spectrum at the given periods.

    For each ``T_i`` in ``periods``::

        ln CMS(T_i) = mu_lnSa(T_i)
                    + rho(T_i, T*) · epsilon* · sigma_lnSa(T_i)

    Parameters
    ----------
    T_star : float
        Conditioning period (s).
    epsilon_star : float
        Epsilon at the conditioning period (typically 1.0-2.5 for
        MCE-level demands; see :func:`compute_epsilon`).
    periods : array
        Periods (s) at which to evaluate the CMS.
    mu_lnSa : array
        GMPE-predicted MEAN of ln(Sa) at each period.
    sigma_lnSa : array
        GMPE-predicted standard deviation of ln(Sa) at each period.

    Returns
    -------
    CMS_Sa : np.ndarray
        Conditional mean Sa(T_i) (linear), same shape as ``periods``.
    """
    periods = np.asarray(periods, dtype=float).ravel()
    mu = np.asarray(mu_lnSa, dtype=float).ravel()
    sig = np.asarray(sigma_lnSa, dtype=float).ravel()
    if periods.shape != mu.shape or periods.shape != sig.shape:
        raise ValueError(
            "periods, mu_lnSa, sigma_lnSa must have same shape"
        )
    if T_star <= 0.0:
        raise ValueError("T_star must be positive")
    if np.any(periods <= 0.0):
        raise ValueError("periods must all be positive")
    if np.any(sig < 0.0):
        raise ValueError("sigma_lnSa must be non-negative")

    rho = np.array([baker_jayaram_correlation(T_i, T_star)
                    for T_i in periods])
    ln_CMS = mu + rho * epsilon_star * sig
    return np.exp(ln_CMS)


def conditional_spectrum_variance(
    T_star: float,
    periods: np.ndarray,
    sigma_lnSa: np.ndarray,
) -> np.ndarray:
    """Conditional variance of ``ln Sa(T_i) | Sa(T*) = Sa_target``.

    Per Baker (2011) the conditional variance is::

        var_cond(T_i) = sigma_lnSa(T_i)^2 · (1 - rho(T_i, T*)^2)

    Useful for selecting/matching records to the CMS with explicit
    bands on permissible variance (i.e., diversifying around the CMS).

    Returns standard deviation (not variance) for direct use as a
    one-sigma envelope, sqrt of the above.
    """
    periods = np.asarray(periods, dtype=float).ravel()
    sig = np.asarray(sigma_lnSa, dtype=float).ravel()
    if periods.shape != sig.shape:
        raise ValueError("periods and sigma_lnSa must have same shape")
    rho = np.array([baker_jayaram_correlation(T_i, T_star)
                    for T_i in periods])
    var_cond = sig ** 2 * (1.0 - rho ** 2)
    return np.sqrt(np.maximum(var_cond, 0.0))
