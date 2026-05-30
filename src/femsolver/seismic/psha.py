"""Probabilistic Seismic Hazard Analysis (PSHA) -- Cornell-McGuire
formulation.

The annual rate of exceedance of an intensity measure ``im`` is::

    lambda(IM > im) = sum_sources nu_i * integral_M integral_R
                     P(IM > im | M, R) * f_M(M) f_R(R | M) dR dM

where:

* ``nu_i`` -- mean annual rate of earthquakes ``M >= M_min`` on
  source ``i``.
* ``f_M(M)`` -- magnitude-frequency distribution (truncated
  Gutenberg-Richter here).
* ``f_R(R | M)`` -- distance distribution (point or area-source
  geometry).
* ``P(IM > im | M, R)`` -- from the GMPE assuming log-normal
  variability with sigma_lnSa.

The hazard curve gives ``lambda(im)`` at a series of ``im`` values;
the UHS reads, for each spectral period, the IM matching a target
return period.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from femsolver.seismic.gmpe import BooreAtkinsonLike


# ============================================================ magnitude

class GutenbergRichterMFD:
    """Truncated exponential magnitude-frequency distribution::

        f_M(m) = beta * exp(-beta (m - M_min)) /
                 (1 - exp(-beta (M_max - M_min)))   for M_min <= m <= M_max

    with ``beta = b * ln(10)``. The mean rate of earthquakes
    ``M >= M_min`` is ``nu_M_min = 10^(a - b M_min)`` from the
    Gutenberg-Richter law ``log10 N(M) = a - b M``.
    """

    def __init__(
        self,
        *,
        a: float,
        b: float,
        M_min: float,
        M_max: float,
    ):
        if b <= 0:
            raise ValueError(f"b must be positive, got {b}")
        if not M_max > M_min:
            raise ValueError(
                f"M_max ({M_max}) must exceed M_min ({M_min})"
            )
        self.a = float(a)
        self.b = float(b)
        self.M_min = float(M_min)
        self.M_max = float(M_max)
        self.beta = self.b * math.log(10.0)
        self.nu_M_min = 10.0 ** (self.a - self.b * self.M_min)

    def pdf(self, m: float) -> float:
        """``f_M(m)`` (probability density)."""
        if m < self.M_min or m > self.M_max:
            return 0.0
        denom = 1.0 - math.exp(-self.beta * (self.M_max - self.M_min))
        return self.beta * math.exp(-self.beta * (m - self.M_min)) / denom


# ============================================================ sources

@dataclass
class PointSource:
    """Point seismic source: all earthquakes occur at a single
    distance from the site."""

    name: str
    R_jb_km: float
    mfd: GutenbergRichterMFD


@dataclass
class AreaSource:
    """Areal source: distance distribution discretised as
    ``(R_i, w_i)`` weights where ``sum w_i = 1``."""

    name: str
    distances_km: np.ndarray
    weights: np.ndarray
    mfd: GutenbergRichterMFD

    def __post_init__(self):
        self.distances_km = np.asarray(self.distances_km, dtype=float)
        self.weights = np.asarray(self.weights, dtype=float)
        if self.distances_km.shape != self.weights.shape:
            raise ValueError(
                "AreaSource: distances and weights must have the same shape"
            )
        if not np.isclose(self.weights.sum(), 1.0):
            raise ValueError(
                "AreaSource: weights must sum to 1, "
                f"got {self.weights.sum():.4f}"
            )


# ============================================================ hazard curve

@dataclass
class HazardCurve:
    """A hazard curve at one period."""

    period: float
    im_levels: np.ndarray
    annual_rates: np.ndarray         # lambda(IM > im) for each im_level

    def annual_rate_at(self, im: float) -> float:
        """Interpolate annual rate at the queried IM value."""
        return float(np.interp(
            im, self.im_levels, self.annual_rates,
            left=self.annual_rates[0], right=self.annual_rates[-1],
        ))

    def im_at_return_period(self, T_R: float) -> float:
        """IM level at a target return period (years)."""
        lambda_target = 1.0 / T_R
        # Hazard curve is monotonically decreasing in im
        # Use inverse interpolation on ln(rate) for accuracy
        ln_rates = np.log(np.maximum(self.annual_rates, 1e-300))
        # interp in im space using -ln(lambda) increasing
        order = np.argsort(-ln_rates)
        ims_sorted = self.im_levels[order]
        ln_target = -math.log(lambda_target)
        ln_neg = -ln_rates[order]
        return float(np.interp(ln_target, ln_neg, ims_sorted))


@dataclass
class UniformHazardSpectrum:
    """Uniform Hazard Spectrum -- IM at one target return period
    across multiple periods."""

    return_period: float
    periods: np.ndarray
    sa_values: np.ndarray            # Sa at each period


def _exceedance_prob_lognormal(
    *, ln_im: float, ln_median: float, sigma_lnSa: float,
) -> float:
    """``P(ln IM > ln_im) = 1 - Phi((ln_im - ln_median) / sigma)``."""
    z = (ln_im - ln_median) / sigma_lnSa
    # Survival function of standard normal
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def compute_hazard_curve(
    *,
    gmpe: BooreAtkinsonLike,
    sources: Sequence,
    im_levels: np.ndarray,
    V_s30: float = 760.0,
    n_M: int = 25,
) -> HazardCurve:
    """Cornell PSHA hazard curve.

    Parameters
    ----------
    gmpe : BooreAtkinsonLike
        Ground-motion model (defines period T via gmpe.T).
    sources : sequence of PointSource / AreaSource
    im_levels : (n,) array of IM levels (g) to evaluate.
    V_s30 : float
        Site shear-wave velocity (m/s).
    n_M : int
        Number of magnitude bins for integration over each source's MFD.

    Returns
    -------
    HazardCurve
    """
    im_levels = np.asarray(im_levels, dtype=float)
    rates = np.zeros_like(im_levels)
    for src in sources:
        mfd = src.mfd
        Ms = np.linspace(mfd.M_min, mfd.M_max, n_M + 1)
        Ms_mid = 0.5 * (Ms[1:] + Ms[:-1])
        dM = Ms[1] - Ms[0]
        # f_M weights at mid-bin
        f_M = np.array([mfd.pdf(m) for m in Ms_mid])
        # Distance distribution: PointSource has single R; AreaSource has
        # discretised R_i, w_i
        if isinstance(src, PointSource):
            Rs = np.array([src.R_jb_km])
            R_w = np.array([1.0])
        else:
            Rs = src.distances_km
            R_w = src.weights
        # Loop over M, R, IM and accumulate
        nu = mfd.nu_M_min
        for i_M, (M, fM) in enumerate(zip(Ms_mid, f_M)):
            for R, wR in zip(Rs, R_w):
                # Median Sa at this scenario
                res = gmpe.evaluate(M=M, R_jb=float(R), V_s30=V_s30)
                ln_median = res.median_lnSa
                sigma = res.sigma_lnSa
                # P(IM > im | M, R) for each im
                for j, im in enumerate(im_levels):
                    if im <= 0:
                        continue
                    P_exc = _exceedance_prob_lognormal(
                        ln_im=math.log(im),
                        ln_median=ln_median,
                        sigma_lnSa=sigma,
                    )
                    rates[j] += nu * fM * dM * wR * P_exc
    return HazardCurve(
        period=gmpe.T,
        im_levels=im_levels.copy(),
        annual_rates=rates,
    )


def return_period_to_im(curve: HazardCurve, T_R: float) -> float:
    """Convenience wrapper around :meth:`HazardCurve.im_at_return_period`."""
    return curve.im_at_return_period(T_R)


def compute_uhs(
    *,
    gmpes_by_period: dict[float, BooreAtkinsonLike],
    sources: Sequence,
    return_period: float,
    im_levels: np.ndarray,
    V_s30: float = 760.0,
    n_M: int = 25,
) -> UniformHazardSpectrum:
    """Build a UHS across periods at a fixed return period.

    Parameters
    ----------
    gmpes_by_period : dict[float, BooreAtkinsonLike]
        One GMPE instance per period of interest.
    sources : sequence
    return_period : float
        Return period of interest (years).
    im_levels : (n,) array
        IM grid used to construct each hazard curve.
    V_s30, n_M : see :func:`compute_hazard_curve`.
    """
    periods = sorted(gmpes_by_period.keys())
    sa_values = []
    for T in periods:
        gmpe = gmpes_by_period[T]
        curve = compute_hazard_curve(
            gmpe=gmpe, sources=sources, im_levels=im_levels,
            V_s30=V_s30, n_M=n_M,
        )
        sa_values.append(curve.im_at_return_period(return_period))
    return UniformHazardSpectrum(
        return_period=float(return_period),
        periods=np.array(periods, dtype=float),
        sa_values=np.array(sa_values, dtype=float),
    )
