"""ASCE 7-22 §16.2.4.1 amplitude scaling of ground-motion records.

For nonlinear response-history analysis the standard workflow is:

1. **Compute** the 5%-damped pseudo-acceleration response spectrum
   ``Sa(T)`` of each candidate record by integrating an SDOF
   oscillator under the recorded acceleration history at a series
   of periods (the response-spectrum computation).
2. **Scale** each record by a single scalar factor so that the
   geometric mean of its spectrum over the period range
   ``[0.2 T_1, 2.0 T_1]`` matches the target design spectrum's
   geometric mean over the same range. (ASCE 7-22 Method A.)
3. **Verify** that the average of the scaled-suite spectra does not
   fall below 90 percent of the target spectrum at any period in
   the range -- the ASCE 7-22 §16.2.4.1 acceptance test.

This module provides the three pieces and a top-level
:func:`scale_record_suite` that runs the full workflow on a list of
records and returns the per-record scale factors plus an aggregated
suite-acceptance report.

References
----------
* ASCE 7-22 §16.2.4 "Ground Motion Modification".
* Chopra, *Dynamics of Structures*, 5e §6.5, "Response Spectrum
  Concept" -- the Newmark constant-average-acceleration algorithm
  used here for SDOF integration.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ============================================================ SDOF spectrum

def compute_sdof_response_spectrum(
    accel: np.ndarray,
    dt: float,
    periods: np.ndarray,
    *,
    zeta: float = 0.05,
) -> np.ndarray:
    """Compute the 5%-damped pseudo-acceleration response spectrum
    of a ground-acceleration time series.

    For each period ``T`` in ``periods``, integrates the SDOF
    response::

        u_ddot + 2 zeta omega u_dot + omega^2 u = -a_g(t)

    with Newmark-beta (constant-average-acceleration: gamma = 1/2,
    beta = 1/4 -- unconditionally stable) and returns the peak
    pseudo-acceleration ``Sa = omega^2 * max|u|`` at each period.

    Parameters
    ----------
    accel : array
        Ground acceleration samples ``a_g[i]`` at equal spacing ``dt``.
    dt : float
        Sample interval (s).
    periods : array
        Periods (s) at which to compute the spectrum. Must be all > 0.
    zeta : float, default 0.05
        Damping ratio (0..1, fraction of critical).

    Returns
    -------
    Sa : np.ndarray
        Peak pseudo-acceleration at each ``periods[i]``, same units
        as ``accel``.
    """
    accel = np.asarray(accel, dtype=float).ravel()
    periods = np.asarray(periods, dtype=float).ravel()
    if accel.size < 2:
        raise ValueError("need at least 2 acceleration samples")
    if dt <= 0.0:
        raise ValueError(f"dt must be positive, got {dt}")
    if np.any(periods <= 0.0):
        raise ValueError("periods must all be positive")
    if not (0.0 <= zeta < 1.0):
        raise ValueError(f"zeta must be in [0, 1), got {zeta}")

    n_steps = accel.size
    omega = 2.0 * np.pi / periods                  # (n_T,)
    n_T = periods.size

    # State vectors, one per period (vectorised).
    u = np.zeros(n_T)
    v = np.zeros(n_T)
    # Initial acceleration: with u = v = 0, u_ddot_0 = -a_g[0].
    a = -accel[0] * np.ones(n_T)
    Sa = np.zeros(n_T)

    # Effective stiffness for Newmark constant-avg-accel:
    #   ü_next · (1 + ζ ω Δt + ω² Δt²/4) = -a_g_next - 2 ζ ω v_pred - ω² u_pred
    coef = 1.0 + zeta * omega * dt + 0.25 * omega ** 2 * dt ** 2

    for i in range(n_steps - 1):
        a_next = accel[i + 1]
        u_pred = u + dt * v + 0.25 * dt ** 2 * a
        v_pred = v + 0.5 * dt * a
        a_new = (-a_next - 2.0 * zeta * omega * v_pred
                 - omega ** 2 * u_pred) / coef
        u_new = u_pred + 0.25 * dt ** 2 * a_new
        v_new = v_pred + 0.5 * dt * a_new
        u, v, a = u_new, v_new, a_new
        Sa = np.maximum(Sa, omega ** 2 * np.abs(u))

    return Sa


def record_response_spectrum(
    accel_function,
    *,
    t_end: float,
    dt: float,
    periods: np.ndarray,
    zeta: float = 0.05,
) -> np.ndarray:
    """Convenience wrapper that samples ``accel_function`` then runs
    :func:`compute_sdof_response_spectrum`.

    Parameters
    ----------
    accel_function : callable
        ``a_g(t) -> float``.
    t_end : float
        Total record duration (s).
    dt : float
        Sample interval (s).
    periods : array
    zeta : float
    """
    times = np.arange(0.0, t_end + 0.5 * dt, dt)
    accel = np.array([accel_function(t) for t in times])
    return compute_sdof_response_spectrum(accel, dt, periods, zeta=zeta)


# ============================================================ scaling helpers

def amplitude_scale_factor(
    record_Sa: np.ndarray,
    target_Sa: np.ndarray,
    *,
    weights: np.ndarray | None = None,
) -> float:
    """Scale factor that matches the geometric mean of ``record_Sa``
    to that of ``target_Sa`` (ASCE 7-22 Method A).

    The scale factor is::

        SF = exp(  mean( w_i · ln(target_Sa_i / record_Sa_i) )  /
                   mean( w_i )  )

    For equal weights this reduces to the ratio of geometric means.
    The same SF applied multiplicatively to the time-history will
    bring the spectrum to a least-squares match in log space over
    the period range.

    Parameters
    ----------
    record_Sa, target_Sa : array
        Spectra at the same periods. Must be same shape and > 0.
    weights : array, optional
        Per-period weights (defaults to uniform).

    Returns
    -------
    SF : float
    """
    rec = np.asarray(record_Sa, dtype=float).ravel()
    tar = np.asarray(target_Sa, dtype=float).ravel()
    if rec.shape != tar.shape:
        raise ValueError("record_Sa and target_Sa must have same shape")
    if rec.size == 0:
        raise ValueError("spectra must be non-empty")
    if np.any(rec <= 0.0) or np.any(tar <= 0.0):
        raise ValueError("spectra must be strictly positive")
    if weights is None:
        ln_ratio_mean = float(np.mean(np.log(tar / rec)))
    else:
        w = np.asarray(weights, dtype=float).ravel()
        if w.shape != rec.shape:
            raise ValueError("weights must have same shape as spectra")
        if np.sum(w) <= 0.0:
            raise ValueError("weights must sum to > 0")
        ln_ratio_mean = float(np.sum(w * np.log(tar / rec)) / np.sum(w))
    return float(np.exp(ln_ratio_mean))


# ============================================================ suite check

@dataclass
class SuiteScalingResult:
    """Result of scaling an entire suite to a target spectrum.

    Attributes
    ----------
    scale_factors : np.ndarray
        Per-record scale factor (same order as input records).
    periods_check : np.ndarray
        Periods at which the suite-average check was evaluated.
    suite_average_Sa : np.ndarray
        Average of the scaled-suite spectra at each ``periods_check``.
    target_Sa : np.ndarray
        Target spectrum at each ``periods_check``.
    min_ratio : float
        Smallest value of ``suite_average_Sa / target_Sa`` over the
        period range.
    passes_90pct : bool
        True iff ``min_ratio >= 0.90`` -- the ASCE 7-22 §16.2.4.1
        acceptance criterion.
    n_records : int
    """

    scale_factors: np.ndarray = field(default_factory=lambda: np.empty(0))
    periods_check: np.ndarray = field(default_factory=lambda: np.empty(0))
    suite_average_Sa: np.ndarray = field(default_factory=lambda: np.empty(0))
    target_Sa: np.ndarray = field(default_factory=lambda: np.empty(0))
    min_ratio: float = 0.0
    passes_90pct: bool = False
    n_records: int = 0


def scale_record_suite(
    record_spectra: list[np.ndarray],
    target_Sa_at_periods: np.ndarray,
    *,
    period_range_mask: np.ndarray | None = None,
) -> SuiteScalingResult:
    """Apply ASCE 7-22 amplitude scaling to a suite of records.

    Each record's scale factor is chosen by :func:`amplitude_scale_factor`
    matching its spectrum's geometric-mean to the target over the
    selected period range (``period_range_mask``). After scaling,
    the suite-average spectrum is computed and compared to the target
    -- the suite passes acceptance if the smallest ratio is at least
    0.90 at every period in the range.

    Parameters
    ----------
    record_spectra : list of arrays
        One per record. Each is ``Sa`` sampled at the SAME periods
        as ``target_Sa_at_periods``.
    target_Sa_at_periods : array
        Target ``Sa(T)`` at the analysis periods.
    period_range_mask : boolean array, optional
        Mask selecting the period band over which scaling and the
        suite-average check are evaluated. If omitted, all periods
        are used.

    Returns
    -------
    SuiteScalingResult
    """
    target = np.asarray(target_Sa_at_periods, dtype=float).ravel()
    if target.size == 0:
        raise ValueError("target spectrum must be non-empty")
    if len(record_spectra) == 0:
        raise ValueError("need at least one record")

    if period_range_mask is None:
        mask = np.ones(target.size, dtype=bool)
    else:
        mask = np.asarray(period_range_mask, dtype=bool).ravel()
        if mask.shape != target.shape:
            raise ValueError(
                "period_range_mask must match target spectrum shape"
            )
        if not np.any(mask):
            raise ValueError("period_range_mask selects no periods")

    target_in = target[mask]
    scale_factors = np.empty(len(record_spectra))
    scaled_spectra = np.empty((len(record_spectra), target.size))
    for i, rec in enumerate(record_spectra):
        rec = np.asarray(rec, dtype=float).ravel()
        if rec.shape != target.shape:
            raise ValueError(
                f"record {i} spectrum shape {rec.shape} != target shape "
                f"{target.shape}"
            )
        scale_factors[i] = amplitude_scale_factor(rec[mask], target_in)
        scaled_spectra[i] = scale_factors[i] * rec

    suite_avg = np.mean(scaled_spectra, axis=0)
    ratio = suite_avg[mask] / target_in
    min_ratio = float(np.min(ratio))
    return SuiteScalingResult(
        scale_factors=scale_factors,
        periods_check=np.flatnonzero(mask),
        suite_average_Sa=suite_avg,
        target_Sa=target,
        min_ratio=min_ratio,
        passes_90pct=bool(min_ratio >= 0.90),
        n_records=len(record_spectra),
    )


def period_range_mask(
    periods: np.ndarray, T1: float,
    *,
    low_mult: float = 0.2, high_mult: float = 2.0,
) -> np.ndarray:
    """Boolean mask selecting periods in ``[low_mult·T1, high_mult·T1]``.

    The default ``0.2 T_1`` to ``2.0 T_1`` reflects ASCE 7-22
    §16.2.4.1 for nonlinear response-history analysis.
    """
    periods = np.asarray(periods, dtype=float).ravel()
    if T1 <= 0.0:
        raise ValueError("T1 must be positive")
    if low_mult <= 0.0 or high_mult <= low_mult:
        raise ValueError("require 0 < low_mult < high_mult")
    return (periods >= low_mult * T1) & (periods <= high_mult * T1)
