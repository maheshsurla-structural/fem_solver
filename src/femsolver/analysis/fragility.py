"""Lognormal fragility-curve fitting from collapse-IM samples.

The classical seismic-collapse fragility curve (FEMA P695, Baker 2015)
is a lognormal CDF in the intensity measure ``IM``:

    P(collapse | IM) = Φ( ln(IM / θ) / β )

where ``θ`` is the **median** collapse IM and ``β`` the **logarithmic
standard deviation** (dispersion). The two parameters are estimated
from a set of per-record collapse IMs via:

* **Method-of-moments** (Baker 2015, "Efficient analytical fragility
  function fitting"): θ = exp(mean(ln IM_i)),
                       β = std(ln IM_i, sample).
  This is the standard estimator when every record collapsed within
  the IDA sweep. Returns the maximum-likelihood estimates of θ and β
  for a lognormal model.

* **Maximum-likelihood / SPO2IDA-style** robust to non-collapsing
  records via censoring (FEMA P695, Appendix F). Records that did not
  collapse contribute via ``P(no collapse | IM_max) = 1 - Φ(...)``,
  preventing the median from being biased downward when some records
  flatlined at the top of the sweep.

Use :func:`fit_lognormal_method_of_moments` when all records collapsed,
and :func:`fit_lognormal_mle` when some did not.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ============================================================ result type

@dataclass
class FragilityFit:
    """Result of fitting a lognormal collapse fragility.

    Attributes
    ----------
    theta : float
        Median collapse IM.
    beta : float
        Logarithmic standard deviation (dispersion).
    n_records : int
        Total number of records considered (collapsed + censored).
    n_collapsed : int
        Number of records that actually collapsed (used in MLE
        censoring).
    method : str
        ``"moments"`` or ``"mle"``.
    """

    theta: float
    beta: float
    n_records: int
    n_collapsed: int
    method: str

    def P_collapse(self, IM: float) -> float:
        """Probability of collapse at intensity ``IM``: lognormal CDF."""
        if IM <= 0.0:
            return 0.0
        z = math.log(IM / self.theta) / self.beta
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    def P_collapse_array(self, IMs) -> np.ndarray:
        return np.array([self.P_collapse(im) for im in IMs])


# ============================================================ method of moments

def fit_lognormal_method_of_moments(collapse_IMs) -> FragilityFit:
    """Fit a lognormal fragility from a list of per-record collapse
    IMs using the method-of-moments / maximum-likelihood estimator
    when all records collapsed.

    Discards non-finite (``inf``, ``nan``) entries, then estimates:

        θ = exp(mean(ln IM_i))
        β = std(ln IM_i, sample)    (i.e., dividing by N-1)

    Parameters
    ----------
    collapse_IMs : array-like
        Per-record collapse intensity-measure values. Use ``inf`` for
        records that did not collapse (those will be ignored by this
        estimator; use :func:`fit_lognormal_mle` to include them
        properly via censoring).

    Returns
    -------
    FragilityFit
    """
    arr = np.asarray(collapse_IMs, dtype=float)
    finite = arr[np.isfinite(arr) & (arr > 0)]
    if finite.size < 2:
        raise ValueError(
            f"need at least 2 finite collapse IMs, got {finite.size}"
        )
    ln_IM = np.log(finite)
    theta = float(np.exp(np.mean(ln_IM)))
    beta = float(np.std(ln_IM, ddof=1))
    return FragilityFit(
        theta=theta, beta=beta,
        n_records=int(arr.size),
        n_collapsed=int(finite.size),
        method="moments",
    )


# ============================================================ MLE (censored)

def _lognormal_log_likelihood(
    theta: float, beta: float,
    collapse_IMs: np.ndarray,
    no_collapse_IM_max: np.ndarray,
) -> float:
    """Log-likelihood for a censored-collapse sample.

    Records with finite collapse IM contribute the log-density of the
    lognormal at that IM. Records that survived (did not collapse) up
    to IM_max contribute log(1 - F(IM_max)), where F is the lognormal
    CDF.
    """
    if beta <= 0.0:
        return -np.inf
    # Density part
    ll = 0.0
    if collapse_IMs.size > 0:
        z = np.log(collapse_IMs / theta) / beta
        # log-pdf = -log(IM β √(2π)) - z²/2
        ll += np.sum(
            -np.log(collapse_IMs * beta * math.sqrt(2.0 * math.pi))
            - 0.5 * z * z
        )
    # Censoring part
    if no_collapse_IM_max.size > 0:
        z_c = np.log(no_collapse_IM_max / theta) / beta
        # 1 - Φ(z) using erfc for numerical stability
        survival = 0.5 * np.array([math.erfc(zi / math.sqrt(2.0))
                                      for zi in z_c])
        survival = np.maximum(survival, 1.0e-300)    # avoid log(0)
        ll += np.sum(np.log(survival))
    return float(ll)


def fit_lognormal_mle(
    collapse_IMs,
    no_collapse_IM_max,
    *,
    theta_grid=None,
    beta_grid=None,
) -> FragilityFit:
    """Fit a censored-data lognormal fragility via grid-search MLE.

    Records that collapsed contribute their actual collapse IM;
    records that did not collapse contribute via censoring at their
    maximum-IM-tested value (the IDA sweep cap).

    Parameters
    ----------
    collapse_IMs : array-like
        Collapse IMs for records that DID collapse.
    no_collapse_IM_max : array-like
        Maximum IM tested for records that did NOT collapse (their
        IDA sweep cap, i.e., they survived this much).
    theta_grid : array, optional
        Grid of θ candidates. If omitted, spans 0.1× to 10× the
        method-of-moments θ estimate from the collapsed records.
    beta_grid : array, optional
        Grid of β candidates. If omitted, spans 0.10 to 1.20 in
        steps of 0.02.
    """
    c_IMs = np.asarray(collapse_IMs, dtype=float)
    nc_IMs = np.asarray(no_collapse_IM_max, dtype=float)
    c_IMs = c_IMs[np.isfinite(c_IMs) & (c_IMs > 0)]
    nc_IMs = nc_IMs[np.isfinite(nc_IMs) & (nc_IMs > 0)]
    n_total = int(c_IMs.size + nc_IMs.size)
    if n_total < 2:
        raise ValueError(f"need >= 2 records total, got {n_total}")

    if theta_grid is None:
        # Use a moments-style seed if we have any collapses
        if c_IMs.size >= 2:
            theta0 = float(np.exp(np.mean(np.log(c_IMs))))
        elif nc_IMs.size > 0:
            theta0 = 2.0 * float(np.max(nc_IMs))
        else:
            theta0 = 1.0
        theta_grid = np.exp(
            np.linspace(np.log(theta0 / 10.0),
                          np.log(theta0 * 10.0), 100)
        )
    if beta_grid is None:
        beta_grid = np.arange(0.10, 1.21, 0.02)

    best_ll = -np.inf
    best_theta = float(theta_grid[0])
    best_beta = float(beta_grid[0])
    for th in theta_grid:
        for bt in beta_grid:
            ll = _lognormal_log_likelihood(float(th), float(bt),
                                              c_IMs, nc_IMs)
            if ll > best_ll:
                best_ll = ll
                best_theta = float(th)
                best_beta = float(bt)
    return FragilityFit(
        theta=best_theta, beta=best_beta,
        n_records=n_total,
        n_collapsed=int(c_IMs.size),
        method="mle",
    )


# ============================================================ convenience

def fit_collapse_fragility(
    collapse_IMs,
    no_collapse_IM_max=None,
) -> FragilityFit:
    """Dispatch to the appropriate fragility-fit method.

    If ``no_collapse_IM_max`` is provided AND non-empty, uses
    :func:`fit_lognormal_mle`. Otherwise (all records collapsed)
    uses :func:`fit_lognormal_method_of_moments`.
    """
    if no_collapse_IM_max is not None:
        nc = np.asarray(no_collapse_IM_max, dtype=float)
        nc = nc[np.isfinite(nc) & (nc > 0)]
        if nc.size > 0:
            return fit_lognormal_mle(collapse_IMs, nc)
    return fit_lognormal_method_of_moments(collapse_IMs)
