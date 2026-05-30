"""Hazard deaggregation -- identify which (M, R, epsilon) scenarios
contribute most to a given hazard level.

Given the PSHA integrand

    d lambda(IM > im) = nu * f_M(M) * f_R(R | M) * P(IM > im | M, R)

we compute the joint posterior::

    p(M, R, eps | IM > im) =
        contribution(M, R, eps) / sum(contributions)

and report:

* Modal (M, R) -- the scenario with the highest marginal posterior.
* Mean (M, R, eps).
* The joint histogram (returned for downstream display).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from femsolver.seismic.gmpe import BooreAtkinsonLike
from femsolver.seismic.psha import (
    AreaSource,
    GutenbergRichterMFD,
    PointSource,
)


@dataclass
class DeaggregationResult:
    """Output of a deaggregation calculation at one IM level."""

    im_target: float
    period: float
    contributions: np.ndarray    # (n_M, n_R, n_eps) joint probabilities
    M_edges: np.ndarray
    R_edges: np.ndarray
    eps_edges: np.ndarray
    mean_M: float
    mean_R: float
    mean_eps: float
    modal_M: float
    modal_R: float
    modal_eps: float


def _normal_pdf(z: float) -> float:
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)


def deaggregate(
    *,
    gmpe: BooreAtkinsonLike,
    sources: Sequence,
    im_target: float,
    V_s30: float = 760.0,
    M_edges: np.ndarray | None = None,
    R_edges: np.ndarray | None = None,
    eps_edges: np.ndarray | None = None,
) -> DeaggregationResult:
    """Compute the joint deaggregation ``p(M, R, eps | IM = im)``.

    Parameters
    ----------
    gmpe : BooreAtkinsonLike
    sources : sequence of seismic sources
    im_target : float
        IM level to deaggregate (g).
    V_s30 : float, default 760
    M_edges, R_edges, eps_edges : arrays, optional
        Bin edges. Defaults: M in [4.0, 8.0] step 0.25, R in [0, 200]
        step 20, eps in [-3, +3] step 0.5.

    Returns
    -------
    DeaggregationResult
    """
    if im_target <= 0:
        raise ValueError(f"im_target must be positive, got {im_target}")
    if M_edges is None:
        M_edges = np.arange(4.0, 8.0 + 0.25, 0.25)
    if R_edges is None:
        R_edges = np.arange(0.0, 200.0 + 20.0, 20.0)
    if eps_edges is None:
        eps_edges = np.arange(-3.0, 3.0 + 0.5, 0.5)
    M_edges = np.asarray(M_edges, dtype=float)
    R_edges = np.asarray(R_edges, dtype=float)
    eps_edges = np.asarray(eps_edges, dtype=float)
    M_mids = 0.5 * (M_edges[1:] + M_edges[:-1])
    R_mids = 0.5 * (R_edges[1:] + R_edges[:-1])
    eps_mids = 0.5 * (eps_edges[1:] + eps_edges[:-1])
    dM = M_edges[1] - M_edges[0]
    dR = R_edges[1] - R_edges[0]
    dEps = eps_edges[1] - eps_edges[0]
    cube = np.zeros((len(M_mids), len(R_mids), len(eps_mids)))
    ln_im = math.log(im_target)
    for src in sources:
        mfd = src.mfd
        nu = mfd.nu_M_min
        if isinstance(src, PointSource):
            Rs = np.array([src.R_jb_km])
            wR = np.array([1.0])
        else:
            Rs = src.distances_km
            wR = src.weights
        # Map source's R distribution onto the R bins
        for r_val, w in zip(Rs, wR):
            # Snap to nearest R bin
            if r_val < R_edges[0] or r_val > R_edges[-1]:
                continue
            r_bin = int(np.clip(np.searchsorted(R_edges, r_val) - 1,
                                  0, len(R_mids) - 1))
            for i_M, Mc in enumerate(M_mids):
                if Mc < mfd.M_min or Mc > mfd.M_max:
                    continue
                f_M = mfd.pdf(Mc)
                if f_M <= 0:
                    continue
                # Median ln Sa at this scenario
                res = gmpe.evaluate(M=Mc, R_jb=float(r_val), V_s30=V_s30)
                sigma = res.sigma_lnSa
                # epsilon implied by im_target = (ln_im - ln_median) / sigma
                eps_im = (ln_im - res.median_lnSa) / sigma
                # Joint contribution to lambda(IM > im) from
                # (M=Mc, R=r_val, eps=eps_im); use a width on eps as a
                # small Gaussian bandwidth to smear the delta.
                # Approximate the eps distribution as a band of width
                # 'dEps' centred at eps_im.
                for i_eps, eps_c in enumerate(eps_mids):
                    # Probability mass at this eps bin = phi(eps) * dEps
                    p_eps = _normal_pdf(eps_c) * dEps
                    # Indicator that this eps is at or above eps_im (so
                    # IM > im_target). Use a smooth approximation:
                    # contribute proportional to phi(eps_c) where
                    # eps_c >= eps_im - 0.5*dEps (half-bin tolerance).
                    if eps_c >= eps_im - 0.5 * dEps:
                        cube[i_M, r_bin, i_eps] += (
                            nu * f_M * dM * w * p_eps
                        )
    total = cube.sum()
    if total <= 0:
        raise RuntimeError(
            "Deaggregation: no source scenarios produced contributions "
            "at this IM level -- check that im_target is achievable."
        )
    probs = cube / total
    # Marginal means
    p_M = probs.sum(axis=(1, 2))
    p_R = probs.sum(axis=(0, 2))
    p_eps = probs.sum(axis=(0, 1))
    mean_M = float((M_mids * p_M).sum())
    mean_R = float((R_mids * p_R).sum())
    mean_eps = float((eps_mids * p_eps).sum())
    # Modal (joint argmax)
    idx = np.unravel_index(np.argmax(probs), probs.shape)
    modal_M = float(M_mids[idx[0]])
    modal_R = float(R_mids[idx[1]])
    modal_eps = float(eps_mids[idx[2]])
    return DeaggregationResult(
        im_target=float(im_target),
        period=float(gmpe.T),
        contributions=probs,
        M_edges=M_edges.copy(),
        R_edges=R_edges.copy(),
        eps_edges=eps_edges.copy(),
        mean_M=mean_M, mean_R=mean_R, mean_eps=mean_eps,
        modal_M=modal_M, modal_R=modal_R, modal_eps=modal_eps,
    )
