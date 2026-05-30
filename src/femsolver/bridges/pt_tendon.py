"""Post-tensioned tendon: profile, friction/wobble losses, anchorage
slip, and equivalent prestress load.

A post-tensioned (PT) tendon is a high-strength steel cable that is
stressed AFTER the concrete has hardened. The jacking force ``P_0``
is applied at one (or both) ends, but is attenuated along the cable
by:

* **Curvature friction** (``mu * alpha``) -- friction between the
  cable and the duct curving through the structure.
* **Wobble friction** (``k * x``) -- unintended wavy deviations of
  the duct, linear in the path length ``x``.
* **Anchorage slip** -- as the wedges seat, the tendon shortens at
  the anchorage by an amount ``Delta_a``; the prestress redistributes
  over an "affected length" ``l_a`` near the jacking end.

The tendon's friction loss formula (AASHTO LRFD 5.9.5.2; IS 1343
Cl. 18.5.2):

    P(x) = P_0 * exp(-(mu * alpha(x) + k * x))

This module provides:

* :class:`TendonProfile` -- piecewise straight + circular-arc
  geometry (the canonical PT idealisation).
* :func:`friction_loss` -- closed-form ``P(x) / P_0`` for the
  profile.
* :func:`anchorage_slip_loss` -- effective length and prestress
  redistribution from anchor slip.
* :func:`equivalent_prestress_load` -- distributed load on the
  structure equivalent to the prestress (the "balanced load" method).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ============================================================ tendon profile

@dataclass
class TendonProfile:
    """A piecewise tendon profile of ``n`` segments.

    Each segment is either a straight or a circular arc; we capture
    only the cumulative angle change ``alpha(x)`` (the only friction-
    relevant geometry) and the length.

    Attributes
    ----------
    segment_lengths : np.ndarray
        Length of each segment (m).
    segment_dalpha : np.ndarray
        Angle subtended by each segment (rad). For straight segments,
        ``dalpha = 0``; for circular arcs, ``dalpha = arc_length /
        radius``.
    """

    segment_lengths: np.ndarray
    segment_dalpha: np.ndarray

    def __post_init__(self) -> None:
        self.segment_lengths = np.asarray(self.segment_lengths,
                                            dtype=float).ravel()
        self.segment_dalpha = np.asarray(self.segment_dalpha,
                                           dtype=float).ravel()
        if self.segment_lengths.shape != self.segment_dalpha.shape:
            raise ValueError(
                "segment_lengths and segment_dalpha must have same shape"
            )
        if np.any(self.segment_lengths < 0.0):
            raise ValueError("segment_lengths must be >= 0")
        if np.any(self.segment_dalpha < 0.0):
            raise ValueError("segment_dalpha (absolute) must be >= 0")

    @property
    def total_length(self) -> float:
        return float(np.sum(self.segment_lengths))

    def cumulative_x_and_alpha(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(x, alpha)`` at segment ENDPOINTS (n+1 points,
        starting at the jacking end with x=0, alpha=0)."""
        x = np.concatenate([[0.0], np.cumsum(self.segment_lengths)])
        a = np.concatenate([[0.0], np.cumsum(self.segment_dalpha)])
        return x, a


def parabolic_drape_profile(
    *,
    L: float, drape: float,
    n_segments: int = 20,
) -> TendonProfile:
    """Approximate a parabolic tendon profile y = -4·drape·x·(L-x)/L^2
    with ``n_segments`` equal-length straight chords. Computes the
    cumulative angle change between consecutive chord directions.

    Parameters
    ----------
    L : float
        Span length (m).
    drape : float
        Maximum sag of the tendon below the centroid at mid-span (m,
        positive).
    n_segments : int, default 20
    """
    if L <= 0.0:
        raise ValueError("L must be > 0")
    if drape <= 0.0:
        raise ValueError("drape must be > 0")
    if n_segments < 4:
        raise ValueError("need >= 4 segments to capture the curve")
    x_nodes = np.linspace(0.0, L, n_segments + 1)
    y_nodes = -4.0 * drape * x_nodes * (L - x_nodes) / L ** 2
    dx = np.diff(x_nodes)
    dy = np.diff(y_nodes)
    seg_len = np.hypot(dx, dy)
    # Direction angle of each chord
    theta = np.arctan2(dy, dx)
    # Cumulative deviation magnitude
    dalpha = np.zeros_like(seg_len)
    dalpha[1:] = np.abs(np.diff(theta))
    return TendonProfile(segment_lengths=seg_len,
                          segment_dalpha=dalpha)


# ============================================================ friction loss

@dataclass
class FrictionLossResult:
    """Friction + wobble loss along the tendon.

    Attributes
    ----------
    x : np.ndarray
        Distance from the jacking end at each segment endpoint (m).
    P_over_P0 : np.ndarray
        Force ratio P(x) / P_0 at each endpoint.

    Notes
    -----
    The cumulative angle change ``alpha(x)`` is available from the
    underlying profile via :meth:`TendonProfile.cumulative_x_and_alpha`.
    """

    x: np.ndarray
    P_over_P0: np.ndarray


def friction_loss(
    profile: TendonProfile,
    *,
    mu: float = 0.20,
    k: float = 0.0066,
) -> FrictionLossResult:
    """Friction-loss profile P(x)/P_0 = exp(-(mu·alpha + k·x)).

    Parameters
    ----------
    mu : float, default 0.20
        Curvature friction coefficient (typical 0.15-0.30 for
        seven-wire strand in galvanised duct).
    k : float, default 0.0066 /m
        Wobble (length-effect) coefficient (typical 0.001-0.015 /m).
    """
    if mu < 0.0 or k < 0.0:
        raise ValueError("mu and k must be >= 0")
    x, alpha = profile.cumulative_x_and_alpha()
    P_over_P0 = np.exp(-(mu * alpha + k * x))
    return FrictionLossResult(x=x, P_over_P0=P_over_P0)


# ============================================================ anchorage slip

@dataclass
class AnchorageSlipResult:
    """Anchorage-slip redistribution at the jacking end.

    Attributes
    ----------
    l_a : float
        Length over which the slip is absorbed (m).
    P0_after_seating : float
        Force at the jacking end AFTER seating (N).
    P_profile : np.ndarray
        Force at each sample point along the tendon AFTER seating (N).
    x_sample : np.ndarray
    """

    l_a: float
    P0_after_seating: float
    P_profile: np.ndarray
    x_sample: np.ndarray


def anchorage_slip_loss(
    profile: TendonProfile,
    *,
    P_0: float,
    mu: float = 0.20,
    k: float = 0.0066,
    slip: float = 0.006,
    E_s: float = 1.95e11,
    A_ps: float,
) -> AnchorageSlipResult:
    """Anchorage-slip prestress loss (after-seating profile).

    The seating slip ``Delta_a`` at the anchor causes the tendon to
    shorten by that much; the friction loss curve is mirrored about
    the friction-loss-slope line over the affected length ``l_a``,
    where::

        Delta_a = (1 / (E_s A_ps)) * integral over l_a of (P_jack(x) - P_after(x)) dx
                ≈ (l_a · DP_loss(l_a)) / (E_s A_ps)         (small-loss approx)

    A simplified analytical solution (assuming constant friction
    slope p = P_0 · (mu·alpha'(0) + k)) gives::

        l_a = sqrt(Delta_a · E_s · A_ps / p)

    Force after seating is then::

        P_after(x) = P_jack(2·l_a - x) for 0 <= x <= l_a
                    = P_jack(x)         for x > l_a
    """
    fric = friction_loss(profile, mu=mu, k=k)
    x = fric.x
    P_jack = P_0 * fric.P_over_P0

    # Friction-loss slope near the anchor (use the first segment).
    if len(x) < 2:
        raise ValueError("profile must have at least 1 segment")
    p_slope = (P_0 - P_jack[1]) / (x[1] - x[0])     # N / m
    if p_slope <= 0.0:
        # No friction loss at all -- slip is absorbed only at the
        # anchor; effective l_a -> 0.
        l_a = 0.0
        P_after = P_jack.copy()
        return AnchorageSlipResult(
            l_a=0.0, P0_after_seating=float(P_jack[0]),
            P_profile=P_after, x_sample=x,
        )
    l_a = float(np.sqrt(slip * E_s * A_ps / p_slope))
    l_a = min(l_a, x[-1])

    # Mirror the friction curve about its slope line over [0, l_a]
    # to produce the post-seating curve.
    # P_after(x) = P_jack(x) - 2*p_slope*(l_a - x) for x in [0, l_a]
    P_after = P_jack.copy()
    in_la = x <= l_a
    P_after[in_la] = P_jack[in_la] - 2.0 * p_slope * (l_a - x[in_la])

    return AnchorageSlipResult(
        l_a=l_a,
        P0_after_seating=float(P_after[0]),
        P_profile=P_after,
        x_sample=x,
    )


# ============================================================ equivalent prestress load

def equivalent_uniform_load_parabolic(
    *,
    P: float, drape: float, L: float,
) -> float:
    """Equivalent upward UDL from a parabolic tendon profile.

    For a parabolic drape ``y = -4·drape·x·(L-x)/L^2`` with constant
    prestress ``P`` along the tendon, the equivalent upward distributed
    load on the beam is::

        w_eq = 8 P drape / L^2

    Parameters
    ----------
    P : float
        Effective prestress force (N), assumed constant along span.
    drape : float
        Maximum tendon eccentricity below the section centroid (m).
    L : float
        Span length (m).
    """
    if P <= 0.0:
        raise ValueError("P must be > 0")
    if drape <= 0.0:
        raise ValueError("drape must be > 0")
    if L <= 0.0:
        raise ValueError("L must be > 0")
    return float(8.0 * P * drape / L ** 2)
