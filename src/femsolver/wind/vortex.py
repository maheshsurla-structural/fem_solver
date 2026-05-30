"""Vortex shedding -- Strouhal frequency, Scruton number, lock-in.

When wind flows past a bluff body, alternating vortices shed
periodically at a frequency

    f_s = St * U / D

where ``St`` (the Strouhal number) is geometry-dependent: ~0.20 for
a circular cylinder in the sub-critical Reynolds range, ~0.10 for a
square section, ~0.08-0.15 for D-section bridge decks.

If ``f_s`` approaches the structure's natural frequency ``f_n`` and
the Scruton number is low, the vortex wake *locks in* to the
structure's natural motion and the resulting transverse vibration
can be much larger than the steady wind amplitude would suggest.

The classical lock-in criterion (CICIND, ESDU) is

    Sc = 2 * m_e * zeta / (rho * D^2) < ~10..20

with ``m_e`` the equivalent mass per unit length and ``zeta`` the
structural damping ratio. Lower Sc = greater lock-in risk.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StrouhalResult:
    """Vortex-shedding-frequency calculation."""

    f_s: float            # shedding frequency (Hz)
    St: float             # Strouhal number used
    U: float              # mean wind speed (m/s)
    D: float              # characteristic dimension (m)


def vortex_shedding_frequency(
    U: float,
    D: float,
    *,
    St: float = 0.20,
) -> StrouhalResult:
    """Strouhal frequency of vortex shedding past a cylinder of
    diameter ``D`` (m) under wind speed ``U`` (m/s).

    Typical Strouhal numbers:

    * ``St = 0.20`` -- circular cylinder, sub-critical Re (default)
    * ``St = 0.13`` -- square section
    * ``St = 0.11`` -- typical bridge-deck D-section
    """
    if U <= 0:
        raise ValueError(f"U must be positive, got {U}")
    if D <= 0:
        raise ValueError(f"D must be positive, got {D}")
    if St <= 0:
        raise ValueError(f"St must be positive, got {St}")
    return StrouhalResult(f_s=float(St * U / D), St=float(St),
                            U=float(U), D=float(D))


def scruton_number(
    *,
    m_e: float,
    zeta: float,
    D: float,
    rho: float = 1.25,
) -> float:
    """Scruton number ``Sc = 2 m_e zeta / (rho D^2)``.

    Parameters
    ----------
    m_e : float
        Equivalent mass per unit length (kg/m).
    zeta : float
        Damping ratio (fraction of critical).
    D : float
        Characteristic dimension (m).
    rho : float, default 1.25
        Air density (kg/m^3).
    """
    if m_e <= 0 or D <= 0 or rho <= 0:
        raise ValueError("m_e, D, rho must all be positive")
    if zeta < 0:
        raise ValueError(f"zeta must be non-negative, got {zeta}")
    return float(2.0 * m_e * zeta / (rho * D * D))


def is_lock_in_risk(
    *,
    f_s: float,
    f_n: float,
    Sc: float,
    bandwidth: float = 0.20,
    Sc_threshold: float = 10.0,
) -> bool:
    """Heuristic lock-in flag: True if the shedding frequency lies
    within ``bandwidth`` (default ±20%) of the natural frequency AND
    the Scruton number is below the threshold (default 10)."""
    if f_n <= 0 or f_s <= 0:
        raise ValueError("f_s and f_n must both be positive")
    if not 0.0 < bandwidth < 1.0:
        raise ValueError(f"bandwidth must be in (0, 1), got {bandwidth}")
    in_band = abs(f_s - f_n) / f_n <= bandwidth
    low_Sc = Sc < Sc_threshold
    return bool(in_band and low_Sc)
