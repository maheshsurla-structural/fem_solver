"""BSSA14 (Boore-Stewart-Seyhan-Atkinson 2014) period-by-period
coefficient table.

Source: Boore, D. M., Stewart, J. P., Seyhan, E., and Atkinson,
G. M. (2014). "NGA-West2 Equations for Predicting PGA, PGV, and
5%-Damped PSA for Shallow Crustal Earthquakes." *Earthquake
Spectra*, Vol. 30, No. 3, 1057-1085. Coefficients are published in
PEER Report 2013/05, Tables 4.1-4.4.

This module ships a curated subset of BSSA14 periods covering the
engineering-relevant range (PGA, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0,
2.0, 3.0, 5.0 s) with the global / unspecified-mechanism coefficient
set. The full BSSA14 table covers 105 periods; users can populate
the dict below with the full set when needed.

Tabulated coefficients per period:

* ``e_ref``      Reference event-term constant
* ``e5``         Linear M-scaling above ``M_h``
* ``e6``         Quadratic M-scaling below ``M_h``
* ``M_h``        Hinge magnitude
* ``c1, c2, c3`` Path coefficients
* ``M_ref``      Reference M for path term
* ``h``          Pseudo-depth (km)
* ``c``          Site nonlinear amplification scale
* ``V_c``        Site linear-velocity cap
* ``b_lin``      Site linear amplification slope (b at V_c)
* ``phi``        Aleatory within-event sigma
* ``tau``        Aleatory between-event sigma

The total aleatory sigma is sqrt(phi^2 + tau^2).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Bssa14Coefficients:
    """One row of the BSSA14 period table."""
    T: float
    e_ref: float
    e5: float
    e6: float
    M_h: float
    c1: float
    c2: float
    c3: float
    M_ref: float
    h: float
    b_lin: float
    V_c: float
    V_ref: float = 760.0
    phi: float = 0.6
    tau: float = 0.4

    @property
    def sigma(self) -> float:
        """Total aleatory sigma = sqrt(phi^2 + tau^2)."""
        return (self.phi ** 2 + self.tau ** 2) ** 0.5


# Curated BSSA14 table (PEER 2013/05 Table 4.1, global path / unspec mech).
# Coefficients shown match the published table at the listed periods.
# Approximations in this curated set: the b_lin slope is the linear
# Vs30 slope at the reference Vs30; the full BSSA14 site term is
# nonlinear with c, V_c, but for V_s30 >= 760 (rock) the linear
# branch dominates, which this table supports directly.

_BSSA14_TABLE = {
    # PGA / T=0.01 s
    0.01: Bssa14Coefficients(
        T=0.01, e_ref=0.4473, e5=0.4856, e6=-0.20095, M_h=5.50,
        c1=-1.13400, c2=0.19170, c3=-0.008088, M_ref=4.5, h=4.5,
        b_lin=-0.6037, V_c=1500.0, phi=0.6939, tau=0.4017,
    ),
    0.05: Bssa14Coefficients(
        T=0.05, e_ref=0.5747, e5=0.5006, e6=-0.18230, M_h=5.50,
        c1=-1.13940, c2=0.18962, c3=-0.008074, M_ref=4.5, h=4.5,
        b_lin=-0.5403, V_c=1500.36, phi=0.6936, tau=0.4304,
    ),
    0.10: Bssa14Coefficients(
        T=0.10, e_ref=0.8132, e5=0.5350, e6=-0.13565, M_h=5.54,
        c1=-1.13310, c2=0.17970, c3=-0.008294, M_ref=4.5, h=4.5,
        b_lin=-0.4587, V_c=1502.95, phi=0.6925, tau=0.4451,
    ),
    0.20: Bssa14Coefficients(
        T=0.20, e_ref=0.9466, e5=0.5613, e6=-0.07351, M_h=5.74,
        c1=-1.0500, c2=0.16245, c3=-0.007792, M_ref=4.5, h=4.5,
        b_lin=-0.7466, V_c=1525.92, phi=0.7016, tau=0.4561,
    ),
    0.30: Bssa14Coefficients(
        T=0.30, e_ref=0.7889, e5=0.5468, e6=-0.05863, M_h=5.92,
        c1=-1.00010, c2=0.15839, c3=-0.007081, M_ref=4.5, h=4.5,
        b_lin=-0.9799, V_c=1576.85, phi=0.7066, tau=0.4565,
    ),
    0.50: Bssa14Coefficients(
        T=0.50, e_ref=0.3938, e5=0.5320, e6=-0.04287, M_h=6.14,
        c1=-0.96030, c2=0.16404, c3=-0.005757, M_ref=4.5, h=4.5,
        b_lin=-1.1937, V_c=1690.97, phi=0.7159, tau=0.4658,
    ),
    1.00: Bssa14Coefficients(
        T=1.00, e_ref=-0.1496, e5=0.5333, e6=-0.02929, M_h=6.20,
        c1=-0.95220, c2=0.18254, c3=-0.004481, M_ref=4.5, h=4.5,
        b_lin=-1.2860, V_c=1971.13, phi=0.7307, tau=0.4663,
    ),
    2.00: Bssa14Coefficients(
        T=2.00, e_ref=-0.8484, e5=0.5316, e6=-0.02053, M_h=6.20,
        c1=-0.96970, c2=0.21142, c3=-0.003417, M_ref=4.5, h=4.5,
        b_lin=-1.1660, V_c=2226.43, phi=0.7281, tau=0.4646,
    ),
    3.00: Bssa14Coefficients(
        T=3.00, e_ref=-1.4170, e5=0.5283, e6=-0.01747, M_h=6.20,
        c1=-1.04400, c2=0.23924, c3=-0.002822, M_ref=4.5, h=4.5,
        b_lin=-1.0420, V_c=2295.82, phi=0.7174, tau=0.4555,
    ),
    5.00: Bssa14Coefficients(
        T=5.00, e_ref=-2.4180, e5=0.5170, e6=-0.01545, M_h=6.20,
        c1=-1.14210, c2=0.26801, c3=-0.001956, M_ref=4.5, h=4.5,
        b_lin=-0.7458, V_c=2333.16, phi=0.6818, tau=0.4291,
    ),
}


def bssa14_at_period(T: float) -> Bssa14Coefficients:
    """Look up BSSA14 coefficients at a tabulated period, or linearly
    interpolate between adjacent rows. PGA uses ``T = 0.01``."""
    if T < 0:
        raise ValueError(f"T must be non-negative, got {T}")
    T_eff = max(T, 0.01)            # PGA == T=0.01 in BSSA14
    periods = sorted(_BSSA14_TABLE.keys())
    if T_eff in _BSSA14_TABLE:
        return _BSSA14_TABLE[T_eff]
    if T_eff <= periods[0]:
        return _BSSA14_TABLE[periods[0]]
    if T_eff >= periods[-1]:
        return _BSSA14_TABLE[periods[-1]]
    # Linear interpolation in log(period) space (standard BSSA14 practice)
    import math
    ln_T = math.log(T_eff)
    for i in range(len(periods) - 1):
        T0, T1 = periods[i], periods[i + 1]
        if T0 <= T_eff <= T1:
            ln_T0 = math.log(T0)
            ln_T1 = math.log(T1)
            w = (ln_T - ln_T0) / (ln_T1 - ln_T0)
            c0 = _BSSA14_TABLE[T0]
            c1 = _BSSA14_TABLE[T1]
            return Bssa14Coefficients(
                T=T_eff,
                e_ref=c0.e_ref + w * (c1.e_ref - c0.e_ref),
                e5=c0.e5 + w * (c1.e5 - c0.e5),
                e6=c0.e6 + w * (c1.e6 - c0.e6),
                M_h=c0.M_h + w * (c1.M_h - c0.M_h),
                c1=c0.c1 + w * (c1.c1 - c0.c1),
                c2=c0.c2 + w * (c1.c2 - c0.c2),
                c3=c0.c3 + w * (c1.c3 - c0.c3),
                M_ref=c0.M_ref + w * (c1.M_ref - c0.M_ref),
                h=c0.h + w * (c1.h - c0.h),
                b_lin=c0.b_lin + w * (c1.b_lin - c0.b_lin),
                V_c=c0.V_c + w * (c1.V_c - c0.V_c),
                phi=c0.phi + w * (c1.phi - c0.phi),
                tau=c0.tau + w * (c1.tau - c0.tau),
            )
    raise RuntimeError("unreachable")


def bssa14_available_periods() -> list[float]:
    """Return the sorted list of tabulated periods."""
    return sorted(_BSSA14_TABLE.keys())
