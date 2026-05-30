"""Ground Motion Prediction Equations (GMPEs).

A GMPE predicts the *median* spectral acceleration ``Sa(T, M, R,
V_s30)`` and the *aleatory standard deviation* ``sigma_lnSa`` for a
given earthquake scenario at a given site.

This module provides a single, **parameterised** Boore-Atkinson-style
functional form that captures the salient scaling behaviour:

* **Magnitude scaling** -- low-magnitude linear + saturating
  quadratic term around a hinge magnitude ``M_h``.
* **Distance attenuation** -- geometric + anelastic on R_jb (Joyner-
  Boore distance, km).
* **Site amplification** -- linear in ``ln(V_s30 / V_ref)``.

The supplied default coefficients give realistic PGA scaling for
crustal earthquakes (Western US-like). To use a specific published
GMPE (BSSA14 / CB14 / ASK14), instantiate :class:`BooreAtkinsonLike`
with the published period-by-period coefficients.

References
----------
Boore D., Atkinson G. (2008). "Ground-motion prediction equations
for the average horizontal component of PGA, PGV, and 5%-damped PSA
at spectral periods between 0.01 s and 10.0 s." *Earthq. Spectra*
24(1), 99-138.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class GmpeResult:
    """Median Sa and aleatory sigma at one (T, M, R, V_s30) scenario."""

    median_Sa: float     # in units of g
    median_lnSa: float   # ln(median_Sa)
    sigma_lnSa: float    # aleatory standard deviation of ln Sa


class BooreAtkinsonLike:
    """Parameterised Boore-Atkinson-2008-style GMPE.

    Parameters
    ----------
    T : float
        Spectral period (s). Use ``T = 0.0`` for PGA.
    e1, e5, e6 : float
        Magnitude-scaling coefficients.
    M_h : float
        Hinge magnitude (typical 6.75 for crustal events).
    c1, c2, c3 : float
        Geometric/anelastic attenuation coefficients.
    M_ref : float
        Reference magnitude used in c2*(M - M_ref) ln R term.
    R_ref : float
        Reference distance (km).
    h : float
        Pseudo-depth (km) used in ``R = sqrt(R_jb^2 + h^2)``.
    b_lin : float
        Site-amplification slope on ``ln(V_s30 / V_ref)``.
    V_ref : float
        Reference site shear-wave velocity (m/s), typical 760.
    sigma : float
        Aleatory standard deviation of ``ln Sa``.

    The default values below give a *generic crustal-event PGA*
    relationship suitable for unit testing and the first PSHA
    examples. **Replace with published coefficients for real
    analyses.**
    """

    def __init__(
        self,
        T: float = 0.0,
        *,
        e1: float = -0.6,
        e5: float = 0.5,
        e6: float = -0.05,
        M_h: float = 6.75,
        c1: float = -1.4,
        c2: float = 0.1,
        c3: float = -0.0015,
        M_ref: float = 4.5,
        R_ref: float = 1.0,
        h: float = 5.0,
        b_lin: float = -0.6,
        V_ref: float = 760.0,
        sigma: float = 0.60,
    ):
        if T < 0:
            raise ValueError(f"T must be non-negative, got {T}")
        if M_h <= 0 or M_ref <= 0:
            raise ValueError("M_h, M_ref must be positive")
        if R_ref <= 0 or h < 0 or V_ref <= 0:
            raise ValueError("R_ref, V_ref must be positive; h non-negative")
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        self.T = float(T)
        self.e1 = float(e1)
        self.e5 = float(e5)
        self.e6 = float(e6)
        self.M_h = float(M_h)
        self.c1 = float(c1)
        self.c2 = float(c2)
        self.c3 = float(c3)
        self.M_ref = float(M_ref)
        self.R_ref = float(R_ref)
        self.h = float(h)
        self.b_lin = float(b_lin)
        self.V_ref = float(V_ref)
        self.sigma = float(sigma)

    def evaluate(
        self,
        *,
        M: float,
        R_jb: float,
        V_s30: float = 760.0,
    ) -> GmpeResult:
        """Predict median ``Sa`` (in g) at the scenario.

        Parameters
        ----------
        M : float
            Moment magnitude.
        R_jb : float
            Joyner-Boore distance (km).
        V_s30 : float
            Time-averaged shear-wave velocity in the upper 30 m
            (m/s).
        """
        if M <= 0:
            raise ValueError(f"M must be positive, got {M}")
        if R_jb < 0:
            raise ValueError(f"R_jb must be non-negative, got {R_jb}")
        if V_s30 <= 0:
            raise ValueError(f"V_s30 must be positive, got {V_s30}")
        # Magnitude scaling
        if M <= self.M_h:
            f_M = self.e1 + self.e5 * (M - self.M_h) \
                  + self.e6 * (M - self.M_h) ** 2
        else:
            f_M = self.e1 + self.e5 * (M - self.M_h)
        # Distance term (with pseudo-depth h)
        R = math.sqrt(R_jb * R_jb + self.h * self.h)
        f_R = (self.c1 + self.c2 * (M - self.M_ref)) \
              * math.log(R / self.R_ref) + self.c3 * (R - self.R_ref)
        # Linear site term
        f_S = self.b_lin * math.log(V_s30 / self.V_ref)
        ln_Sa = f_M + f_R + f_S
        return GmpeResult(
            median_Sa=float(math.exp(ln_Sa)),
            median_lnSa=float(ln_Sa),
            sigma_lnSa=float(self.sigma),
        )
