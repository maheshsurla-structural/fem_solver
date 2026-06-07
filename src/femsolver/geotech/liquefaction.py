"""Liquefaction-triggering analysis -- Idriss & Boulanger (2014).

Implements the deterministic simplified procedure for evaluating
the factor of safety against soil liquefaction:

    FS = CRR · MSF · K_sigma · K_alpha / CSR,

where

* ``CSR`` (cyclic stress ratio) is the seismic demand on the soil
  element at depth ``z``;
* ``CRR`` (cyclic resistance ratio) is the soil's resistance, derived
  from SPT ``N_60`` or CPT ``q_c`` data corrected for fines content;
* ``MSF`` is the magnitude scaling factor (converts the curve to
  the design earthquake's magnitude);
* ``K_sigma`` is the overburden correction;
* ``K_alpha`` is the sloping-ground correction (= 1.0 for level
  ground).

The Idriss & Boulanger 2014 update refined the CRR-N_60 correlation
and MSF formulation; this module implements those forms.

References
----------
* Idriss, I.M. & Boulanger, R.W. (2008, updated 2014). *Soil
  Liquefaction During Earthquakes*. EERI Monograph MNO-12.
* Youd, T.L. et al. (2001). "Liquefaction Resistance of Soils:
  Summary Report from the 1996 NCEER and 1998 NCEER/NSF Workshops."
  *J. Geotech. Geoenv. Eng.*, 127(10).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ============================================================ CSR

def stress_reduction_coefficient(*, z: float, M: float) -> float:
    """Stress-reduction coefficient ``r_d`` per Idriss 1999 / I&B 2014::

        ln(r_d) = alpha(z) + beta(z) · M
        alpha(z) = -1.012 - 1.126 · sin(z/11.73 + 5.133)
        beta(z)  =  0.106 + 0.118 · sin(z/11.28 + 5.142)

    valid for z <= 34 m and 5.5 <= M <= 8.5.
    """
    if z < 0.0:
        raise ValueError("z must be >= 0")
    alpha = -1.012 - 1.126 * math.sin(z / 11.73 + 5.133)
    beta_z = 0.106 + 0.118 * math.sin(z / 11.28 + 5.142)
    return float(math.exp(alpha + beta_z * M))


def cyclic_stress_ratio(
    *,
    a_max_g: float,
    sigma_v_total: float, sigma_v_eff: float,
    z: float, M: float,
) -> float:
    """CSR per Seed-Idriss 1971 with the Idriss r_d profile.

    ``CSR = 0.65 · (a_max / g) · (sigma_v / sigma_v_eff) · r_d``

    Parameters
    ----------
    a_max_g : float
        Peak ground acceleration in units of ``g`` (e.g. 0.30).
    sigma_v_total : float
        Total vertical stress at depth z (Pa).
    sigma_v_eff : float
        Effective vertical stress at depth z (Pa).
    z : float
        Depth below ground surface (m).
    M : float
        Earthquake moment magnitude.
    """
    if a_max_g <= 0.0:
        raise ValueError("a_max_g must be > 0")
    if sigma_v_eff <= 0.0:
        raise ValueError("sigma_v_eff must be > 0")
    r_d = stress_reduction_coefficient(z=z, M=M)
    return float(0.65 * a_max_g * sigma_v_total / sigma_v_eff * r_d)


# ============================================================ CRR from SPT

def fines_content_correction(
    *,
    N_60: float, FC_percent: float,
) -> float:
    """``Delta N_60`` correction for fines content (I&B 2008 Eq. 2.18b).

    Parameters
    ----------
    N_60 : float
        Corrected SPT N-value at 60% energy efficiency.
    FC_percent : float
        Fines-content percentage (passing #200 sieve).
    """
    if N_60 < 0.0:
        raise ValueError("N_60 must be >= 0")
    if FC_percent < 0.0 or FC_percent > 100.0:
        raise ValueError("FC_percent must be in [0, 100]")
    return float(math.exp(
        1.63 + 9.7 / max(FC_percent + 0.01, 1e-3)
        - (15.7 / max(FC_percent + 0.01, 1e-3)) ** 2
    ))


def CRR_from_N1_60cs(N1_60cs: float) -> float:
    """CRR_{7.5, 1 atm} from clean-sand-equivalent (N_1)_{60cs}, per
    Idriss & Boulanger 2014 Eq. 70:

        CRR = exp(N/14.1 + (N/126)^2 - (N/23.6)^3 + (N/25.4)^4 - 2.8)

    Valid for ``N1_60cs <= 37``.  Above 37 the soil is considered
    non-liquefiable for practical purposes (CRR clipped to a high
    value).
    """
    if N1_60cs < 0.0:
        raise ValueError("N1_60cs must be >= 0")
    N = min(N1_60cs, 37.0)
    return float(math.exp(
        N / 14.1
        + (N / 126.0) ** 2
        - (N / 23.6) ** 3
        + (N / 25.4) ** 4
        - 2.8
    ))


# ============================================================ MSF and K_sigma

def magnitude_scaling_factor(M: float) -> float:
    """MSF per Idriss & Boulanger 2014 Eq. 71::

        MSF = min(1 + (M_SF_max - 1) * (8.64 * exp(-M/4) - 1.325), 1.8)
        M_SF_max = 1.8

    Returns 1.0 at M = 7.5 and rises (more demand-relief) for smaller
    magnitudes.
    """
    M_SF_max = 1.8
    msf = 1.0 + (M_SF_max - 1.0) * (8.64 * math.exp(-M / 4.0) - 1.325)
    return float(max(min(msf, M_SF_max), 0.5))


def K_sigma(*, sigma_v_eff: float, N1_60cs: float | None = None) -> float:
    """Overburden correction ``K_sigma`` per I&B 2014 Eq. 73::

        K_sigma = 1 - C_sigma · ln(sigma_v_eff / Pa)

    with ``C_sigma = 1 / (18.9 - 2.55 sqrt(N1_60cs))``, clamped to
    [0, 0.3]. ``Pa = 101.3 kPa`` is atmospheric pressure.

    Parameters
    ----------
    sigma_v_eff : float
        Effective vertical stress (Pa).
    N1_60cs : float, optional
        Clean-sand equivalent N_1. If omitted, default ``C_sigma = 0.1``
        is used (conservative for typical soils).
    """
    if sigma_v_eff <= 0.0:
        raise ValueError("sigma_v_eff must be > 0")
    Pa = 101.3e3
    if N1_60cs is None:
        C_sigma = 0.1
    else:
        C_sigma = 1.0 / max(18.9 - 2.55 * math.sqrt(N1_60cs), 0.1)
        C_sigma = min(max(C_sigma, 0.0), 0.3)
    K_s = 1.0 - C_sigma * math.log(sigma_v_eff / Pa)
    return float(max(min(K_s, 1.1), 0.5))


# ============================================================ factor of safety

@dataclass
class LiquefactionTriggeringResult:
    """Full liquefaction-triggering evaluation at a single depth."""

    z: float
    CSR: float
    CRR: float
    MSF: float
    K_sigma: float
    K_alpha: float
    CRR_corrected: float
    FS: float
    liquefies: bool                # True if FS < 1


def evaluate_liquefaction(
    *,
    z: float, M: float, a_max_g: float,
    sigma_v_total: float, sigma_v_eff: float,
    N_60: float, FC_percent: float = 0.0,
    K_alpha: float = 1.0,
    FS_threshold: float = 1.0,
) -> LiquefactionTriggeringResult:
    """Evaluate FS against liquefaction at a single depth.

    Pipeline::

        CSR  = cyclic_stress_ratio(...)
        N_1_60cs = N_60 + Delta_N(FC)
        CRR  = CRR_from_N1_60cs(N_1_60cs)
        MSF  = magnitude_scaling_factor(M)
        K_s  = K_sigma(sigma_v_eff, N_1_60cs)
        FS   = CRR · MSF · K_s · K_alpha / CSR
    """
    csr = cyclic_stress_ratio(
        a_max_g=a_max_g,
        sigma_v_total=sigma_v_total,
        sigma_v_eff=sigma_v_eff,
        z=z, M=M,
    )
    # Fines correction
    if FC_percent > 5.0:
        dN = fines_content_correction(N_60=N_60, FC_percent=FC_percent)
    else:
        dN = 0.0
    N1_60cs = N_60 + dN
    crr_75 = CRR_from_N1_60cs(N1_60cs)
    msf = magnitude_scaling_factor(M)
    k_s = K_sigma(sigma_v_eff=sigma_v_eff, N1_60cs=N1_60cs)
    CRR_corr = crr_75 * msf * k_s * K_alpha
    FS = CRR_corr / csr if csr > 0 else float("inf")
    return LiquefactionTriggeringResult(
        z=float(z), CSR=float(csr), CRR=float(crr_75),
        MSF=float(msf), K_sigma=float(k_s), K_alpha=float(K_alpha),
        CRR_corrected=float(CRR_corr),
        FS=float(FS),
        liquefies=bool(FS < FS_threshold),
    )
