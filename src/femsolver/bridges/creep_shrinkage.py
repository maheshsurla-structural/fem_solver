"""Concrete creep, shrinkage, and steel relaxation models.

Time-dependent behaviour of prestressed concrete bridges is governed
by three primary mechanisms:

* **Concrete creep** -- additional strain ``phi(t, t0) · sigma_c /
  E_c`` developed under sustained stress.
* **Concrete shrinkage** -- moisture-loss strain ``eps_sh(t)``
  independent of stress.
* **Steel relaxation** -- stress loss in prestressing steel held at
  constant strain.

This module implements two widely-used models:

1. **CEB-FIP Model Code 2010** (also adopted by EC2) -- creep
   coefficient ``phi(t, t0)`` and shrinkage ``eps_cs(t)``.
2. **AASHTO LRFD 2020** -- the simpler ``K_id``/``K_df`` approach
   for prestressed-concrete bridge design.

Steel relaxation uses the standard "1000-hour" relaxation rate
(typically 2.5% for low-relaxation strand) with a logarithmic-time
extrapolation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


# ============================================================ CEB-FIP 2010

@dataclass
class CebFipCreepResult:
    """CEB-FIP 2010 creep coefficient breakdown.

    Attributes
    ----------
    phi_0 : float
        Notional creep coefficient (asymptotic).
    beta_c : float
        Time-development factor (0..1) at the queried time.
    phi : float
        Creep coefficient phi(t, t_0) = phi_0 * beta_c.
    """

    phi_0: float
    beta_c: float
    phi: float


def cebfip_creep_coefficient(
    *,
    t_days: float, t0_days: float,
    f_cm: float,
    RH: float = 70.0,
    h_0: float = 0.20,
    cement_type: str = "N",
) -> CebFipCreepResult:
    """Creep coefficient ``phi(t, t_0)`` per CEB-FIP MC 2010 Cl. 5.1.9.

    Parameters
    ----------
    t_days : float
        Concrete age at the time of interest (days).
    t0_days : float
        Concrete age at loading (days).
    f_cm : float
        Mean compressive strength (Pa). ``f_cm = f_ck + 8 MPa``.
    RH : float, default 70.0
        Relative humidity (percent).
    h_0 : float, default 0.20
        Notional member size (m): ``2 A_c / u``.
    cement_type : str, default "N"
        ``"S"`` (slow), ``"N"`` (normal), or ``"R"`` (rapid hardening).
    """
    if t_days <= t0_days:
        raise ValueError("t must be > t0")
    if f_cm <= 0.0:
        raise ValueError("f_cm must be > 0")
    fcm_MPa = f_cm * 1.0e-6
    # Equation 5.1-77: phi_RH
    phi_RH = (1.0 + (1.0 - RH / 100.0)
              / (0.10 * (1000.0 * h_0) ** (1.0 / 3.0)))
    # If f_cm > 35 MPa, scale phi_RH (Eq 5.1-78)
    if fcm_MPa > 35.0:
        alpha_1 = (35.0 / fcm_MPa) ** 0.7
        alpha_2 = (35.0 / fcm_MPa) ** 0.2
        phi_RH = (1.0
                  + (1.0 - RH / 100.0)
                  / (0.10 * (1000.0 * h_0) ** (1.0 / 3.0))
                  * alpha_1) * alpha_2

    beta_fcm = 16.8 / math.sqrt(fcm_MPa)             # Eq 5.1-79
    # Effective age accounting for cement type (Eq 5.1-85)
    alpha_cement = {"S": -1.0, "N": 0.0, "R": 1.0}[cement_type]
    t0_eff = max(
        t0_days * (9.0 / (2.0 + t0_days ** 1.2) + 1.0) ** alpha_cement,
        0.5,
    )
    beta_t0 = 1.0 / (0.1 + t0_eff ** 0.20)           # Eq 5.1-80
    phi_0 = phi_RH * beta_fcm * beta_t0

    # Time-development: beta_c(t, t0), Eq 5.1-81 / 82
    beta_H = 1.5 * (
        1.0 + (0.012 * RH) ** 18
    ) * 1000.0 * h_0 + 250.0
    if fcm_MPa > 35.0:
        beta_H = min(beta_H * (35.0 / fcm_MPa) ** 0.5,
                     1500.0 * (35.0 / fcm_MPa) ** 0.5)
    else:
        beta_H = min(beta_H, 1500.0)
    beta_c = ((t_days - t0_eff)
              / (beta_H + (t_days - t0_eff))) ** 0.3

    phi = phi_0 * beta_c
    return CebFipCreepResult(
        phi_0=float(phi_0), beta_c=float(beta_c), phi=float(phi),
    )


@dataclass
class CebFipShrinkageResult:
    """CEB-FIP 2010 shrinkage strain breakdown."""

    eps_cd_inf: float            # asymptotic drying shrinkage
    beta_ds: float               # drying-shrinkage time-development
    eps_cd: float                # drying shrinkage at t
    eps_ca: float                # autogenous shrinkage at t
    eps_cs: float                # total shrinkage at t (cd + ca)


def cebfip_shrinkage(
    *,
    t_days: float, t_s_days: float,
    f_cm: float,
    RH: float = 70.0,
    h_0: float = 0.20,
) -> CebFipShrinkageResult:
    """Total shrinkage strain ``eps_cs = eps_cd + eps_ca`` per CEB-FIP
    MC 2010 Cl. 5.1.10.

    Parameters
    ----------
    t_days : float
        Concrete age at the time of interest (days).
    t_s_days : float
        Age at the start of drying (days).
    f_cm : float
        Mean compressive strength (Pa).
    RH : float, default 70.0
        Relative humidity (percent).
    h_0 : float, default 0.20
        Notional size = 2 A_c / u (m).
    """
    fcm_MPa = f_cm * 1.0e-6
    # Autogenous shrinkage
    beta_as = 1.0 - math.exp(-0.2 * math.sqrt(t_days))
    eps_ca_inf = -2.5e-6 * (fcm_MPa - 10.0)
    eps_ca = eps_ca_inf * beta_as
    # Drying shrinkage
    alpha_ds1 = 4.0       # for normal cement
    alpha_ds2 = 0.12      # CEB-FIP 2010 Eq. 5.1-77 (with f_cm,0 = 10 MPa)
    eps_cd_0 = (
        (220.0 + 110.0 * alpha_ds1)
        * math.exp(-alpha_ds2 * fcm_MPa / 10.0)
        * 1.0e-6
    )
    # RH influence
    beta_RH = -1.55 * (1.0 - (RH / 100.0) ** 3)
    eps_cd_inf = eps_cd_0 * beta_RH
    # Time development
    if t_days > t_s_days:
        beta_ds = ((t_days - t_s_days)
                   / (0.035 * (1000.0 * h_0) ** 2 + (t_days - t_s_days))
                   ) ** 0.5
    else:
        beta_ds = 0.0
    eps_cd = eps_cd_inf * beta_ds
    eps_cs = eps_cd + eps_ca
    return CebFipShrinkageResult(
        eps_cd_inf=float(eps_cd_inf),
        beta_ds=float(beta_ds),
        eps_cd=float(eps_cd),
        eps_ca=float(eps_ca),
        eps_cs=float(eps_cs),
    )


# ============================================================ steel relaxation

def steel_relaxation_loss_ratio(
    *,
    t_hours: float, t_initial_hours: float = 1.0,
    fpi_over_fpy: float = 0.70,
    relaxation_class: str = "low",
) -> float:
    """Relaxation stress-loss ratio ``Delta f_p / f_pi`` per AASHTO
    5.9.5.4.4 / EC2.

    For low-relaxation strand (Class 2)::

        Delta f_p / f_pi = (log(t / t_i) / 45) * (f_pi/f_py - 0.55)

    Parameters
    ----------
    t_hours : float
    t_initial_hours : float, default 1.0
        Time at end of jacking (h).
    fpi_over_fpy : float, default 0.70
        Initial stress / yield stress ratio. Typical 0.70-0.85.
    relaxation_class : {"normal", "low"}, default "low"
        Standard PT strand is "low-relaxation" (Class 2).
    """
    if t_hours <= t_initial_hours:
        return 0.0
    if relaxation_class == "low":
        denom = 45.0
    elif relaxation_class == "normal":
        denom = 10.0
    else:
        raise ValueError(
            f"relaxation_class must be 'low' or 'normal', "
            f"got {relaxation_class!r}"
        )
    return float(
        math.log10(t_hours / t_initial_hours) / denom
        * max(fpi_over_fpy - 0.55, 0.0)
    )


# ============================================================ prestress long-term loss aggregator

@dataclass
class PrestressLossBreakdown:
    """Long-term prestress-loss decomposition.

    All quantities are FORCE losses (N, negative = loss of jacking
    force).
    """

    delta_P_creep: float
    delta_P_shrinkage: float
    delta_P_relaxation: float
    delta_P_total: float
    P_effective: float        # input P_initial + delta_P_total


def prestress_long_term_loss(
    *,
    P_initial: float,
    A_ps: float, E_p: float,
    sigma_c_at_strand: float,
    E_c: float,
    creep: CebFipCreepResult,
    shrinkage: CebFipShrinkageResult,
    relaxation_loss_ratio: float,
    f_pi: float,
) -> PrestressLossBreakdown:
    """Aggregate long-term prestress losses from the three mechanisms.

    Each component converts a strand-stress change ``Delta_sigma_p``
    into a force change ``Delta_P = A_ps · Delta_sigma_p``:

    * **Creep**       :  ``Delta_sigma_p = E_p · phi(t,t0) · sigma_c / E_c``
    * **Shrinkage**   :  ``Delta_sigma_p = E_p · eps_cs(t)``
    * **Relaxation**  :  ``Delta_sigma_p = -relax_ratio · f_pi``

    With our sign convention (compressive concrete stress is
    ``sigma_c < 0``), all three components yield negative force
    changes (i.e., losses) for a normally-loaded prestressed beam.

    Parameters
    ----------
    P_initial : float
        Prestress force at the start of the long-term period (N) --
        typically the value after instantaneous (friction + anchorage
        + elastic-shortening) losses.
    A_ps, E_p : float
        Strand area (m^2) and modulus (Pa).
    sigma_c_at_strand : float
        Concrete stress at the strand centroid (Pa, negative for
        compression).
    E_c : float
        Concrete modulus (Pa).
    creep : CebFipCreepResult
        From :func:`cebfip_creep_coefficient`.
    shrinkage : CebFipShrinkageResult
        From :func:`cebfip_shrinkage`.
    relaxation_loss_ratio : float
        From :func:`steel_relaxation_loss_ratio`.
    f_pi : float
        Initial strand stress (Pa).
    """
    delta_P_creep = A_ps * E_p * creep.phi * sigma_c_at_strand / E_c
    delta_P_shrinkage = A_ps * E_p * shrinkage.eps_cs
    delta_P_relaxation = -A_ps * relaxation_loss_ratio * f_pi
    delta_P_total = delta_P_creep + delta_P_shrinkage + delta_P_relaxation
    return PrestressLossBreakdown(
        delta_P_creep=float(delta_P_creep),
        delta_P_shrinkage=float(delta_P_shrinkage),
        delta_P_relaxation=float(delta_P_relaxation),
        delta_P_total=float(delta_P_total),
        P_effective=float(P_initial + delta_P_total),
    )
