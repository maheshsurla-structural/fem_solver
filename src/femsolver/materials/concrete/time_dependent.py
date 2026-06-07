"""Time-dependent concrete properties: strength, modulus, tensile-strength
gain with age, per EN 1992-1-1 and ACI 209R-92.

Concrete keeps maturing after the 28-day reference: its compressive
strength, elastic modulus, and tensile strength all rise with age (and
matter for transfer, staged construction, and long-term checks). This
module gives the **property-vs-age curves** -- the "compression
strength graph" -- that feed creep/loss/staged analyses.

Models
------
* **EN 1992-1-1 §3.1.2 / §3.1.3** -- ``β_cc(t) = exp{s[1 - √(28/t)]}``
  scales the mean strength; the modulus follows ``E_cm(t) =
  (f_cm(t)/f_cm)^0.3 · E_cm`` and the tensile strength ``f_ctm(t) =
  β_cc(t)^α · f_ctm`` (``α = 1`` before 28 d, ``2/3`` after).
* **ACI 209R-92** -- ``f_c(t) = t / (a + b·t) · f_c28`` with ``(a, b)``
  by cement type and curing; modulus ``E_c = 4700 √f'c`` (ACI 318 SI,
  normal-weight).

Creep φ(t,t₀) and shrinkage ε_cs(t) live in
:mod:`femsolver.bridges.creep_shrinkage` (CEB-FIP MC 2010); the
structural *effects* (shrinkage restraint forces, creep deflection,
applying shrinkage to a model) live in
:mod:`femsolver.analysis.time_dependent`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

_MPA = 1.0e6
_GPA = 1.0e9


# ============================================================ EN 1992-1-1

# §3.1.2(6): cement-class coefficient s
_EN_S = {"R": 0.20, "N": 0.25, "S": 0.38}


def en1992_beta_cc(t_days: float, *, cement_class: str = "N") -> float:
    """EN 1992-1-1 §3.1.2(6) age function ``β_cc(t) =
    exp{s·[1 - √(28/t)]}``.

    ``cement_class`` is ``"R"`` (rapid, s=0.20), ``"N"`` (normal,
    s=0.25), or ``"S"`` (slow, s=0.38). At ``t = 28`` d, ``β_cc = 1``.
    """
    if t_days <= 0:
        raise ValueError("t_days must be > 0")
    if cement_class not in _EN_S:
        raise ValueError("cement_class must be 'R', 'N', or 'S'")
    s = _EN_S[cement_class]
    return float(math.exp(s * (1.0 - math.sqrt(28.0 / t_days))))


def en1992_E_cm(f_cm: float) -> float:
    """EN 1992-1-1 Table 3.1 secant modulus ``E_cm = 22·(f_cm/10)^0.3``
    (GPa, with ``f_cm`` in MPa). Returns Pa."""
    f_cm_MPa = f_cm / _MPA
    return float(22.0 * (f_cm_MPa / 10.0) ** 0.3 * _GPA)


@dataclass
class ConcreteAgeProperties:
    """Concrete properties at a given age."""
    t_days: float
    beta_cc: float
    f_cm: float        # mean compressive strength (Pa)
    f_ck: float        # characteristic compressive strength (Pa)
    f_ctm: float       # mean tensile strength (Pa)
    E_cm: float        # secant modulus (Pa)
    code: str = ""


def en1992_strength_gain(
    t_days: float, *, f_ck_28: float, cement_class: str = "N",
) -> ConcreteAgeProperties:
    """EN 1992-1-1 time-dependent concrete properties at age ``t``.

    Parameters
    ----------
    t_days : float
        Concrete age (days).
    f_ck_28 : float
        28-day characteristic cylinder strength (Pa).
    cement_class : {"R", "N", "S"}

    Returns
    -------
    ConcreteAgeProperties with ``f_cm(t)``, ``f_ck(t)``, ``f_ctm(t)``,
    ``E_cm(t)``.
    """
    f_cm_28 = f_ck_28 + 8.0 * _MPA
    beta = en1992_beta_cc(t_days, cement_class=cement_class)
    f_cm_t = beta * f_cm_28
    f_ck_t = max(f_cm_t - 8.0 * _MPA, 0.0)
    # tensile strength: f_ctm(28) = 0.30 f_ck^(2/3) (<=C50); age exponent
    f_ck28_MPa = f_ck_28 / _MPA
    f_ctm_28 = (0.30 * f_ck28_MPa ** (2.0 / 3.0) if f_ck28_MPa <= 50.0
                else 2.12 * math.log(1.0 + (f_ck28_MPa + 8.0) / 10.0)) * _MPA
    alpha = 1.0 if t_days < 28.0 else 2.0 / 3.0
    f_ctm_t = beta ** alpha * f_ctm_28
    # modulus follows the 0.3-power of the strength ratio (§3.1.3(3))
    E_cm_28 = en1992_E_cm(f_cm_28)
    E_cm_t = (f_cm_t / f_cm_28) ** 0.3 * E_cm_28
    return ConcreteAgeProperties(
        t_days=float(t_days), beta_cc=float(beta),
        f_cm=float(f_cm_t), f_ck=float(f_ck_t),
        f_ctm=float(f_ctm_t), E_cm=float(E_cm_t), code="EN1992",
    )


# ============================================================ ACI 209R-92

# (a, b) by (cement_type, curing): f_c(t) = t/(a + b t) f_c28
_ACI209_AB = {
    ("I", "moist"): (4.0, 0.85),
    ("I", "steam"): (1.0, 0.95),
    ("III", "moist"): (2.3, 0.92),
    ("III", "steam"): (0.70, 0.98),
}


def aci209_strength_gain(
    t_days: float, *, f_c_28: float,
    cement_type: str = "I", curing: str = "moist",
) -> ConcreteAgeProperties:
    """ACI 209R-92 strength gain ``f_c(t) = t / (a + b·t) · f_c28`` with
    ``(a, b)`` from cement type (``"I"`` / ``"III"``) and curing
    (``"moist"`` / ``"steam"``).

    The modulus uses ACI 318 normal-weight ``E_c = 4700·√f'c`` (SI,
    MPa). ``f_ctm`` uses ``0.62·√f'c`` (ACI 318 modulus of rupture
    family, taken here as a tensile measure).
    """
    if t_days <= 0:
        raise ValueError("t_days must be > 0")
    key = (cement_type, curing)
    if key not in _ACI209_AB:
        raise ValueError(
            f"unsupported (cement_type, curing)={key}; "
            f"available: {sorted(_ACI209_AB)}"
        )
    a, b = _ACI209_AB[key]
    ratio = t_days / (a + b * t_days)
    f_c_t = ratio * f_c_28
    f_c_t_MPa = f_c_t / _MPA
    E_c_t = 4700.0 * math.sqrt(max(f_c_t_MPa, 0.0)) * _MPA
    f_ct_t = 0.62 * math.sqrt(max(f_c_t_MPa, 0.0)) * _MPA
    return ConcreteAgeProperties(
        t_days=float(t_days), beta_cc=float(ratio),
        f_cm=float(f_c_t), f_ck=float(f_c_t),     # ACI works in f'c
        f_ctm=float(f_ct_t), E_cm=float(E_c_t), code="ACI209",
    )


# ============================================================ curve helper

def strength_gain_curve(
    t_days,
    *,
    f_ck_28: float,
    code: str = "EN1992",
    cement_class: str = "N",
    cement_type: str = "I",
    curing: str = "moist",
) -> dict:
    """Return property-vs-age arrays for plotting the strength-gain
    (and modulus / tensile) curves.

    Parameters
    ----------
    t_days : array-like
        Ages to evaluate (days).
    f_ck_28 : float
        28-day strength (Pa) -- ``f_ck`` (EN 1992) or ``f'c`` (ACI).
    code : {"EN1992", "ACI209"}

    Returns
    -------
    dict with arrays ``t``, ``f_cm``, ``f_ck``, ``f_ctm``, ``E_cm``,
    ``beta`` (all Pa except ``t`` in days and ``beta`` dimensionless).
    """
    t = np.atleast_1d(np.asarray(t_days, dtype=float))
    rows = []
    for ti in t:
        if code == "EN1992":
            rows.append(en1992_strength_gain(
                float(ti), f_ck_28=f_ck_28, cement_class=cement_class))
        elif code == "ACI209":
            rows.append(aci209_strength_gain(
                float(ti), f_c_28=f_ck_28,
                cement_type=cement_type, curing=curing))
        else:
            raise ValueError("code must be 'EN1992' or 'ACI209'")
    return {
        "t": t,
        "beta": np.array([r.beta_cc for r in rows]),
        "f_cm": np.array([r.f_cm for r in rows]),
        "f_ck": np.array([r.f_ck for r in rows]),
        "f_ctm": np.array([r.f_ctm for r in rows]),
        "E_cm": np.array([r.E_cm for r in rows]),
        "code": code,
    }
