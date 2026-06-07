"""Material grade lookups across ACI / EC / IS conventions.

Three families:

* :class:`ConcreteGrade` -- ``f_ck``, ``f_cm``, ``E_cm`` for a named
  grade (e.g., ``"C30"``, ``"M30"``, ``"4000 psi"``).
* :class:`SteelGrade`    -- structural steel grades (``"S275"``,
  ``"Fe410"``, ``"Grade 50"``).
* :class:`RebarGrade`    -- reinforcing-steel grades (``"B500"``,
  ``"Fe415"``, ``"Grade 60"``).

All output units are Pa for strength, Pa for E.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


# ============================================================ concrete

@dataclass
class ConcreteGrade:
    name: str
    f_ck: float          # characteristic cylinder strength (Pa)
    f_cm: float          # mean cylinder strength = f_ck + 8 MPa (EC2)
    E_cm: float          # secant modulus (Pa) per EC2 = 22 * (f_cm/10)^0.3 GPa
    f_ctm: float         # mean tensile strength (Pa) = 0.30 * f_ck^(2/3) MPa


_CONCRETE_TABLE = {
    # EC2 / EN 206 (cylinder strength shown; cube class given for cross-ref)
    "C16": 16e6, "C20": 20e6, "C25": 25e6, "C30": 30e6,
    "C35": 35e6, "C40": 40e6, "C45": 45e6, "C50": 50e6,
    "C55": 55e6, "C60": 60e6, "C70": 70e6, "C80": 80e6, "C90": 90e6,
    # IS 456 (cube strength)
    "M15": 15e6, "M20": 20e6, "M25": 25e6, "M30": 30e6,
    "M35": 35e6, "M40": 40e6, "M45": 45e6, "M50": 50e6,
    "M60": 60e6, "M70": 70e6, "M80": 80e6,
    # ACI psi -> Pa
    "3000 psi": 20.7e6, "4000 psi": 27.6e6, "5000 psi": 34.5e6,
    "6000 psi": 41.4e6, "8000 psi": 55.2e6, "10000 psi": 69.0e6,
}


def concrete_grade(name: str) -> ConcreteGrade:
    """Look up a concrete grade. Accepts EC, IS, or ACI conventions."""
    key = name.strip()
    if key not in _CONCRETE_TABLE:
        raise KeyError(
            f"unknown concrete grade {name!r}; supported: "
            f"{list(_CONCRETE_TABLE.keys())}"
        )
    f_ck = _CONCRETE_TABLE[key]
    f_ck_MPa = f_ck / 1e6
    f_cm = f_ck + 8.0e6                       # EC2
    E_cm = 22.0e9 * (f_cm / 1e7) ** 0.3       # EC2 secant modulus
    f_ctm = 0.30e6 * (f_ck_MPa) ** (2.0 / 3.0)
    return ConcreteGrade(
        name=key, f_ck=float(f_ck), f_cm=float(f_cm),
        E_cm=float(E_cm), f_ctm=float(f_ctm),
    )


# ============================================================ steel

@dataclass
class SteelGrade:
    name: str
    f_y: float           # nominal yield (Pa)
    f_u: float           # ultimate tensile (Pa)
    E: float = 200.0e9   # Young's modulus
    nu: float = 0.30


_STEEL_TABLE = {
    # EC3 (EN 10025)
    "S235": (235e6, 360e6),
    "S275": (275e6, 430e6),
    "S355": (355e6, 510e6),
    "S420": (420e6, 520e6),
    "S460": (460e6, 540e6),
    "S690": (690e6, 770e6),
    # IS 800 (IS 2062)
    "E165": (165e6, 290e6),
    "E250": (250e6, 410e6), "Fe410": (250e6, 410e6),
    "E350": (350e6, 490e6), "Fe490": (350e6, 490e6),
    "E410": (410e6, 540e6),
    "E450": (450e6, 570e6), "Fe550": (450e6, 570e6),
    # ASTM (commonly cited values)
    "A36": (250e6, 400e6),
    "A572 Grade 50": (345e6, 450e6),
    "A992": (345e6, 450e6),
    "A913 Grade 65": (450e6, 550e6),
}


def steel_grade(name: str) -> SteelGrade:
    """Look up a structural-steel grade."""
    key = name.strip()
    if key not in _STEEL_TABLE:
        raise KeyError(
            f"unknown steel grade {name!r}; supported: "
            f"{list(_STEEL_TABLE.keys())}"
        )
    f_y, f_u = _STEEL_TABLE[key]
    return SteelGrade(name=key, f_y=float(f_y), f_u=float(f_u))


# ============================================================ rebar

@dataclass
class RebarGrade:
    name: str
    f_yk: float          # characteristic yield (Pa)
    f_uk: float          # characteristic ultimate (Pa)
    E_s: float = 200.0e9


_REBAR_TABLE = {
    # EC2 (EN 10080)
    "B500":  (500e6, 540e6),
    "B500A": (500e6, 525e6),
    "B500B": (500e6, 540e6),
    "B500C": (500e6, 575e6),
    # IS 1786
    "Fe415": (415e6, 485e6),
    "Fe500": (500e6, 545e6),
    "Fe550": (550e6, 600e6),
    "Fe600": (600e6, 660e6),
    # ASTM A615
    "Grade 40": (276e6, 414e6),
    "Grade 60": (414e6, 621e6),
    "Grade 80": (552e6, 690e6),
    "Grade 100": (690e6, 793e6),
}


def rebar_grade(name: str) -> RebarGrade:
    """Look up a rebar grade."""
    key = name.strip()
    if key not in _REBAR_TABLE:
        raise KeyError(
            f"unknown rebar grade {name!r}; supported: "
            f"{list(_REBAR_TABLE.keys())}"
        )
    f_yk, f_uk = _REBAR_TABLE[key]
    return RebarGrade(name=key, f_yk=float(f_yk), f_uk=float(f_uk))
