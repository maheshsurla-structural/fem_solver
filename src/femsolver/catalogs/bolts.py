"""Bolt catalogues -- A325/A490 (AISC), ISO 8.8/10.9 (Eurocode/IS).

For each diameter the catalogue records:

* Nominal diameter ``d`` (mm)
* Gross cross-section area ``A_b`` (mm^2)
* Tensile-stress area ``A_t`` (mm^2; from ISO 898-1 / ASTM tables)
* Specified minimum tensile strength ``f_ub`` (Pa)
* Specified minimum yield strength ``f_yb`` (Pa)

Use these for connection design checks (shear, tension, slip, prying).
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class BoltProperties:
    grade: str
    d_mm: float
    A_b: float          # gross area (mm^2)
    A_t: float          # tensile stress area (mm^2)
    f_ub: float         # ultimate tensile strength (Pa)
    f_yb: float         # yield strength (Pa)


# ISO 898 metric tensile-stress-area table (mm^2)
_ISO_TENSILE_AREA = {
    8:  36.6,
    10: 58.0,
    12: 84.3,
    14: 115.0,
    16: 157.0,
    20: 245.0,
    22: 303.0,
    24: 353.0,
    27: 459.0,
    30: 561.0,
    33: 694.0,
    36: 817.0,
}


# Strength tables (Pa)
_GRADE_STRENGTHS = {
    # ISO 898-1 (EN ISO 4014/4017)
    "8.8":  (800.0e6, 640.0e6),
    "10.9": (1000.0e6, 900.0e6),
    "12.9": (1200.0e6, 1080.0e6),
    # AISC / ASTM
    "A325":  (830.0e6, 660.0e6),
    "A490":  (1035.0e6, 900.0e6),
    # IS 1367
    "4.6":  (400.0e6, 240.0e6),
    "5.6":  (500.0e6, 300.0e6),
}


def list_bolt_grades() -> list[str]:
    """Return the list of supported grade names."""
    return list(_GRADE_STRENGTHS.keys())


def bolt_lookup(grade: str, d_mm: float) -> BoltProperties:
    """Return :class:`BoltProperties` for the requested grade and
    nominal diameter (mm).

    Parameters
    ----------
    grade : str
        One of ``"4.6"``, ``"5.6"``, ``"8.8"``, ``"10.9"``,
        ``"12.9"``, ``"A325"``, ``"A490"``.
    d_mm : float
        Nominal shank diameter (mm). Must be in the ISO 898 table
        (``8, 10, 12, 14, 16, 20, 22, 24, 27, 30, 33, 36``).
    """
    g = str(grade).strip()
    if g not in _GRADE_STRENGTHS:
        raise ValueError(
            f"unknown bolt grade {grade!r}; supported: "
            f"{list_bolt_grades()}"
        )
    d = int(round(d_mm))
    if d not in _ISO_TENSILE_AREA:
        raise ValueError(
            f"diameter {d_mm} mm not in standard ISO 898 table "
            f"(supported: {sorted(_ISO_TENSILE_AREA.keys())} mm)"
        )
    A_t = _ISO_TENSILE_AREA[d]
    A_b = math.pi * (d / 2.0) ** 2
    f_ub, f_yb = _GRADE_STRENGTHS[g]
    return BoltProperties(
        grade=g, d_mm=float(d),
        A_b=float(A_b), A_t=float(A_t),
        f_ub=float(f_ub), f_yb=float(f_yb),
    )
