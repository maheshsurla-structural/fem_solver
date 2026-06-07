"""Bill of materials / quantity takeoff.

Three flavours of helper that an engineer might run at the end of an
analysis to produce material tonnage and concrete volumes:

* :func:`bom_concrete_frame` -- volume per RC member (beam, column),
  rebar weight from supplied ratio, total concrete + total rebar.
* :func:`bom_steel_frame`     -- tonnage by section (W-shape, HSS),
  total steel weight.
* :func:`bom_rebar`           -- rebar weight from a list of
  ``(diameter, length, count)`` triples.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import math


_REBAR_DENSITY = 7850.0     # kg/m^3
_STEEL_DENSITY = 7850.0
_CONCRETE_DENSITY = 2400.0


# Common rebar diameters and unit weights (kg/m)
#   diameter (mm) -> unit weight (kg/m)
_REBAR_TABLE = {
    8:  0.395,
    10: 0.617,
    12: 0.888,
    14: 1.208,
    16: 1.578,
    20: 2.466,
    25: 3.854,
    28: 4.834,
    32: 6.313,
    40: 9.864,
}


@dataclass
class BomLine:
    item: str
    quantity: float
    unit: str
    description: str = ""


@dataclass
class BomReport:
    lines: list = field(default_factory=list)

    def total_for(self, unit: str) -> float:
        return float(sum(l.quantity for l in self.lines if l.unit == unit))

    def summary(self) -> dict:
        # Aggregate by item
        agg: dict[str, float] = {}
        units: dict[str, str] = {}
        for l in self.lines:
            agg[l.item] = agg.get(l.item, 0.0) + l.quantity
            units[l.item] = l.unit
        return {k: (agg[k], units[k]) for k in agg}


def bom_rebar(
    bars: Sequence[tuple[float, float, int]],
) -> BomReport:
    """Compute rebar weight from a list of ``(diameter_mm, length_m,
    count)`` triples. Diameters not in the standard table use the
    formula ``rho * (pi * (d/2)^2)``."""
    rep = BomReport()
    for diameter_mm, length_m, count in bars:
        if diameter_mm <= 0 or length_m <= 0 or count < 0:
            raise ValueError(
                f"rebar entry invalid: d={diameter_mm}, L={length_m}, "
                f"n={count}"
            )
        if diameter_mm in _REBAR_TABLE:
            unit_w = _REBAR_TABLE[diameter_mm]
        else:
            area_m2 = math.pi * (diameter_mm * 1e-3 / 2.0) ** 2
            unit_w = _REBAR_DENSITY * area_m2
        weight = unit_w * length_m * count
        rep.lines.append(BomLine(
            item=f"#{int(diameter_mm)} bar",
            quantity=weight,
            unit="kg",
            description=f"{count} x {length_m:.2f} m @ {unit_w:.3f} kg/m",
        ))
    return rep


def bom_concrete_frame(
    members: Sequence[tuple[str, float, float]],
    rebar_kg_per_m3: float = 80.0,
) -> BomReport:
    """Concrete and rebar takeoff for a list of RC members.

    Parameters
    ----------
    members : sequence of ``(name, length_m, section_area_m2)``
        Each frame member's centerline length and gross cross-section
        area.
    rebar_kg_per_m3 : float, default 80
        Typical rebar density per m^3 of concrete (80-120 kg/m^3 for
        normal RC).
    """
    rep = BomReport()
    total_vol = 0.0
    for name, L, A in members:
        if L <= 0 or A <= 0:
            raise ValueError(
                f"member {name}: L and A must be positive, got L={L}, A={A}"
            )
        vol = L * A
        total_vol += vol
        rep.lines.append(BomLine(
            item="concrete", quantity=vol, unit="m3",
            description=f"{name}: L={L:.2f} m, A={A*1e4:.0f} cm^2",
        ))
    rep.lines.append(BomLine(
        item="rebar", quantity=total_vol * rebar_kg_per_m3, unit="kg",
        description=f"{rebar_kg_per_m3} kg/m^3 over {total_vol:.2f} m^3",
    ))
    rep.lines.append(BomLine(
        item="formwork", quantity=total_vol * 4.0, unit="m2",
        description="approx. 4 m^2 / m^3 (frame typ.)",
    ))
    return rep


def bom_steel_frame(
    members: Sequence[tuple[str, float, float]],
) -> BomReport:
    """Steel tonnage by member from ``(name, length_m, section_area_m2)``.

    Steel density = 7850 kg/m^3.
    """
    rep = BomReport()
    total_kg = 0.0
    for name, L, A in members:
        if L <= 0 or A <= 0:
            raise ValueError(
                f"member {name}: L and A must be positive, got L={L}, A={A}"
            )
        w = _STEEL_DENSITY * L * A
        total_kg += w
        rep.lines.append(BomLine(
            item="steel", quantity=w, unit="kg",
            description=f"{name}: L={L:.2f} m, A={A*1e4:.1f} cm^2",
        ))
    return rep
