"""Dynamic unit system (plan U1) — a MIDAS-style Force × Length pair from which
every derived quantity's unit follows.

The engine and the stored model always live in **SI base** (N, m, Pa, rad,
kg). This module is the *presentation + input* layer: it turns a stored SI
value into the number shown in the chosen unit (``to_display``) and turns a
number typed in the chosen unit back into SI (``to_si``), plus the label to
print next to it (``label``).

A ``UnitSystem`` is defined by just two primitives — a force unit and a length
unit. Everything else (stress, moment, distributed load, area, second moment
of area, …) is derived from those two, exactly like MIDAS/CSI: pick ``kN`` and
``m`` and moments read in ``kN·m``, stresses in ``kN/m²``.

Pure and GUI-free so it can be unit-tested in isolation; the widgets in U2/U3
call into it.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

# --------------------------------------------------------------------------- #
# Primitive catalogs — factor is "how many SI base units in one of these".
# SI base: force = newton (N), length = metre (m).  First cut = SI family only
# (plan U5 adds kgf/tonf/kip/lbf, cm-free imperial lengths, etc.).
# --------------------------------------------------------------------------- #
# SI factors are exact by definition; imperial factors use the exact
# international pound-force (1 lbf = 4.4482216152605 N) and inch (0.0254 m).
FORCE_UNITS: dict[str, float] = {
    "N": 1.0,
    "kN": 1.0e3,
    "kgf": 9.80665,          # kilogram-force
    "tonf": 9.80665e3,       # metric tonne-force
    "kip": 4448.2216152605,  # kilopound-force (1000 lbf) — plan U5
    "lbf": 4.4482216152605,  # pound-force
}

LENGTH_UNITS: dict[str, float] = {
    "m": 1.0,
    "cm": 1.0e-2,
    "mm": 1.0e-3,
    "in": 0.0254,            # inch (exact) — plan U5
    "ft": 0.3048,            # foot (exact, = 12 in)
}

DEFAULT_FORCE = "kN"
DEFAULT_LENGTH = "m"


class Quantity(Enum):
    """Every physical quantity the GUI shows, keyed to how its unit is built
    from the force (F) and length (L) primitives."""
    LENGTH = "length"          # L
    DISP = "disp"              # L      (displacement — same as length)
    AREA = "area"              # L²
    INERTIA = "inertia"        # L⁴     (second moment of area)
    FORCE = "force"            # F
    MOMENT = "moment"          # F·L
    STRESS = "stress"          # F/L²
    DIST_LOAD = "dist_load"    # F/L    (line / distributed load)
    ROTATION = "rotation"      # rad    (unit-invariant)


# Exponents of (force, length) for each quantity's SI-conversion factor.
_DIM: dict[Quantity, tuple[int, int]] = {
    Quantity.LENGTH: (0, 1),
    Quantity.DISP: (0, 1),
    Quantity.AREA: (0, 2),
    Quantity.INERTIA: (0, 4),
    Quantity.FORCE: (1, 0),
    Quantity.MOMENT: (1, 1),
    Quantity.STRESS: (1, -2),
    Quantity.DIST_LOAD: (1, -1),
    Quantity.ROTATION: (0, 0),
}


def _compose_label(force: str, length: str, qty: Quantity) -> str:
    """Human label for ``qty`` under the given primitives, e.g. (kN, m) →
    MOMENT ``kN·m``, STRESS ``kN/m²``, DIST_LOAD ``kN/m``."""
    sup = {1: "", 2: "²", 3: "³", 4: "⁴"}
    if qty is Quantity.ROTATION:
        return "rad"
    fe, le = _DIM[qty]
    # numerator: force^fe · length^(le if le>0)
    num: list[str] = []
    if fe:
        num.append(force)
    if le > 0:
        num.append(length + sup[le])
    # denominator: length^(-le) when le<0
    den = length + sup[-le] if le < 0 else ""
    numerator = "·".join(num) if num else "1"
    return f"{numerator}/{den}" if den else numerator


@dataclass(frozen=True)
class UnitSystem:
    """The current display units — a force + a length, MIDAS-style. Immutable;
    switching units means constructing a new one (cheap)."""
    force: str = DEFAULT_FORCE
    length: str = DEFAULT_LENGTH

    def __post_init__(self):
        if self.force not in FORCE_UNITS:
            raise ValueError(f"unknown force unit {self.force!r}")
        if self.length not in LENGTH_UNITS:
            raise ValueError(f"unknown length unit {self.length!r}")

    # -- conversion factors ------------------------------------------------- #
    def factor(self, qty: Quantity) -> float:
        """SI-base value of one display unit of ``qty`` — i.e. multiply a
        *display* number by this to get SI, divide an SI number by it to get
        display. Example (kN, m), MOMENT → 1000.0 (one kN·m = 1000 N·m)."""
        fe, le = _DIM[qty]
        return (FORCE_UNITS[self.force] ** fe) * (LENGTH_UNITS[self.length] ** le)

    def to_si(self, value: float, qty: Quantity) -> float:
        """A number entered in display units → SI base."""
        return float(value) * self.factor(qty)

    def to_display(self, value_si: float, qty: Quantity) -> float:
        """A stored SI-base value → the number to show in display units."""
        return float(value_si) / self.factor(qty)

    # -- labels ------------------------------------------------------------- #
    def label(self, qty: Quantity) -> str:
        """Unit label for ``qty``, e.g. MOMENT under (kN, m) → ``kN·m``."""
        return _compose_label(self.force, self.length, qty)

    def fmt(self, value_si: float, qty: Quantity, *, decimals: int = 3,
            with_label: bool = True) -> str:
        """Format a stored SI value for display: convert, round, and (by
        default) append the unit label."""
        disp = self.to_display(value_si, qty)
        text = f"{disp:.{decimals}f}"
        return f"{text} {self.label(qty)}" if with_label else text

    # -- convenience -------------------------------------------------------- #
    @property
    def pair_label(self) -> str:
        """The status-bar chip text, e.g. ``kN · m``."""
        return f"{self.force} · {self.length}"

    def dof_quantity(self, dof_label: str) -> Quantity:
        """The quantity of a DOF component from its label: rotational DOFs
        (``Rx``/``Ry``/``Rz``) carry moments (F·L), translational ones forces.
        Used to convert a load vector whose entries mix the two."""
        return (Quantity.MOMENT if dof_label.strip().upper().startswith("R")
                else Quantity.FORCE)

    @classmethod
    def from_project(cls, project) -> "UnitSystem":
        """Build from a ``project.py`` Project's stored force/length strings,
        tolerating unknown/legacy values by falling back to defaults."""
        f = getattr(project, "force_unit", DEFAULT_FORCE)
        l = getattr(project, "length_unit", DEFAULT_LENGTH)
        return cls(f if f in FORCE_UNITS else DEFAULT_FORCE,
                   l if l in LENGTH_UNITS else DEFAULT_LENGTH)
