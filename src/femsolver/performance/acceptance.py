"""ASCE 41 fibre-strain acceptance criteria for fiber sections / hinges.

Classifies a fibre's strain against per-material Immediate-Occupancy (IO),
Life-Safety (LS) and Collapse-Prevention (CP) limits, and rolls the fibres of a
section up to a governing performance state — the fiber-hinge analogue of
CSI's *MATERIAL PROPERTIES 09 – ACCEPTANCE CRITERIA* (per-material tension /
compression strain limits) and Caltrans/ASCE 41 performance assessment.

Defaults are taken from the benchmark CSI model's acceptance table:

* Concrete — compression IO/LS/CP = 0.003 / 0.006 / 0.015, tension ignored.
* Reinforcing steel — tension IO/LS/CP = 0.01 / 0.02 / 0.05,
  compression 0.005 / 0.01 / 0.02.

Sign convention matches the fiber materials: compression strains are negative.
"""
from __future__ import annotations

from dataclasses import dataclass

# performance levels, most-severe-limit-reached (0 = below IO / acceptable)
LEVELS = ("Elastic", "IO", "LS", "CP")            # index 0..3; CP = at/beyond CP
#            gray        green      amber      red
LEVEL_COLORS = ("#9e9e9e", "#2e7d32", "#f9a825", "#c62828")


@dataclass(frozen=True)
class FiberStrainLimits:
    """Fibre-strain acceptance limits (positive magnitudes).

    Compression limits (``*_c``) apply to compressive strain (``eps < 0``),
    tension limits (``*_t``) to tensile strain (``eps > 0``). ``ignore_tension``
    disables the tension check (concrete cracks rather than reaching a strain
    limit, so its tension side carries no acceptance limit).
    """
    io_c: float
    ls_c: float
    cp_c: float
    io_t: float
    ls_t: float
    cp_t: float
    ignore_tension: bool = False


# ASCE 41 / Caltrans defaults (benchmark CSI acceptance table, §2)
CONCRETE_LIMITS = FiberStrainLimits(
    io_c=0.003, ls_c=0.006, cp_c=0.015,
    io_t=0.01, ls_t=0.02, cp_t=0.05, ignore_tension=True)
STEEL_LIMITS = FiberStrainLimits(
    io_c=0.005, ls_c=0.01, cp_c=0.02,
    io_t=0.01, ls_t=0.02, cp_t=0.05)


def is_concrete(material) -> bool:
    """True for concrete fibre laws (incl. tension-stiffening wrappers)."""
    return ("Concrete" in type(material).__name__
            or hasattr(material, "compression"))


def default_limits(material) -> FiberStrainLimits:
    """Acceptance limits for a fibre's material — concrete vs reinforcing steel."""
    return CONCRETE_LIMITS if is_concrete(material) else STEEL_LIMITS


def classify_strain(eps: float, limits: FiberStrainLimits) -> int:
    """Most-severe acceptance level a strain has reached: 0 Elastic, 1 IO,
    2 LS, 3 CP (>= CP limit = failed)."""
    if eps >= 0.0:
        if limits.ignore_tension:
            return 0
        if eps >= limits.cp_t:
            return 3
        if eps >= limits.ls_t:
            return 2
        if eps >= limits.io_t:
            return 1
        return 0
    m = -eps
    if m >= limits.cp_c:
        return 3
    if m >= limits.ls_c:
        return 2
    if m >= limits.io_c:
        return 1
    return 0


def section_state(fibers, eps_a: float, kappa: float, *,
                  limits_for=default_limits) -> int:
    """Governing (worst) acceptance level over a section's fibres at the plane-
    section strain field ``eps(y) = eps_a - y*kappa``.

    ``fibers`` is a sequence with ``.y`` and ``.material`` (a
    :class:`FiberSection2D`'s fibres). ``limits_for(material)`` maps a fibre's
    material to its :class:`FiberStrainLimits` (default: concrete vs steel).
    """
    worst = 0
    for f in fibers:
        eps = eps_a - f.y * kappa
        lvl = classify_strain(eps, limits_for(f.material))
        if lvl > worst:
            worst = lvl
            if worst == len(LEVELS) - 1:
                break
    return worst


def level_name(level: int) -> str:
    return LEVELS[max(0, min(int(level), len(LEVELS) - 1))]
