"""Load patterns + load combinations (ASCE 7-22 LRFD).

Provides a thin orchestration layer on top of the per-node /
per-element load API (``model.add_nodal_load`` and
``BeamColumn2D.add_uniform_load``) so that calling code can:

1. Define each physical load case once (dead, live, wind, EQ in
   each direction, ...) as a :class:`LoadPattern`.
2. Combine those patterns into design-code combinations
   (:class:`LoadCombination`).
3. Loop over combinations in :class:`~femsolver.analysis.envelope.EnvelopeAnalysis`
   (Phase 31.2) to extract per-member force / displacement envelopes.

Pattern recording
-----------------
A :class:`LoadPattern` wraps a Python callable
``apply_fn(model, factor)`` -- typically a small function the user
writes that calls ``model.add_nodal_load`` (and / or
``element.add_uniform_load``) for the loads in that pattern, scaled
by ``factor``. This lets users keep their existing loading code
unchanged: the pattern is just a named, scalable handle to it.

Example
-------
::

    def dead(model, factor=1.0):
        for tag in range(1, 5):
            model.add_nodal_load(tag, [0, -100e3 * factor, 0])
        for etag in (1, 2, 3):
            model.elements[etag].add_uniform_load(-30e3 * factor)

    def wind(model, factor=1.0):
        for floor_node in (5, 9, 13):
            model.add_nodal_load(floor_node, [50e3 * factor, 0, 0])

    patterns = {
        "D": LoadPattern("D", dead),
        "W": LoadPattern("W", wind),
    }
    combos = asce7_lrfd_combinations()
    for combo in combos:
        apply_combination(model, patterns, combo)
        LinearStaticAnalysis(model).run()
        # ... record envelope ...
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


# ============================================================ pattern + combo

@dataclass
class LoadPattern:
    """A named, scalable load case.

    Attributes
    ----------
    name : str
        Identifier used in combinations (e.g. ``"D"``, ``"L"``,
        ``"W"``, ``"E_x"``).
    apply_fn : Callable[[Model, float], None]
        Function the user supplies that applies this pattern's loads
        to the model, scaled by ``factor``. The function should call
        ``model.add_nodal_load`` and / or ``element.add_uniform_load``
        with values pre-multiplied by ``factor``.
    """

    name: str
    apply_fn: Callable

    def apply_to(self, model, factor: float = 1.0) -> None:
        """Apply this pattern's loads to ``model``, scaled by ``factor``."""
        self.apply_fn(model, factor)


@dataclass
class LoadCombination:
    """A weighted sum of named load patterns per a design code combo.

    Attributes
    ----------
    name : str
        Display name (e.g. ``"1.2D + 1.6L"``).
    factors : dict[str, float]
        Mapping pattern-name -> load factor. Patterns not appearing
        in this dict are treated as zero-factor for the combination.
    """

    name: str
    factors: dict = field(default_factory=dict)

    def factor(self, pattern_name: str) -> float:
        """Return the factor for ``pattern_name`` (0 if absent)."""
        return self.factors.get(pattern_name, 0.0)


# ============================================================ apply

def apply_combination(model, patterns: dict, combo: LoadCombination) -> None:
    """Clear the model's loads, then apply each pattern listed in
    ``combo.factors`` scaled by its factor.

    Patterns named in ``combo.factors`` but missing from ``patterns``
    are silently skipped -- the combination still applies the ones
    that are defined. This is the standard behaviour: if a project
    has no live-load pattern, the ``1.2D + 1.6L`` combo simply applies
    ``1.2D``.
    """
    model.clear_loads()
    for name, f in combo.factors.items():
        if f == 0.0:
            continue
        pat = patterns.get(name)
        if pat is not None:
            pat.apply_to(model, factor=f)


# ============================================================ ASCE 7-22 combos

def asce7_lrfd_combinations(
    *,
    include_seismic: bool = True,
    include_wind: bool = True,
    include_snow_rain_roof_live: bool = True,
) -> list[LoadCombination]:
    """Standard ASCE 7-22 §2.3.1 basic LRFD strength combinations.

    Pattern naming convention (caller's patterns must use these names):

    * ``D`` -- dead load
    * ``L`` -- live load (floor)
    * ``Lr`` -- roof live load
    * ``S`` -- snow load
    * ``R`` -- rain load
    * ``W`` -- wind load
    * ``E`` -- earthquake load

    The seven basic combinations from ASCE 7-22 §2.3.1:

    1. ``1.4 D``
    2. ``1.2 D + 1.6 L + 0.5 (Lr or S or R)``
    3. ``1.2 D + 1.6 (Lr or S or R) + (1.0 L or 0.5 W)``
    4. ``1.2 D + 1.0 W + L + 0.5 (Lr or S or R)``
    5. ``1.2 D + 1.0 E + L + 0.2 S``
    6. ``0.9 D + 1.0 W``
    7. ``0.9 D + 1.0 E``

    Combinations 2 and 3 with "or" clauses expand into multiple
    variants (one per choice of Lr / S / R). The expansion produces
    the full envelope-relevant set.

    Parameters
    ----------
    include_seismic : bool, default True
        Include combinations 5 and 7 (1.2D+1.0E+L+0.2S; 0.9D+1.0E).
    include_wind : bool, default True
        Include combinations 4 and 6 (1.2D+1.0W+L+0.5Lr; 0.9D+1.0W).
    include_snow_rain_roof_live : bool, default True
        Include the Lr / S / R variants in combinations 2-4.

    Returns
    -------
    list[LoadCombination]
    """
    combos: list[LoadCombination] = []
    # 1. 1.4 D
    combos.append(LoadCombination("1.4D", {"D": 1.4}))
    # 2. 1.2 D + 1.6 L + 0.5 (Lr or S or R)
    if include_snow_rain_roof_live:
        for short_name, key in [("Lr", "Lr"), ("S", "S"), ("R", "R")]:
            combos.append(LoadCombination(
                f"1.2D + 1.6L + 0.5{short_name}",
                {"D": 1.2, "L": 1.6, key: 0.5},
            ))
    else:
        combos.append(LoadCombination("1.2D + 1.6L", {"D": 1.2, "L": 1.6}))
    # 3. 1.2 D + 1.6 (Lr or S or R) + (1.0 L or 0.5 W)
    if include_snow_rain_roof_live:
        for short_name, key in [("Lr", "Lr"), ("S", "S"), ("R", "R")]:
            combos.append(LoadCombination(
                f"1.2D + 1.6{short_name} + L",
                {"D": 1.2, key: 1.6, "L": 1.0},
            ))
            if include_wind:
                combos.append(LoadCombination(
                    f"1.2D + 1.6{short_name} + 0.5W",
                    {"D": 1.2, key: 1.6, "W": 0.5},
                ))
    # 4. 1.2 D + 1.0 W + L + 0.5 (Lr or S or R)
    if include_wind:
        if include_snow_rain_roof_live:
            for short_name, key in [("Lr", "Lr"), ("S", "S"), ("R", "R")]:
                combos.append(LoadCombination(
                    f"1.2D + 1.0W + L + 0.5{short_name}",
                    {"D": 1.2, "W": 1.0, "L": 1.0, key: 0.5},
                ))
        else:
            combos.append(LoadCombination(
                "1.2D + 1.0W + L",
                {"D": 1.2, "W": 1.0, "L": 1.0},
            ))
    # 5. 1.2 D + 1.0 E + L + 0.2 S
    if include_seismic:
        combo5_factors = {"D": 1.2, "E": 1.0, "L": 1.0}
        if include_snow_rain_roof_live:
            combo5_factors["S"] = 0.2
        combos.append(LoadCombination(
            "1.2D + 1.0E + L" + (" + 0.2S" if include_snow_rain_roof_live else ""),
            combo5_factors,
        ))
    # 6. 0.9 D + 1.0 W
    if include_wind:
        combos.append(LoadCombination("0.9D + 1.0W", {"D": 0.9, "W": 1.0}))
    # 7. 0.9 D + 1.0 E
    if include_seismic:
        combos.append(LoadCombination("0.9D + 1.0E", {"D": 0.9, "E": 1.0}))
    return combos


def asce7_lrfd_seismic_combinations_per_direction() -> list[LoadCombination]:
    """Seismic LRFD combos with **direction-resolved** E patterns.

    When the seismic analysis is performed once for ``+E_x`` and once
    for ``+E_y``, callers commonly want the eight 100-30 / SRSS
    permutations applied to the design combination. This helper
    returns the four sign-permuted ``1.2D + 1.0E + L`` and
    ``0.9D + 1.0E`` combos, using pattern names ``E_x`` and ``E_y``
    with 100% / 30% factors.

    Pattern names expected: ``"D"``, ``"L"``, ``"E_x"``, ``"E_y"``.
    """
    combos: list[LoadCombination] = []
    for sign_x in (+1.0, -1.0):
        for sign_y in (+1.0, -1.0):
            for primary in ("x", "y"):
                if primary == "x":
                    fac_x, fac_y = sign_x * 1.0, sign_y * 0.3
                else:
                    fac_x, fac_y = sign_x * 0.3, sign_y * 1.0
                combos.append(LoadCombination(
                    f"1.2D + 1.0E({primary},{sign_x:+.0f}{sign_y:+.0f}) + L",
                    {"D": 1.2, "L": 1.0, "E_x": fac_x, "E_y": fac_y},
                ))
                combos.append(LoadCombination(
                    f"0.9D + 1.0E({primary},{sign_x:+.0f}{sign_y:+.0f})",
                    {"D": 0.9, "E_x": fac_x, "E_y": fac_y},
                ))
    return combos
