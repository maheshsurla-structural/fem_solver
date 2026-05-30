"""Diaphragm classification and force-transfer helpers.

Per ASCE 7-22 12.3.1, a horizontal diaphragm is classified by
comparing its midspan deflection ``delta_d`` under in-plane load to
the average storey drift ``delta_drift`` of the lateral system:

* **Flexible** if ``delta_d >= 2 * delta_drift`` (loads distributed
  by tributary area).
* **Rigid** (with no further check) for cast-in-place concrete or
  composite-steel diaphragms with span/depth <= 3, regardless of
  drift comparison (12.3.1.2).
* **Semi-rigid** otherwise; loads distributed by relative lateral
  stiffness (after a 3-D analysis).

This module provides:

* :func:`classify_diaphragm` -- returns one of ``"rigid"``,
  ``"flexible"``, ``"semi_rigid"`` from the drift / deflection
  inputs.
* :func:`flexible_transfer` -- distributes a horizontal floor force
  to lateral elements by tributary area.
* :func:`rigid_transfer` -- distributes by relative lateral stiffness
  (with optional centre-of-rigidity eccentricity torsion).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def classify_diaphragm(
    delta_d: float,
    delta_drift_avg: float,
    *,
    span_over_depth: float | None = None,
    material: str = "concrete",
) -> str:
    """ASCE 7-22 12.3.1 diaphragm classification.

    Parameters
    ----------
    delta_d : float
        Diaphragm midspan deflection under the floor's lateral load
        (m).
    delta_drift_avg : float
        Average storey drift of the vertical lateral system at the
        same floor (m).
    span_over_depth : float, optional
        Diaphragm span / depth ratio. If <= 3 and ``material`` is
        ``"concrete"`` or ``"composite_steel"``, ASCE 7 12.3.1.2
        gives an automatic *rigid* classification.
    material : str, default "concrete"
        Diaphragm construction. Currently used only for the
        automatic-rigid shortcut.

    Returns
    -------
    {"rigid", "flexible", "semi_rigid"}
    """
    if delta_d < 0 or delta_drift_avg < 0:
        raise ValueError("deflections must be non-negative")
    # ASCE 7 12.3.1.2: automatic rigid for cast-in-place concrete /
    # composite-steel diaphragms with span/depth <= 3.
    if (span_over_depth is not None and span_over_depth <= 3.0
            and material in ("concrete", "composite_steel")):
        return "rigid"
    if delta_drift_avg <= 0.0:
        # No drift baseline -- conservatively treat as semi-rigid
        return "semi_rigid"
    ratio = delta_d / delta_drift_avg
    if ratio >= 2.0:
        return "flexible"
    if ratio < 0.5:
        # Note: ASCE 7-22 does not explicitly define a *rigid*
        # quantitative threshold (it relies on the construction-type
        # shortcut). The 0.5 ratio is a widely-used engineering
        # heuristic (commentary; matches NEHRP and IBC 2018).
        return "rigid"
    return "semi_rigid"


# ============================================================ transfer

@dataclass
class TributaryShare:
    """Force shared to one lateral element."""

    element_id: str
    tributary_length: float
    fraction: float
    force: float


def flexible_transfer(
    *,
    F_total: float,
    elements: Sequence[tuple[str, float, float]],
) -> list[TributaryShare]:
    """Distribute a floor lateral force ``F_total`` (N) among lateral
    elements by tributary length.

    Parameters
    ----------
    F_total : float
        Total lateral force at the floor (N).
    elements : sequence of (id, x_position, x_extent)
        Each lateral element has a unique ``id`` (e.g., wall name),
        a position ``x`` (m, along the load-perpendicular direction),
        and an in-plane width (m). The simplest tributary rule is
        Voronoi between adjacent elements; here we simplify to
        equal share of the immediate neighbourhood width.
    """
    if F_total < 0:
        raise ValueError(f"F_total must be non-negative, got {F_total}")
    if len(elements) == 0:
        raise ValueError("at least one element required")
    # Sort by x position
    sorted_elems = sorted(elements, key=lambda t: t[1])
    xs = [t[1] for t in sorted_elems]
    # Tributary lengths from Voronoi midpoints, clamped at the
    # outermost lateral element (no diaphragm overhang beyond the
    # outermost walls -- typical engineering convention for closed
    # rectangular bays).
    diaph_left = xs[0]
    diaph_right = xs[-1]
    trib = []
    for i, (_, x, _) in enumerate(sorted_elems):
        if i == 0:
            left_b = diaph_left
        else:
            left_b = 0.5 * (xs[i - 1] + xs[i])
        if i == len(sorted_elems) - 1:
            right_b = diaph_right
        else:
            right_b = 0.5 * (xs[i] + xs[i + 1])
        trib.append(right_b - left_b)
    total = sum(trib)
    out = []
    for (eid, _, _), t in zip(sorted_elems, trib):
        frac = t / total if total > 0 else 0.0
        out.append(TributaryShare(
            element_id=eid, tributary_length=float(t),
            fraction=float(frac), force=float(frac * F_total),
        ))
    return out


@dataclass
class StiffnessShare:
    """Force + torsion-induced shear shared to one lateral element."""

    element_id: str
    K: float                        # lateral stiffness used
    F_direct: float                 # force from direct (translational) share
    F_torsion: float                # additional from torsion about CR
    F_total: float                  # signed combined


def rigid_transfer(
    *,
    F_total: float,
    elements: Sequence[tuple[str, float, float]],
    F_position: float | None = None,
) -> tuple[list[StiffnessShare], float]:
    """Distribute a floor lateral force ``F_total`` to lateral
    elements by *relative stiffness*, with optional torsion about
    the centre of rigidity when the line of action is offset.

    Parameters
    ----------
    F_total : float
        Floor lateral force (N).
    elements : sequence of (id, x_position, K)
        Each element has a unique id, a position perpendicular to
        the loading direction (m), and a lateral stiffness (N/m).
    F_position : float, optional
        Position of the applied force perpendicular to the load
        direction (m). If ``None``, taken at the geometric centre
        of the elements (no torsion).

    Returns
    -------
    (shares, e_x) where ``e_x`` is the eccentricity between F's line
    of action and the centre of rigidity (m).
    """
    if len(elements) == 0:
        raise ValueError("at least one element required")
    ids = [e[0] for e in elements]
    xs = np.array([e[1] for e in elements], dtype=float)
    Ks = np.array([e[2] for e in elements], dtype=float)
    if (Ks <= 0).any():
        raise ValueError("all element stiffnesses must be positive")
    sum_K = float(Ks.sum())
    # Centre of rigidity
    x_cr = float((Ks * xs).sum() / sum_K)
    if F_position is None:
        F_position = x_cr
    e_x = F_position - x_cr
    # Direct shear: by stiffness ratio
    F_direct = F_total * (Ks / sum_K)
    # Torsion: T = F * e_x, distributed by K_i (x_i - x_cr) / I_p
    # where I_p = sum K_i (x_i - x_cr)^2
    arms = xs - x_cr
    I_p = float((Ks * arms * arms).sum())
    if I_p < 1.0e-30:
        F_torsion = np.zeros_like(F_direct)
    else:
        T = F_total * e_x
        F_torsion = T * Ks * arms / I_p
    F_combined = F_direct + F_torsion
    shares = [
        StiffnessShare(
            element_id=str(ids[i]),
            K=float(Ks[i]),
            F_direct=float(F_direct[i]),
            F_torsion=float(F_torsion[i]),
            F_total=float(F_combined[i]),
        )
        for i in range(len(ids))
    ]
    return shares, float(e_x)
