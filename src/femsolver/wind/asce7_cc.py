"""ASCE 7-22 Components-and-Cladding (C&C) wind pressures.

The MWFRS calculation in :mod:`asce7` gives global design forces
acting on the lateral system. *Components and cladding* are the
individual panels, mullions, purlins, fasteners that span between
the main system -- they are designed to localised peak gusts that
can far exceed the building-averaged MWFRS pressures.

ASCE 7-22 Chapter 30 partitions the building exterior into **zones**
where the peak pressure coefficient ``GC_p`` differs:

Walls
-----
* **Zone 4** -- interior of wall, away from corners (less peak gust).
* **Zone 5** -- near corners, within edge distance ``a`` (highest
  peak suction).

Roofs (flat / low-slope, theta <= 7 deg)
----------------------------------------
* **Zone 1**  -- interior of roof.
* **Zone 1'** -- inner interior region of large roofs.
* **Zone 2**  -- roof edges, within ``a`` of any edge.
* **Zone 3**  -- roof corners, within ``a`` of two adjacent edges.

Edge distance
-------------
``a = min(0.1 * smallest_horizontal_dimension, 0.4 * h)``,
but >= 0.04 * smallest_horizontal_dimension and >= 0.91 m (3 ft).

Effective wind area
-------------------
``A_e`` is the larger of the element's span * its width, or one
third of the span. GC_p magnitudes *decrease* as A_e increases (the
larger the element, the less probable that the peak gust covers it
all simultaneously).

Internal pressure (GC_pi)
-------------------------
* Enclosed building: GC_pi = +/- 0.18
* Partially-enclosed: GC_pi = +/- 0.55
* Open: GC_pi = 0

Combined design pressure
------------------------
::

    p = q_h * GC_p - q_i * GC_pi

with ``q_i = q_h`` for enclosed and partially-enclosed buildings.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


# ============================================================ edge distance

def cc_edge_distance(*, B: float, h: float) -> float:
    """Compute the C&C edge-zone width ``a`` per ASCE 7-22 26.2.

    Parameters
    ----------
    B : float
        Smallest horizontal plan dimension (m).
    h : float
        Mean roof height (m).
    """
    if B <= 0 or h <= 0:
        raise ValueError("B and h must be positive")
    a = min(0.10 * B, 0.40 * h)
    a = max(a, 0.04 * B, 0.9144)         # 3 ft = 0.9144 m
    return float(a)


# ============================================================ coefficient curves

@dataclass
class CCPressureCoefficient:
    """One zone's positive/negative GC_p at a specific effective area."""
    zone: str                      # "wall_4", "wall_5", "roof_1", ...
    GC_p_pos: float                # positive (pressure into surface)
    GC_p_neg: float                # negative (suction)
    effective_area_m2: float


def _interp_log_area(
    A_e: float, A_low: float, A_high: float,
    val_low: float, val_high: float,
) -> float:
    """Linear interpolation in log(A_e) between two reference areas."""
    if A_e <= A_low:
        return val_low
    if A_e >= A_high:
        return val_high
    return float(val_low + (val_high - val_low)
                 * (math.log(A_e) - math.log(A_low))
                 / (math.log(A_high) - math.log(A_low)))


# Curve anchor values (Pa-free, just GC_p). All from ASCE 7-22 Figure
# 30.5-1 for low-rise (h <= 60 ft) buildings. Areas in m^2.
# Format: ((A_low, val_low), (A_high, val_high)) -- log-interp between.

_WALL_GCp = {
    # Zone 4 (interior wall): A_low=10 sf=0.93 m^2, A_high=500 sf=46.5 m^2
    "wall_4_pos": ((0.93, +1.0), (46.5, +0.7)),
    "wall_4_neg": ((0.93, -1.1), (46.5, -0.8)),
    # Zone 5 (corner wall): higher peak negative pressure
    "wall_5_pos": ((0.93, +1.0), (46.5, +0.7)),
    "wall_5_neg": ((0.93, -1.4), (46.5, -0.8)),
}

_ROOF_FLAT_GCp = {
    # Zone 1 (interior of flat / low-slope roof): A_low=10 sf, A_high=100 sf
    "roof_1_pos": ((0.93, +0.2), (9.30, +0.2)),
    "roof_1_neg": ((0.93, -1.7), (9.30, -1.2)),
    # Zone 2 (roof edge)
    "roof_2_pos": ((0.93, +0.2), (9.30, +0.2)),
    "roof_2_neg": ((0.93, -2.3), (9.30, -1.6)),
    # Zone 3 (roof corner)
    "roof_3_pos": ((0.93, +0.2), (9.30, +0.2)),
    "roof_3_neg": ((0.93, -3.2), (9.30, -2.3)),
}


# ============================================================ wall GC_p

def cc_wall_GCp(
    *,
    A_e: float,
    zone: str = "wall_5",
) -> CCPressureCoefficient:
    """Wall C&C pressure coefficient.

    Parameters
    ----------
    A_e : float
        Effective wind area (m^2).
    zone : {"wall_4", "wall_5"}, default "wall_5" (corner)
    """
    if A_e <= 0:
        raise ValueError(f"A_e must be positive, got {A_e}")
    if zone not in ("wall_4", "wall_5"):
        raise ValueError(
            f"zone must be 'wall_4' or 'wall_5', got {zone!r}"
        )
    a_lo, p_lo = _WALL_GCp[f"{zone}_pos"]
    GC_p_pos = _interp_log_area(A_e, a_lo[0], p_lo[0], a_lo[1], p_lo[1])
    a_lo_n, p_lo_n = _WALL_GCp[f"{zone}_neg"]
    GC_p_neg = _interp_log_area(
        A_e, a_lo_n[0], p_lo_n[0], a_lo_n[1], p_lo_n[1],
    )
    return CCPressureCoefficient(
        zone=zone, GC_p_pos=float(GC_p_pos), GC_p_neg=float(GC_p_neg),
        effective_area_m2=float(A_e),
    )


# ============================================================ roof GC_p

def cc_roof_GCp(
    *,
    A_e: float,
    zone: str = "roof_1",
    roof_type: str = "flat",
) -> CCPressureCoefficient:
    """Flat / low-slope roof C&C pressure coefficient.

    Parameters
    ----------
    A_e : float
        Effective wind area (m^2).
    zone : {"roof_1", "roof_2", "roof_3"}, default "roof_1"
    roof_type : {"flat"}, default "flat"
        Only flat / low-slope (theta <= 7 deg) supported by this MVP;
        sloped (gable, hip) require Figure 30.4-2A/B which is a future
        extension.
    """
    if A_e <= 0:
        raise ValueError(f"A_e must be positive, got {A_e}")
    if zone not in ("roof_1", "roof_2", "roof_3"):
        raise ValueError(
            f"zone must be one of roof_1/roof_2/roof_3, got {zone!r}"
        )
    if roof_type != "flat":
        raise ValueError(
            f"only roof_type='flat' supported in this MVP (got {roof_type!r})"
        )
    a_pos, p_pos = _ROOF_FLAT_GCp[f"{zone}_pos"]
    a_neg, p_neg = _ROOF_FLAT_GCp[f"{zone}_neg"]
    GC_p_pos = _interp_log_area(A_e, a_pos[0], p_pos[0], a_pos[1], p_pos[1])
    GC_p_neg = _interp_log_area(A_e, a_neg[0], p_neg[0], a_neg[1], p_neg[1])
    return CCPressureCoefficient(
        zone=zone, GC_p_pos=float(GC_p_pos), GC_p_neg=float(GC_p_neg),
        effective_area_m2=float(A_e),
    )


# ============================================================ internal

def gcpi_for_enclosure(enclosure: str) -> float:
    """``GC_pi`` magnitude (always +/-) from the building enclosure
    classification per ASCE 7-22 26.13.

    Parameters
    ----------
    enclosure : {"enclosed", "partially_enclosed", "open"}
    """
    table = {
        "enclosed": 0.18,
        "partially_enclosed": 0.55,
        "open": 0.0,
    }
    if enclosure not in table:
        raise ValueError(
            f"enclosure must be one of {list(table)}, got {enclosure!r}"
        )
    return table[enclosure]


# ============================================================ design pressure

@dataclass
class CCDesignPressure:
    """Combined design pressure for one zone / case."""
    zone: str
    p_max: float           # +ve into surface (worst inward)
    p_min: float           # -ve out of surface (worst suction)
    q_h: float
    GC_p_pos: float
    GC_p_neg: float
    GC_pi: float
    enclosure: str


def cc_design_pressure(
    *,
    coeff: CCPressureCoefficient,
    q_h: float,
    enclosure: str = "enclosed",
) -> CCDesignPressure:
    """Combined design pressure per ASCE 7-22 30.3.1::

        p = q_h * [GC_p - GC_pi]

    where ``GC_pi`` is +/- based on enclosure. Returns *both* design
    cases (positive internal + GC_p_pos and negative internal + GC_p_neg)
    as ``(p_max, p_min)``.
    """
    if q_h <= 0:
        raise ValueError(f"q_h must be positive, got {q_h}")
    GC_pi = gcpi_for_enclosure(enclosure)
    # Two design cases (worst inward and worst outward)
    p_pos = q_h * (coeff.GC_p_pos - (-GC_pi))     # outward GC_pi
    p_neg = q_h * (coeff.GC_p_neg - (+GC_pi))     # inward GC_pi
    # Worst-inward case: GC_p_pos with -GC_pi (creates max +ve)
    p_max = q_h * (coeff.GC_p_pos + GC_pi)
    # Worst-outward case: GC_p_neg with +GC_pi (most negative)
    p_min = q_h * (coeff.GC_p_neg - GC_pi)
    return CCDesignPressure(
        zone=coeff.zone,
        p_max=float(p_max),
        p_min=float(p_min),
        q_h=float(q_h),
        GC_p_pos=float(coeff.GC_p_pos),
        GC_p_neg=float(coeff.GC_p_neg),
        GC_pi=float(GC_pi),
        enclosure=enclosure,
    )
