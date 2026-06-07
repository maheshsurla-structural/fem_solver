"""Pile-group behaviour: p-multipliers, group efficiency, settlement.

When piles are installed in a group, each individual pile sees less
lateral and vertical capacity than it would isolated, because the
piles share the soil they bear against. The standard practice is to
multiply the **single-pile p-y soil-spring resistance** by a
``p-multiplier`` ``f_m <= 1`` that depends on the pile's row position
(leading, second, third, …) and the centre-to-centre spacing.

This module provides:

* :func:`p_multiplier` -- Reese et al. / AASHTO LRFD per-row
  p-multipliers as a function of spacing-to-diameter ratio ``s/D``.
* :func:`group_p_multipliers` -- per-pile p-multipliers for an entire
  rectangular pile group, given the group's plan layout.
* :func:`group_efficiency_converse_labarre` -- Converse-Labarre
  efficiency formula for axial group capacity.
* :func:`group_settlement_elastic` -- elastic short-term settlement
  of a pile-cap group from Poulos & Davis 1980 superposition.

References
----------
* AASHTO LRFD Bridge Design Specifications, 9e (2020), Sec. 10.7.2.4.
* Reese, L.C. & Van Impe, W.F. (2011). *Single Piles and Pile Groups
  Under Lateral Loading*, 2e.
* Poulos, H.G. & Davis, E.H. (1980). *Pile Foundation Analysis and
  Design*. Wiley.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


# ============================================================ p-multipliers

def p_multiplier(*, row: int, s_over_D: float) -> float:
    """Lateral-load p-multiplier ``f_m`` for the given row in a group.

    Per AASHTO LRFD 10.7.2.4 (also Reese & Van Impe Table 13.6) for
    pile spacings between 3D and 5D centre-to-centre:

    +------+-----------+-----------+-----------+
    | s/D  | Row 1     | Row 2     | Row 3+    |
    +------+-----------+-----------+-----------+
    | 3.0  | 0.7       | 0.5       | 0.35      |
    | 5.0  | 1.0       | 0.85      | 0.70      |
    +------+-----------+-----------+-----------+

    Linearly interpolated for intermediate ``s/D``. For ``s/D >= 5``,
    no reduction (f_m = 1.0 leading row, etc.). For ``s/D < 3`` the
    interpolation is conservatively clamped to the s/D=3 value.

    Parameters
    ----------
    row : int
        1 for leading row, 2 for second row, 3 or more for trailing.
    s_over_D : float
        Centre-to-centre pile spacing divided by pile diameter.

    Returns
    -------
    f_m : float
        p-multiplier in (0, 1].
    """
    if row < 1:
        raise ValueError(f"row must be >= 1, got {row}")
    if s_over_D <= 0.0:
        raise ValueError("s_over_D must be > 0")
    table = {                       # (row -> (val_at_3D, val_at_5D))
        1: (0.70, 1.00),
        2: (0.50, 0.85),
        3: (0.35, 0.70),            # also used for row >= 3
    }
    key = row if row <= 3 else 3
    f3, f5 = table[key]
    if s_over_D >= 5.0:
        return 1.0 if row == 1 else float(f5)
    if s_over_D <= 3.0:
        return float(f3)
    # Linear interpolation between (3, f3) and (5, f5)
    return float(f3 + (s_over_D - 3.0) / 2.0 * (f5 - f3))


def group_p_multipliers(
    *,
    n_rows: int, n_cols: int,
    s_x: float, s_y: float,
    D: float,
    load_direction_x: bool = True,
) -> np.ndarray:
    """Per-pile p-multipliers for a rectangular pile group of
    ``n_rows x n_cols``.

    Rows are perpendicular to the load direction; columns are
    parallel. When loaded in the +x direction, the front column is
    "row 1" (leading), the second column is "row 2", etc.

    Parameters
    ----------
    n_rows : int
        Number of pile rows perpendicular to the load.
    n_cols : int
        Number of pile rows parallel to the load.
    s_x, s_y : float
        Centre-to-centre spacings (m) along x (load direction) and y.
    D : float
        Pile diameter (m).
    load_direction_x : bool, default True

    Returns
    -------
    f_m : np.ndarray of shape (n_cols, n_rows)
        Multiplier per pile. ``f_m[i, j]`` corresponds to the pile
        in column ``i`` (i = 0 is the leading column under load)
        and row ``j``.
    """
    if n_rows < 1 or n_cols < 1:
        raise ValueError("n_rows, n_cols must be >= 1")
    if D <= 0.0:
        raise ValueError("D must be > 0")
    s_load = s_x if load_direction_x else s_y
    s_over_D = s_load / D
    f_m = np.zeros((n_cols, n_rows))
    for i in range(n_cols):
        # Row number = i + 1 (1 = leading)
        for j in range(n_rows):
            f_m[i, j] = p_multiplier(row=i + 1, s_over_D=s_over_D)
    return f_m


# ============================================================ axial group efficiency

def group_efficiency_converse_labarre(
    *,
    n_rows: int, n_cols: int,
    s_x: float, s_y: float,
    D: float,
) -> float:
    """Converse-Labarre formula for axial pile-group efficiency.

    ``E_g = 1 - (theta / 90) · ((n_r - 1) m + (n_c - 1) n_r) / (n_r n_c)``

    where ``theta = atan(D / s)`` (degrees), ``n_r = n_rows``,
    ``n_c = n_cols``. Returns a value in [0.6, 1.0] for typical
    pile groups.

    Used to scale single-pile axial capacity to a group capacity:
    ``Q_g = E_g · n_piles · Q_single``.
    """
    if n_rows < 1 or n_cols < 1:
        raise ValueError("n_rows, n_cols must be >= 1")
    if D <= 0.0 or s_x <= 0.0 or s_y <= 0.0:
        raise ValueError("D, s_x, s_y must be > 0")
    s_avg = 0.5 * (s_x + s_y)
    theta_deg = math.degrees(math.atan(D / s_avg))
    num = (n_rows - 1) * n_cols + (n_cols - 1) * n_rows
    denom = 90.0 * n_rows * n_cols
    return float(1.0 - theta_deg * num / denom)


# ============================================================ group settlement

@dataclass
class GroupSettlementResult:
    """Elastic short-term settlement of a pile group.

    Attributes
    ----------
    s_single : float
        Single-pile elastic settlement under one-pile load (m).
    R_s : float
        Group-settlement ratio (s_group / s_single).
    s_group : float
        Group elastic settlement (m).
    """

    s_single: float
    R_s: float
    s_group: float


def group_settlement_elastic(
    *,
    P_group: float,
    n_piles: int,
    s_single_per_unit_load: float,
    B_g: float, D: float,
) -> GroupSettlementResult:
    """Elastic group settlement via the Poulos-Davis 1980 group-
    settlement ratio ``R_s``.

    Simplified Poulos-Davis correlation::

        R_s = (B_g / D)^0.5

    for rigid pile caps in clay; the group settlement is then
    ``s_g = R_s · s_single``, where ``s_single`` is the elastic
    settlement that would occur if the group's load were applied
    to a single pile.

    Parameters
    ----------
    P_group : float
        Total load on the group (N).
    n_piles : int
        Number of piles in the group.
    s_single_per_unit_load : float
        Elastic settlement of a SINGLE pile under unit load (m/N).
        Computed from pile / soil elastic compatibility (Poulos
        & Davis Eq 5.13 or via FE).
    B_g : float
        Group plan width (m).
    D : float
        Single-pile diameter (m).
    """
    if P_group <= 0.0 or n_piles < 1:
        raise ValueError("P_group > 0 and n_piles >= 1 required")
    if s_single_per_unit_load <= 0.0:
        raise ValueError("s_single_per_unit_load must be > 0")
    if B_g <= 0.0 or D <= 0.0:
        raise ValueError("B_g, D must be > 0")
    P_per_pile = P_group / n_piles
    s_single = s_single_per_unit_load * P_per_pile
    R_s = math.sqrt(B_g / D)
    s_group = R_s * s_single
    return GroupSettlementResult(
        s_single=float(s_single),
        R_s=float(R_s),
        s_group=float(s_group),
    )
