"""Wood-Armer design moments for orthogonally-reinforced slabs (slab plan S9).

A shell/plate finite element reports three bending stress-resultants per unit
width in its local axes: ``Mx`` (=M11), ``My`` (=M22) and the twisting moment
``Mxy`` (=M12). Reinforcement, however, runs in the two orthogonal directions
only and cannot resist twist directly. The **Wood-Armer** method (Wood 1968,
Armer 1968) folds the twist into equivalent *design moments* for the x- and
y-direction bars, separately for the bottom face (sagging / positive) and the
top face (hogging / negative), so each bar layer is sized for a single moment.

Sign convention: a **positive** moment causes tension on the **bottom** face
(needs bottom steel); a **negative** moment needs top steel.

Bottom steel (Wood-Armer):
    Mx* = Mx + |Mxy|,  My* = My + |Mxy|
    if Mx* < 0:  Mx* = 0,  My* = My + |Mxy²/Mx|   (and clamp ≥ 0)
    elif My* < 0: My* = 0,  Mx* = Mx + |Mxy²/My|   (and clamp ≥ 0)

Top steel is the mirror with ``- |Mxy|`` and the ">0 → 0" corrections, giving
non-positive design moments whose magnitude sizes the top bars.

``required_reinforcement`` then sizes the steel area per unit width for one
design moment via the ACI 318 rectangular stress block.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class WoodArmerResult:
    """Wood-Armer design moments per unit width (N·m/m), in the element's local
    axes. Bottom values are ≥ 0 (size the bottom bars); top values are ≤ 0
    (their magnitude sizes the top bars)."""
    mx_bot: float
    my_bot: float
    mx_top: float
    my_top: float


def wood_armer_moments(mx: float, my: float, mxy: float) -> WoodArmerResult:
    """Design moments for x/y reinforcement, bottom and top faces, from the
    shell resultants ``(Mx, My, Mxy)`` per unit width (sagging positive)."""
    t = abs(float(mxy))
    mx = float(mx)
    my = float(my)

    # ---- bottom (positive / sagging) steel ----
    bx = mx + t
    by = my + t
    if bx < 0.0:
        bx = 0.0
        by = my + (t * t / abs(mx) if mx != 0.0 else 0.0)
        by = max(by, 0.0)
    elif by < 0.0:
        by = 0.0
        bx = mx + (t * t / abs(my) if my != 0.0 else 0.0)
        bx = max(bx, 0.0)

    # ---- top (negative / hogging) steel ----
    tx = mx - t
    ty = my - t
    if tx > 0.0:
        tx = 0.0
        ty = my - (t * t / abs(mx) if mx != 0.0 else 0.0)
        ty = min(ty, 0.0)
    elif ty > 0.0:
        ty = 0.0
        tx = mx - (t * t / abs(my) if my != 0.0 else 0.0)
        tx = min(tx, 0.0)

    return WoodArmerResult(mx_bot=bx, my_bot=by, mx_top=tx, my_top=ty)


def required_reinforcement(m_design: float, d: float, fy: float, fc: float,
                           *, phi: float = 0.9, b: float = 1.0) -> float:
    """Tension steel area per width ``b`` (m²) for a factored design moment
    ``m_design`` (N·m, magnitude) via the ACI 318 rectangular stress block:

        As = (0.85 f'c b / fy) · (d − √(d² − 2·Mu / (φ·0.85·f'c·b)))

    ``d`` effective depth (m), ``fy`` / ``fc`` in Pa. Returns 0.0 for a
    non-positive moment. Raises ``ValueError`` if the section is too shallow to
    develop the moment as singly-reinforced (needs compression steel / redesign).
    """
    m = abs(float(m_design))
    if m == 0.0:
        return 0.0
    if d <= 0.0 or fy <= 0.0 or fc <= 0.0:
        raise ValueError("d, fy, fc must be positive")
    k = 0.85 * fc * b
    disc = d * d - 2.0 * m / (phi * k)
    if disc < 0.0:
        raise ValueError(
            "section inadequate for singly-reinforced design "
            f"(m={m:.3g} N·m exceeds the singly-reinforced capacity at d={d} m)")
    return (k / fy) * (d - math.sqrt(disc))
