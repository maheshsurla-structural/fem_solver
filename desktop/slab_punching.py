"""Punching-shear check at a slab column (slab plan S9 slice 3).

Ties the FE result to the engine punching-shear design: the demand ``V_u`` at a
column is read from that node's vertical reaction after a solve, then checked
against the ACI 318-19 concrete capacity. Pure functions here (no Qt) so they
are unit-testable; the dialog / results live in the desktop layer.
"""
from __future__ import annotations

# ACI 318 strength-reduction factor for shear.
PHI_SHEAR = 0.75


def punching_demand_from_reaction(model, node_tag: int) -> float:
    """Vertical punching demand ``V_u`` (N) at ``node_tag`` — the magnitude of
    its vertical (global-Z) reaction after a solve. 0.0 if unavailable (e.g. the
    node is not a support, so no reaction is stored there)."""
    try:
        r = model.node(node_tag).reaction
    except Exception:
        return 0.0
    if r is None or len(r) < 3:
        return 0.0
    return abs(float(r[2]))


def aci_punching_check(V_u: float, *, c_x: float, c_y: float, d: float,
                       f_c: float, position: str = "interior") -> dict:
    """ACI 318-19 punching-shear check for one column. Returns the capacity /
    demand stresses, the demand-capacity ratio ``dcr = v_u / (phi·v_c)`` and a
    pass/fail flag. ``needs_reinforcement`` when dcr > 1 (and, per ACI, the slab
    can still be reinforced only while dcr ≤ the nominal ceiling)."""
    from femsolver.design.punching import (aci318_punching_capacity,
                                            aci318_punching_demand)
    cap = aci318_punching_capacity(c_x=c_x, c_y=c_y, d=d, f_c=f_c,
                                   position=position)
    v_u = aci318_punching_demand(V_u=abs(V_u), c_x=c_x, c_y=c_y, d=d,
                                 position=position)
    phi_vc = PHI_SHEAR * cap.v_c
    dcr = (v_u / phi_vc) if phi_vc > 0 else float("inf")
    return {
        "V_u": abs(V_u),
        "v_u": v_u,                     # demand stress (Pa)
        "v_c": cap.v_c,                 # capacity stress (Pa)
        "phi_v_c": phi_vc,              # design capacity stress (Pa)
        "V_c": cap.V_c,                 # capacity force (N)
        "b_0": cap.b_0,                 # critical perimeter (m)
        "d": d,
        "position": position,
        "code": cap.code,
        "dcr": dcr,
        "ok": dcr <= 1.0,
        "needs_reinforcement": dcr > 1.0,
        "notes": cap.notes,
    }
