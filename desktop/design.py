"""AISC 360-22 §H1 member checks for the Design view.

Maps each solved model element back to its project member / section /
material, builds the catalog ``SteelSection`` + ``SteelMaterial``, and runs
the engine's combined (axial + flexure) interaction to a demand/capacity
ratio. Members whose section names no W-shape are skipped (DCR ``None`` ->
drawn grey).

Conservative defaults for this first cut: unbraced length = member length,
K = 1, C_b = 1 (no bracing or moment-gradient credit).
"""
from __future__ import annotations

import numpy as np

from femsolver.design.steel.combined import combined_force_check
from femsolver.design.steel.sections import SteelMaterial, get_section


def _catalog(shape: str):
    if not shape:
        return None
    try:
        return get_section(shape.replace("X", "x"))
    except Exception:
        return None


def member_dcr(element, member, project):
    """AISC §H1 DCR for one member, or None (no steel shape / no forces)."""
    sec = next((s for s in project.sections if s.id == member.section), None)
    ss = _catalog(sec.shape) if sec else None
    if ss is None:
        return None
    mat = next((m for m in project.materials if m.id == member.material), None)
    fy = float(getattr(mat, "fy", 0.0) or 345.0e6)
    fu = float(getattr(mat, "fu", 0.0) or 448.0e6)
    E = float(getattr(mat, "E", 200.0e9))
    ef = getattr(element, "end_forces_local", None)
    if ef is None:
        return None
    n = len(ef)
    if n == 6:                                # 2-D: [N, Vy, Mz]*2
        P_r = -float(ef[3])                   # compression positive for §H1
        M_rx = max(abs(float(ef[2])), abs(float(ef[5])))
        M_ry = 0.0
    elif n == 12:                             # 3-D: [N, Vy, Vz, T, My, Mz]*2
        P_r = -float(ef[6])
        M_rx = max(abs(float(ef[5])), abs(float(ef[11])))   # strong (Mz)
        M_ry = max(abs(float(ef[4])), abs(float(ef[10])))   # weak (My)
    else:
        return None
    c = element.node_coords()
    L = float(np.linalg.norm(c[1] - c[0]))
    if L <= 0.0:
        return None
    try:
        sm = SteelMaterial(Fy=fy, Fu=max(fu, fy * 1.05), E=E)
        return float(combined_force_check(ss, sm, P_r=P_r, M_rx=M_rx,
                                          M_ry=M_ry, L=L).DCR)
    except Exception:
        return None


def design_all(model, project) -> dict:
    """{element_tag: DCR or None} for every element in the model."""
    members = {m.id: m for m in project.members}
    return {tag: (member_dcr(e, members[tag], project) if tag in members
                  else None)
            for tag, e in model.elements.items()}
