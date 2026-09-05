"""Member demand/capacity checks for the Design view.

Maps each solved model element back to its project member / section /
material and returns a demand/capacity ratio. Two paths:

* **Concrete / PSC / composite** — a section carrying a General-Section-
  Designer ``gsd_spec`` is checked against its own P-M-M interaction surface
  via ``section_gui_core.demand_check`` (biaxial, using the section's design
  code). This is the P-M-M member-design bridge.
* **Steel** — a section naming an AISC W-shape runs the engine's combined
  (axial + flexure) §H1 interaction. Conservative first cut: unbraced length =
  member length, K = 1, C_b = 1.

Members matching neither are skipped (DCR ``None`` -> drawn grey).
"""
from __future__ import annotations

from functools import lru_cache

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


def _member_forces(element):
    """(P_r [+compression], M_z, M_y) in SI base units from an element's local
    end forces, or None. Uses the worst end for each moment. Shared by both
    the steel and the P-M-M concrete checks."""
    ef = getattr(element, "end_forces_local", None)
    if ef is None:
        return None
    n = len(ef)
    if n == 6:                                # 2-D: [N, Vy, Mz]*2
        return -float(ef[3]), max(abs(float(ef[2])), abs(float(ef[5]))), 0.0
    if n == 12:                               # 3-D: [N, Vy, Vz, T, My, Mz]*2
        return (-float(ef[6]),
                max(abs(float(ef[5])), abs(float(ef[11]))),    # strong (Mz)
                max(abs(float(ef[4])), abs(float(ef[10]))))    # weak (My)
    return None


@lru_cache(maxsize=256)
def _gsd_case(spec):
    import section_gui_core as core
    return core.build_case(spec)


def gsd_member_dcr(element, section):
    """P-M-M utilisation for a member whose section carries a ``gsd_spec``,
    checking its axial + biaxial-moment demand against the section's own
    interaction surface, or None. Design (φ-reduced) capacity, biaxial."""
    gsd = getattr(section, "gsd_spec", None)
    if not gsd:
        return None
    forces = _member_forces(element)
    if forces is None:
        return None
    P_N, Mz_Nm, My_Nm = forces
    try:
        import os
        import sys
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if root not in sys.path:
            sys.path.insert(0, root)
        import section_gui_core as core
        flds = set(core.Spec.__dataclass_fields__)
        spec = core.Spec(**{k: v for k, v in gsd.items() if k in flds})
        case = _gsd_case(spec)
        code = section.gsd_code or core.CODES[0]
        demands = [{"name": "D", "P": P_N / 1e3,          # N  -> kN
                    "Mz": Mz_Nm / 1e3, "My": My_Nm / 1e3}]  # N·m -> kN·m
        res = core.demand_check(case, code, demands, design=True,
                                spec=spec)[0]
        return float(res["util"])
    except Exception:
        return None


def member_dcr(element, member, project):
    """DCR for one member: the P-M-M concrete check when its section carries a
    GSD spec, else the AISC §H1 steel check, else None (no capacity / forces)."""
    sec = next((s for s in project.sections if s.id == member.section), None)
    if sec is None:
        return None
    if getattr(sec, "gsd_spec", None):
        return gsd_member_dcr(element, sec)
    ss = _catalog(sec.shape)
    if ss is None:
        return None
    mat = next((m for m in project.materials if m.id == member.material), None)
    fy = float(getattr(mat, "fy", 0.0) or 345.0e6)
    fu = float(getattr(mat, "fu", 0.0) or 448.0e6)
    E = float(getattr(mat, "E", 200.0e9))
    forces = _member_forces(element)
    if forces is None:
        return None
    P_r, M_rx, M_ry = forces
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
