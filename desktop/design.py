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
    end forces, using the worst end for each moment (a single conservative
    triple). Kept for the steel §H1 path; the P-M-M path envelopes the two end
    sections concurrently instead (see ``_member_sections``)."""
    secs = _member_sections(element)
    if not secs:
        return None
    P = secs[0][0]
    return P, max(abs(s[1]) for s in secs), max(abs(s[2]) for s in secs)


def _member_sections(element):
    """The member's critical sections as concurrent ``(P_r [+compression], M_z,
    M_y)`` triples in SI base units — one per member end, each carrying that
    end's *concurrent* (simultaneous) moments and the (constant) axial force, so
    a P-M-M check envelopes real load states rather than pairing the worst Mz
    with the worst My from different ends. Only nodal loads exist here, so the
    moment varies linearly and the two ends bracket the member; add mid-span
    stations here once member (distributed) loads land. Returns [] if no forces.
    """
    ef = getattr(element, "end_forces_local", None)
    if ef is None:
        return []
    n = len(ef)
    if n == 6:                                # 2-D: [N, Vy, Mz]*2
        P = -float(ef[3])                     # axial ~ constant (nodal loads)
        return [(P, float(ef[2]), 0.0), (P, float(ef[5]), 0.0)]
    if n == 12:                               # 3-D: [N, Vy, Vz, T, My, Mz]*2
        P = -float(ef[6])
        return [(P, float(ef[5]), float(ef[4])),      # end i: (Mz, My)
                (P, float(ef[11]), float(ef[10]))]     # end j: (Mz, My)
    return []


@lru_cache(maxsize=256)
def _gsd_case(spec):
    import section_gui_core as core
    return core.build_case(spec)


def gsd_member_dcr(element, section):
    """Worst P-M-M utilisation for a member whose section carries a ``gsd_spec``,
    enveloping its critical sections (both ends) against the section's own
    interaction surface — each end checked with its *concurrent* axial +
    biaxial moments (design, φ-reduced). Returns the max utilisation, or None."""
    gsd = getattr(section, "gsd_spec", None)
    if not gsd:
        return None
    sections = _member_sections(element)
    if not sections:
        return None
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
        demands = [{"name": f"end{i + 1}", "P": P / 1e3,       # N  -> kN
                    "Mz": Mz / 1e3, "My": My / 1e3}            # N·m -> kN·m
                   for i, (P, Mz, My) in enumerate(sections)]
        results = core.demand_check(case, code, demands, design=True,
                                    spec=spec)
        return max(float(r["util"]) for r in results)
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
    """{element_tag: DCR or None} for every element in the model (under the
    loads currently applied to ``model``). A beam split at a slab edge (BE2)
    resolves each sub-element back to its parent member via ``decode_member_id``
    (BE3), so the whole beam is checked — each sub-element against its own end
    forces."""
    from project import decode_member_id
    members = {m.id: m for m in project.members}

    def _dcr(tag, e):
        mb = members.get(decode_member_id(tag))
        return member_dcr(e, mb, project) if mb is not None else None

    return {tag: _dcr(tag, e) for tag, e in model.elements.items()}


def design_envelope(project):
    """Envelope every member's DCR over all load combinations. Solves the model
    under each combination and, per member, keeps the worst DCR and the combo
    that governed. Returns ``(dcrs, governing)`` where ``dcrs[tag]`` is the
    worst DCR (or None) and ``governing[tag]`` is the governing combo name;
    returns ``(None, None)`` if the project has no combinations (caller falls
    back to :func:`design_all`)."""
    if not getattr(project, "combinations", None):
        return None, None
    from femsolver import LinearStaticAnalysis
    from project import decode_member_id
    members = {m.id: m for m in project.members}
    dcrs: dict = {}
    governing: dict = {}
    for combo in project.combinations:
        model = project.build_model(with_loads=False)
        project.apply_loads(model, ("combination", combo.id))
        try:
            LinearStaticAnalysis(model).run()
        except Exception:
            continue
        for tag, el in model.elements.items():
            mb = members.get(decode_member_id(tag))
            d = member_dcr(el, mb, project) if mb is not None else None
            dcrs.setdefault(tag, None)
            if d is None:
                continue
            if dcrs[tag] is None or d > dcrs[tag]:
                dcrs[tag] = d
                governing[tag] = combo.name
    return dcrs, governing
