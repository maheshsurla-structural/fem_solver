"""Macro (fibre) shear-wall option (wall plan, optional).

The desktop models walls as meshed shells (linear). For nonlinear / pushover
work the classic alternative is the **macro fibre wall** — a single vertical
fibre-section beam-column through the wall centroid with confined boundary
elements + a smeared-reinforcement web (PERFORM-3D / OpenSees ``Wall`` idiom).
The engine already builds that section (``sections.response.wall.wall_section_2d``);
this module bridges a desktop wall ``Area`` to it and reports its gross
properties, so the capability is reachable from the GUI.

This produces a 2-D nonlinear *section* artifact — it is not folded into the
3-D shell solve; it is the representation you would drive a pushover with.
"""
from __future__ import annotations

import math

from femsolver import ConcreteKentPark, ConcreteMander, UniaxialMenegottoPinto
from femsolver.sections.response.wall import wall_section_2d

_ES = 200.0e9
_EPS_C0 = 0.002
_EPS_CU = 0.004


def build_macro_wall_section(*, lw, t, fc_web, fc_boundary, fy, lbe=None,
                             web_rho=0.0025, boundary_rho=0.02, es=_ES):
    """Build the engine :class:`FiberSection2D` for a macro fibre wall.

    ``lw``/``t`` wall length/thickness (m); ``fc_web``/``fc_boundary`` unconfined
    web + confined boundary concrete strengths (Pa, magnitude); ``fy`` rebar
    yield (Pa); ``lbe`` boundary-element length each end (default 0.15·ℓw);
    ``web_rho``/``boundary_rho`` vertical reinforcement ratios."""
    lbe = lbe if lbe else 0.15 * lw
    if 2 * lbe >= lw:
        lbe = 0.2 * lw
    web = ConcreteKentPark(fpc=fc_web, eps_c0=_EPS_C0, fpcu=0.2 * fc_web,
                           eps_cu=_EPS_CU)
    boundary = ConcreteMander(fpc=fc_boundary, eps_c0=_EPS_C0)   # confined core
    rebar = UniaxialMenegottoPinto(E=es, sigma_y=fy)
    return wall_section_2d(
        L_w=lw, t_w=t, L_be=lbe, web_concrete=web, boundary_concrete=boundary,
        rebar_material=rebar, web_rho=web_rho, boundary_rho=boundary_rho)


def macro_wall_properties(section) -> dict:
    """Gross geometric properties of a macro-wall fibre section (SI). The
    fibre-section exposes these as properties, not methods."""
    return {"area": float(section.gross_area),
            "Iz": float(section.gross_Iz),
            "centroid_y": float(section.centroid_y)}


def macro_wall_from_area(project, area, *, fc_web=None, fc_boundary=None,
                         fy=420e6, **kw):
    """Build the macro-wall section for a desktop wall ``Area`` — geometry from
    its quad corners, ``f'c`` from its material's concrete params when present
    (overridable). The confined boundary strength defaults to 1.3·web (typical
    confinement gain). Returns ``(section, geometry_dict)``."""
    c = {n.id: (n.x, n.y, n.z) for n in project.nodes}
    corners = [c[nid] for nid in area.nodes]
    lw = math.dist(corners[0], corners[1])
    # thickness from the shell section
    t = 0.25
    ss = project.shell_section(area.shell_section)
    if ss is not None:
        t = float(ss.thickness)
    if fc_web is None:
        mat = next((m for m in project.materials if m.id == area.material), None)
        fc_web = float(mat.params.get("fc", 30e6)) if mat and mat.params \
            else 30e6
    if fc_boundary is None:
        fc_boundary = 1.3 * fc_web
    sec = build_macro_wall_section(lw=lw, t=t, fc_web=fc_web,
                                   fc_boundary=fc_boundary, fy=fy, **kw)
    return sec, {"lw": lw, "t": t, "fc_web": fc_web, "fc_boundary": fc_boundary}
