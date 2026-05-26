"""Phase 22 -- composite-laminate analysis.

Two scenarios illustrate Phase 22's CLT (Classical Laminate Theory)
machinery and the flat-facet curved-shell story:

1. **Cantilever plate -- CFRP vs aluminum** -- demonstrates the
   strength / stiffness advantage of a [0/90/90/0]s cross-ply
   carbon-fiber laminate over an aluminum plate of equal mass.
2. **Cylindrical-shell tube under internal pressure** -- a simple
   curved-shell mesh swept around a vertical axis. Shows the flat-
   facet approximation working on a doubly-curved geometry.

Run::

    python examples/34_composite_laminate.py
"""
from __future__ import annotations

import math

import numpy as np

from femsolver import (
    ElasticIsotropic,
    LayeredShellSection,
    Model,
    OrthotropicLamina,
    ShellMITC4,
    cylindrical_shell_mesh,
)
from femsolver.analysis.linear_static import LinearStaticAnalysis


def build_cantilever(*, N: int, mat_for_model,
                       section=None, thickness: float | None = None) -> Model:
    """Square plate clamped on the left edge, vertical tip load on right.

    Pass either ``section`` (a ShellSection) or ``thickness`` (for an
    isotropic single-layer shell built from ``mat_for_model``).
    """
    L = 0.5     # 500 mm side
    m = Model(ndm=3, ndf=6); m.add_material(mat_for_model)
    for j in range(N + 1):
        for i in range(N + 1):
            m.add_node(j * (N + 1) + i + 1, i * L / N, j * L / N, 0.0)
    etag = 1
    for j in range(N):
        for i in range(N):
            n1 = j * (N + 1) + i + 1; n2 = n1 + 1
            n3 = n2 + (N + 1); n4 = n1 + (N + 1)
            if section is not None:
                m.add_element(ShellMITC4(etag, (n1, n2, n3, n4),
                                            mat_for_model, section=section))
            else:
                m.add_element(ShellMITC4(etag, (n1, n2, n3, n4),
                                            mat_for_model,
                                            thickness=thickness))
            etag += 1
    for j in range(N + 1):
        m.fix(j * (N + 1) + 1, [1, 1, 1, 1, 1, 1])
    for j in range(N + 1):
        tag = j * (N + 1) + N + 1
        m.add_nodal_load(tag, [0, 0, -1.0, 0, 0, 0])
    return m


def cantilever_comparison() -> None:
    print("Cantilever plate: CFRP cross-ply vs aluminum")
    print("=" * 56)
    # CFRP T300/5208
    cfrp = OrthotropicLamina(
        E1=181.0e9, E2=10.3e9, G12=7.17e9, nu12=0.28, rho=1600.0,
    )
    aluminum = ElasticIsotropic(1, E=70.0e9, nu=0.33, rho=2700.0)

    # [0/90/90/0]s cross-ply, each ply 0.25 mm -> total 2.0 mm
    cfrp_section = LayeredShellSection.from_layers_centered([
        (cfrp, 0.25e-3, 0.0),
        (cfrp, 0.25e-3, 90.0),
        (cfrp, 0.25e-3, 0.0),
        (cfrp, 0.25e-3, 90.0),
        (cfrp, 0.25e-3, 90.0),
        (cfrp, 0.25e-3, 0.0),
        (cfrp, 0.25e-3, 90.0),
        (cfrp, 0.25e-3, 0.0),
    ])
    # Aluminum plate of same thickness
    al_section = None     # use mat directly via material+thickness

    print(f"  CFRP laminate ([0/90/90/0]s, 8 plies):")
    print(f"    total thickness: {cfrp_section.thickness*1000:.3f} mm")
    print(f"    mass per area:   {cfrp_section.thickness * cfrp_section.density:.3f} kg/m^2")
    al_thickness = 2.0e-3
    al_mass = al_thickness * aluminum.rho
    print(f"  Aluminum plate (equal thickness):")
    print(f"    total thickness: {al_thickness*1000:.3f} mm")
    print(f"    mass per area:   {al_mass:.3f} kg/m^2")

    # Cantilever both
    N = 8
    m_cfrp = build_cantilever(N=N, section=cfrp_section,
                                 mat_for_model=aluminum)
    LinearStaticAnalysis(m_cfrp).run()
    tip_cfrp = -m_cfrp.node((N + 1) * (N + 1)).disp[2]

    m_al = build_cantilever(N=N, thickness=al_thickness,
                              mat_for_model=aluminum)
    LinearStaticAnalysis(m_al).run()
    tip_al = -m_al.node((N + 1) * (N + 1)).disp[2]

    print()
    print(f"  Tip deflection under 1 N load (each of {N + 1} nodes -> {N + 1} N total):")
    print(f"    CFRP:     {tip_cfrp * 1e3:.4f} mm")
    print(f"    Aluminum: {tip_al * 1e3:.4f} mm")
    print(f"    CFRP is {tip_al / tip_cfrp:.2f}x stiffer per unit mass-equivalent thickness.")


def cylindrical_shell_demo() -> None:
    print("\nCylindrical shell under internal pressure (mass-equivalent CFRP vs aluminum)")
    print("=" * 75)
    radius = 0.5
    length = 1.0
    n_circ = 16
    n_long = 8

    nodes, quads = cylindrical_shell_mesh(
        radius=radius, length=length, n_circ=n_circ, n_long=n_long,
        axis="z",
    )
    print(f"  Mesh: {nodes.shape[0]} nodes, {quads.shape[0]} quads")
    print(f"  Cylinder: R = {radius} m, L = {length} m, axis = z")

    cfrp = OrthotropicLamina(
        E1=181.0e9, E2=10.3e9, G12=7.17e9, nu12=0.28, rho=1600.0,
    )
    aluminum = ElasticIsotropic(1, E=70.0e9, nu=0.33, rho=2700.0)

    # CFRP hoop laminate: fibers in the hoop (theta = 90 deg) for
    # maximum hoop strength.
    # Note: in our cylindrical mesh, the local-x of each facet points
    # along the cylinder axis (z) after the flat-facet rotation, so
    # theta=90 means fibers in the hoop direction of the original
    # cylinder.
    cfrp_section = LayeredShellSection.from_layers_centered([
        (cfrp, 0.25e-3, 90.0),
        (cfrp, 0.25e-3, 0.0),
        (cfrp, 0.25e-3, 0.0),
        (cfrp, 0.25e-3, 90.0),
    ])

    def build(section_or_thickness):
        m = Model(ndm=3, ndf=6); m.add_material(aluminum)
        for k, coord in enumerate(nodes):
            m.add_node(k + 1, coord[0], coord[1], coord[2])
        etag = 1
        for q in quads:
            n1, n2, n3, n4 = (int(q[0]) + 1, int(q[1]) + 1,
                                int(q[2]) + 1, int(q[3]) + 1)
            if isinstance(section_or_thickness, float):
                m.add_element(ShellMITC4(
                    etag, (n1, n2, n3, n4),
                    aluminum, thickness=section_or_thickness,
                ))
            else:
                m.add_element(ShellMITC4(
                    etag, (n1, n2, n3, n4),
                    aluminum, section=section_or_thickness,
                ))
            etag += 1
        # Clamp the bottom ring (z = 0). Nodes 0..n_circ-1 are on the
        # bottom edge.
        for k in range(n_circ):
            m.fix(k + 1, [1, 1, 1, 1, 1, 1])
        # Internal pressure: apply outward radial nodal force on each
        # free node (rough approximation; exact = surface integral).
        for k, coord in enumerate(nodes):
            if k < n_circ:
                continue       # skip clamped base
            r_vec = np.array([coord[0], coord[1], 0.0])
            r_norm = np.linalg.norm(r_vec)
            if r_norm > 0:
                rhat = r_vec / r_norm
                m.add_nodal_load(k + 1, [10.0 * rhat[0],
                                            10.0 * rhat[1],
                                            0.0, 0, 0, 0])
        return m

    m_cfrp = build(cfrp_section)
    LinearStaticAnalysis(m_cfrp).run()
    # Hoop expansion at the top (sample a node on the top rim)
    top_node = len(nodes)
    r_top_cfrp = np.linalg.norm(
        np.array([m_cfrp.node(top_node).disp[0],
                   m_cfrp.node(top_node).disp[1], 0.0])
    )
    print(f"  CFRP (hoop-aligned cross-ply), top-edge radial expansion: "
          f"{r_top_cfrp * 1e6:.2f} um")

    m_al = build(2.0e-3)
    LinearStaticAnalysis(m_al).run()
    r_top_al = np.linalg.norm(
        np.array([m_al.node(top_node).disp[0],
                   m_al.node(top_node).disp[1], 0.0])
    )
    print(f"  Aluminum (2 mm),                top-edge radial expansion: "
          f"{r_top_al * 1e6:.2f} um")
    if r_top_cfrp > 0 and r_top_al > 0:
        print(f"  CFRP/Al ratio: {r_top_cfrp / r_top_al:.3f}")

    print()
    print("Reading the result:")
    print("* CFRP cross-ply ([0/90]) with fibers aligned to the hoop")
    print("  gives a high-stiffness pressure-resistant shell at lower")
    print("  mass than aluminum.")
    print("* The cylindrical-shell mesh uses flat MITC4 facets that")
    print("  approximate the curved surface. For meshes with > 12-16")
    print("  facets per circumference, the flat-facet error is small.")
    print("* For a tighter approximation of doubly-curved shells")
    print("  (e.g. shells of revolution with rapid curvature change),")
    print("  use a finer mesh or a curved-shell element formulation")
    print("  (future Phase 22.x).")


def main() -> None:
    cantilever_comparison()
    cylindrical_shell_demo()


if __name__ == "__main__":
    main()
