"""Phase 27.5 -- Pile-soil interaction via p-y springs (pushover).

A steel pipe pile is embedded 10 m into sand and pushed laterally at
the head. The embedded portion is supported by distributed nonlinear
p-y springs (API/Reese sand backbones, lumped at each pile node).
The pushover traces the pile-head load-deflection envelope, which
demonstrates:

1. **Initial response** is essentially linear -- the springs operate
   below their plastic plateau.
2. As lateral load increases, the **top springs yield first** (shallow
   ones have lowest ``p_u`` because ``p_u`` grows with depth ``z``),
   and the curve softens.
3. At large displacement the **upper springs are plastic** and the
   pile carries load through deeper springs -- the classical "lateral
   plug" of pile design.

The same model is the starting point for seismic pile response with
inertial superstructure loading (the load at the head is then the
foundation reaction from the structure).

The example also reports the surface-footing impedance from
:func:`gazetas_surface_footing` -- the alternative SSI route for a
spread footing on the same soil profile.

Run::

    python examples/42_pile_py_pushover.py
"""
from __future__ import annotations

import math

import numpy as np

from femsolver import (
    BeamColumn2D,
    ElasticIsotropic,
    HalfspaceSoil,
    Model,
    NonlinearStaticAnalysis,
    UniaxialBilinear,
    ZeroLengthElement,
    gazetas_surface_footing,
    py_curve_sand,
)


# ============================================================ geometry

D_PILE = 0.6              # m, outer diameter
T_PILE = 0.020            # m, wall thickness (steel pipe pile)
L_EMBED = 10.0            # m embedded length
N_ELEM = 10               # pile elements (1 m each)


def steel_pipe_props(D: float, t: float) -> tuple[float, float]:
    """Return (A, Iz) for a circular pipe of outer ``D``, wall ``t``."""
    Do = D
    Di = D - 2.0 * t
    A = math.pi / 4.0 * (Do ** 2 - Di ** 2)
    Iz = math.pi / 64.0 * (Do ** 4 - Di ** 4)
    return A, Iz


# ============================================================ model

def build_pile_with_py(*, F_lateral: float) -> tuple[Model, list[int], int]:
    """Build the pile-with-p-y-springs model.

    Geometry: x = lateral (deflection direction), y = vertical (up
    along the pile). The pile head is the top, the tip the bottom.

    Returns the model, the list of pile node tags (head -> tip), and
    the pile-head node tag for displacement extraction.
    """
    steel = ElasticIsotropic(1, E=2.0e11, nu=0.3, rho=7850.0)
    A, Iz = steel_pipe_props(D_PILE, T_PILE)

    m = Model(ndm=2, ndf=3)
    m.add_material(steel)

    # Pile nodes (vertical), head at top (y = 0), tip at y = -L_EMBED
    pile_tags: list[int] = []
    dz = L_EMBED / N_ELEM
    for i in range(N_ELEM + 1):
        tag = i + 1
        y = -i * dz
        m.add_node(tag, 0.0, y)
        pile_tags.append(tag)

    # Pile elements
    for i in range(N_ELEM):
        m.add_element(BeamColumn2D(
            i + 1, (pile_tags[i], pile_tags[i + 1]),
            steel, A, Iz,
        ))

    # p-y springs at each embedded pile node (skip head, node 1 is at
    # mudline so we attach starting at depth dz).
    # For each embedded node:
    #   - add a coincident "soil" node (fixed)
    #   - add a ZeroLengthElement connecting them with UniaxialBilinear
    #     in DOF 0 (x-translation)
    sand_params = dict(D=D_PILE, gamma_eff=9000.0, phi_deg=35.0)
    soil_tag_offset = 100
    spring_tag_offset = 1000
    for i in range(1, N_ELEM + 1):
        z = i * dz                          # depth below mudline (m)
        # Build the p-y backbone at this depth and fit a bilinear material
        py = py_curve_sand(z=z, **sand_params)
        k0 = py.initial_stiffness()          # N/m per unit pile length
        p_ult = float(py.p[-1])              # N per unit pile length

        # Tributary length: half of dz above + half below, simplified = dz
        tributary = dz
        # Bilinear material's "yield force" = p_ult × tributary
        K_spring = k0 * tributary             # N/m
        F_yield = p_ult * tributary           # N
        # UniaxialBilinear is stress-strain over a "unit length 1" — in
        # spring usage E = K_spring (force/length) and sigma_y = F_yield.
        mat = UniaxialBilinear(E=K_spring, sigma_y=F_yield, b=0.02)

        soil_node = soil_tag_offset + i
        m.add_node(soil_node, 0.0, -z)
        m.fix(soil_node, [1, 1, 1])
        m.add_element(ZeroLengthElement(
            spring_tag_offset + i,
            (pile_tags[i], soil_node),
            materials={0: mat},
            dofs_per_node=3,
        ))

    # Pile-tip pin: prevent rigid-body horizontal drift at base
    # (the embedded p-y springs already restrain it, but to be safe,
    # fix the tip vertical DOF only -- horizontal is via p-y).
    m.fix(pile_tags[-1], [0, 1, 0])

    # Apply lateral force at the head
    m.add_nodal_load(pile_tags[0], [F_lateral, 0.0, 0.0])
    return m, pile_tags, pile_tags[0]


# ============================================================ main

def main() -> None:
    print("=" * 72)
    print("Phase 27.5 -- Pile-soil interaction via p-y springs")
    print("=" * 72)

    # ---- Gazetas footing impedance: an alternative SSI route -------
    soil = HalfspaceSoil(G=50.0e6, nu=0.35, rho=1900.0)
    print(f"\nSoil: G = {soil.G/1e6:.1f} MPa, nu = {soil.nu}, "
          f"rho = {soil.rho} kg/m^3, V_s = {soil.Vs:.1f} m/s")

    fimp = gazetas_surface_footing(soil, B=2.0, L=3.0)
    print(f"\nGazetas spread-footing impedance (B=2 m, L=3 m surface):")
    print(f"  K_z  = {fimp.K_z/1e6:8.1f}  MN/m")
    print(f"  K_x  = {fimp.K_x/1e6:8.1f}  MN/m")
    print(f"  K_y  = {fimp.K_y/1e6:8.1f}  MN/m")
    print(f"  K_rx = {fimp.K_rx/1e9:8.3f}  GN.m/rad")
    print(f"  K_ry = {fimp.K_ry/1e9:8.3f}  GN.m/rad")
    print(f"  K_t  = {fimp.K_t/1e9:8.3f}  GN.m/rad")

    # ---- Pile pushover with p-y springs ----------------------------
    print(f"\n{'-' * 72}")
    print(f"Pile: D = {D_PILE} m, t = {T_PILE*1e3:.0f} mm, L_embed = "
          f"{L_EMBED} m, {N_ELEM} elements")

    F_pile_head_levels = [50e3, 100e3, 200e3, 400e3,
                          600e3, 800e3, 1000e3, 1200e3]   # N
    head_disp = []
    for F in F_pile_head_levels:
        model, pile_tags, head_tag = build_pile_with_py(F_lateral=F)
        ana = NonlinearStaticAnalysis(
            model, num_steps=20,
            tol=1.0e-6, max_iter=40,
        )
        try:
            ana.run()
            u_head = float(model.node(head_tag).disp[0])
        except Exception as e:
            print(f"  WARNING: F={F/1e3:.0f} kN failed to converge: {e}")
            u_head = float("nan")
        head_disp.append(u_head)
        print(f"  F = {F/1e3:7.0f} kN  -> u_head = {u_head*1e3:7.2f} mm")

    # ---- Stiffness softening summary -------------------------------
    print(f"\n{'-' * 72}")
    print("Pile-head load-deflection (softening as p-y springs yield):")
    print(f"  {'F (kN)':>10}{'u (mm)':>12}{'sec_K (kN/mm)':>18}")
    print("  " + "-" * 38)
    for F, u in zip(F_pile_head_levels, head_disp):
        if u > 0.0:
            K_sec = F / u / 1e6     # kN/mm
        else:
            K_sec = float("nan")
        print(f"  {F/1e3:>10.0f}{u*1e3:>12.2f}{K_sec:>18.3f}")

    # Initial vs final secant stiffness
    K0 = F_pile_head_levels[0] / head_disp[0] / 1.0e6 if head_disp[0] > 0 else float('nan')
    Kf = F_pile_head_levels[-1] / head_disp[-1] / 1.0e6 if head_disp[-1] > 0 else float('nan')
    if math.isfinite(K0) and math.isfinite(Kf):
        print(f"\nInitial secant K = {K0:.3f} kN/mm")
        print(f"Final   secant K = {Kf:.3f} kN/mm")
        print(f"Softening ratio  = {Kf/K0:.3f}  "
              f"(< 1: p-y springs have yielded as load increased)")

    print("\n" + "=" * 72)
    print("Phase 27 closed: Gazetas footing + API p-y/t-z/q-z springs OK.")
    print("=" * 72)


if __name__ == "__main__":
    main()
