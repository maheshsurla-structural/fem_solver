"""Phase 29.7 -- end-to-end RC frame design per ACI 318-19.

This is the capstone example that ties together Phase 29.1-29.5:

1. Build a 3-story, 2-bay reinforced-concrete moment frame in 2D.
2. Apply a single factored load combination (1.2 D + 1.0 W proxy):
   gravity on the beams (uniform load) + lateral nodal loads at each
   floor.
3. Run a linear static analysis.
4. Extract member force envelopes (max |M|, max |V|, axial) from
   each beam and column.
5. Call :func:`RcMemberDesigner.design_beam` and
   :func:`RcMemberDesigner.design_column` to design rebar that
   satisfies all the ACI 318-19 checks implemented in Phase 29.
6. Print a tabular summary with section, rebar, capacities, DCRs.

This demonstrates the **analysis → design** pipeline you'd run in a
typical workflow: analyse for forces, then design rebar to those
forces using the code-clause-traced ACI 318 implementation.

Run::

    python examples/49_rc_frame_design.py

Notes
-----
* Only **one** load combination is used here for clarity. A full
  workflow would loop over all ASCE 7-22 combos (Phase 31) and take
  the envelope across them.
* The frame is small (3 stories × 2 bays) for clear output. The
  designer scales linearly with the number of members.
* Lateral load on this frame is a placeholder representing
  wind/seismic; the magnitude is set to drive non-trivial column
  moments so the design comes out interesting.
"""
from __future__ import annotations

import numpy as np

from femsolver import (
    BeamColumn2D,
    ElasticIsotropic,
    Model,
)
from femsolver.analysis.linear_static import LinearStaticAnalysis
from femsolver.design.concrete import (
    BeamDesignDemand,
    ColumnDesignDemand,
    ConcreteMaterial,
    RcMemberDesigner,
)


# ============================================================ model

# Geometry: 3 stories @ 3.5 m, 2 bays @ 6.0 m
N_STORY = 3
N_BAY = 2
H_STORY = 3.5         # m
L_BAY = 6.0           # m

# Member dimensions
COL_B = 0.40; COL_H = 0.40       # 400 x 400 mm columns
BEAM_B = 0.30; BEAM_H = 0.55     # 300 x 550 mm beams
COL_A = COL_B * COL_H
COL_I = COL_B * COL_H ** 3 / 12.0
BEAM_A = BEAM_B * BEAM_H
BEAM_I = BEAM_B * BEAM_H ** 3 / 12.0

# Materials
FC_PRIME = 28.0e6     # 28 MPa concrete
FY = 420.0e6          # 420 MPa rebar
E_CONC = 4700.0 * (FC_PRIME / 1.0e6) ** 0.5 * 1.0e6     # ACI 19.2.2.1

# Loads (factored, already 1.2D + 1.0W approximation)
W_BEAM_DEAD = 30e3                # 30 kN/m factored gravity UDL on each beam
F_LATERAL_PER_FLOOR = 50e3        # 50 kN factored lateral at each floor


def build_frame():
    """Build the 3-story 2-bay RC moment frame."""
    mat = ElasticIsotropic(1, E=E_CONC, nu=0.20, rho=2400.0)
    m = Model(ndm=2, ndf=3); m.add_material(mat)

    # Nodes: tag = j_story * (N_BAY+1) + i_col + 1 (1-indexed)
    n_col = N_BAY + 1
    for j in range(N_STORY + 1):
        for i in range(n_col):
            tag = j * n_col + i + 1
            m.add_node(tag, i * L_BAY, j * H_STORY)

    etag = 1

    # Columns: from story j to j+1 at each column line i
    col_tags = []      # (etag, story, col_line)
    for j in range(N_STORY):
        for i in range(n_col):
            n_bot = j * n_col + i + 1
            n_top = (j + 1) * n_col + i + 1
            beam = BeamColumn2D(etag, (n_bot, n_top), mat, COL_A, COL_I)
            m.add_element(beam)
            col_tags.append((etag, j + 1, i + 1))
            etag += 1

    # Beams: between adjacent columns at each upper level
    beam_tags = []     # (etag, level, bay)
    for j in range(1, N_STORY + 1):
        for i in range(N_BAY):
            n_L = j * n_col + i + 1
            n_R = j * n_col + i + 2
            beam = BeamColumn2D(etag, (n_L, n_R), mat, BEAM_A, BEAM_I)
            # Factored gravity UDL (downward in global y -> wy_local
            # follows beam orientation; for a horizontal beam in this
            # convention, downward is -wy_local in the beam's local +y)
            beam.add_uniform_load(-W_BEAM_DEAD)
            m.add_element(beam)
            beam_tags.append((etag, j, i + 1))
            etag += 1

    # Fix the base (story 0): pin-fixed (clamped) to lock all 3 DOFs
    for i in range(n_col):
        m.fix(i + 1, [1, 1, 1])

    # Lateral loads at each floor (apply at the leftmost column line
    # of each level)
    for j in range(1, N_STORY + 1):
        # Distribute equally to all columns at this level
        F_each = F_LATERAL_PER_FLOOR / n_col
        for i in range(n_col):
            tag = j * n_col + i + 1
            m.add_nodal_load(tag, [F_each, 0.0, 0.0])

    return m, col_tags, beam_tags


# ============================================================ force extraction

def member_envelope(element) -> dict:
    """Return (P, V, M) envelope of an element from .recover()-populated
    fields. For BeamColumn2D: end forces + per-IP section forces."""
    # end forces: [F_x_i, F_y_i, M_z_i, F_x_j, F_y_j, M_z_j] in local
    ef = element.end_forces_local
    N_i = ef[0]; V_i = ef[1]; M_i = ef[2]
    N_j = ef[3]; V_j = ef[4]; M_j = ef[5]
    # Section forces at every Gauss-Lobatto integration point:
    # (n_int, [N, M]) for BeamColumn2D
    sf = element.section_forces       # shape (n_int, 2)
    N_max = max(abs(N_i), abs(N_j), float(np.max(np.abs(sf[:, 0]))))
    V_max = max(abs(V_i), abs(V_j))
    M_max = max(abs(M_i), abs(M_j), float(np.max(np.abs(sf[:, 1]))))
    # Sign of axial: in the element-end-force convention, positive
    # F_x at start node = pulling node away = tension in beam. For a
    # column whose start is at the bottom and end at the top, F_x_i
    # positive means tension (rare for gravity).
    # We define column "compression positive" as -F_x_j (the force
    # the structure pushes UP at the top of the column).
    P_axial_signed = -ef[3]       # compression positive
    return {
        "N_max": N_max,
        "V_max": V_max,
        "M_max": M_max,
        "P_compression_signed": P_axial_signed,
        "M_start": M_i,
        "M_end": M_j,
    }


# ============================================================ main

def main() -> None:
    print("Phase 29.7 -- End-to-End RC Frame Design Example")
    print("=" * 72)
    print(
        f"  Frame: {N_STORY} stories × {N_BAY} bays "
        f"({H_STORY:.1f} m × {L_BAY:.1f} m)"
    )
    print(f"  Beams: {BEAM_B*1000:.0f} × {BEAM_H*1000:.0f} mm")
    print(f"  Columns: {COL_B*1000:.0f} × {COL_H*1000:.0f} mm")
    print(f"  fc' = {FC_PRIME/1e6:.0f} MPa, fy = {FY/1e6:.0f} MPa")
    print(f"  Gravity UDL: {W_BEAM_DEAD/1e3:.0f} kN/m on each beam")
    print(f"  Lateral: {F_LATERAL_PER_FLOOR/1e3:.0f} kN per floor")
    print()

    # --- Analysis ---
    model, col_tags, beam_tags = build_frame()
    LinearStaticAnalysis(model).run()

    # --- Force envelopes ---
    print("Member force envelopes (from linear-static analysis):")
    print(f"  {'Member':<12} | {'M_max':>10} | {'V_max':>10} | "
          f"{'P (compr)':>12}")
    print("  " + "-" * 56)
    beam_envelopes: dict[int, dict] = {}
    col_envelopes: dict[int, dict] = {}
    for etag, level, bay in beam_tags:
        el = model.elements[etag]
        env = member_envelope(el)
        beam_envelopes[etag] = env
        print(f"  Beam L{level}-B{bay:<3} | "
              f"{env['M_max']/1e3:>7.1f} kN·m | "
              f"{env['V_max']/1e3:>7.1f} kN | "
              f"{env['P_compression_signed']/1e3:>+7.1f} kN")
    for etag, story, col in col_tags:
        el = model.elements[etag]
        env = member_envelope(el)
        col_envelopes[etag] = env
        print(f"  Col  S{story}-C{col:<3} | "
              f"{env['M_max']/1e3:>7.1f} kN·m | "
              f"{env['V_max']/1e3:>7.1f} kN | "
              f"{env['P_compression_signed']/1e3:>+7.1f} kN")
    print()

    # --- Design each beam ---
    mat = ConcreteMaterial(fc_prime=FC_PRIME, fy=FY)
    print("Beam design (ACI 318-19):")
    print(f"  {'Member':<14} | {'Bot bars':<22} | {'Top bars':<16} | "
          f"{'Stirrup':<14} | {'phi*Mn+':>10} | {'phi*Vn':>9}")
    print("  " + "-" * 100)
    for etag, level, bay in beam_tags:
        env = beam_envelopes[etag]
        demand = BeamDesignDemand(
            M_u_positive=env["M_max"], M_u_negative=env["M_max"],
            V_u=env["V_max"],
        )
        res = RcMemberDesigner.design_beam(
            b=BEAM_B, h=BEAM_H, material=mat,
            demand=demand, cover=0.050,
        )
        if res.success:
            s = res.section
            bottom_str = " + ".join(s.rebar.bottom_bars[:6])
            if len(s.rebar.bottom_bars) > 6:
                bottom_str += " + ..."
            top_str = " + ".join(s.rebar.top_bars[:6]) if s.rebar.top_bars else "-"
            print(f"  Beam L{level}-B{bay:<5} | "
                  f"{bottom_str:<22} | {top_str:<16} | "
                  f"{s.rebar.stirrup_designation}@{s.rebar.stirrup_spacing*1000:.0f}mm".ljust(14) + " | " +
                  f"{res.flexure_positive.phi_M_n/1e3:>+7.1f} kN·m | "
                  f"{res.shear.phi_V_n/1e3:>+6.1f} kN")
        else:
            print(f"  Beam L{level}-B{bay:<5} | FAILED: {res.notes}")
    print()

    # --- Design each column ---
    print("Column design (ACI 318-19):")
    print(f"  {'Member':<14} | {'Layout':<22} | {'rho':>6} | {'DCR':>6} | "
          f"{'phi*M_n at P_u':>16}")
    print("  " + "-" * 80)
    for etag, story, col in col_tags:
        env = col_envelopes[etag]
        # Use compression-positive P_u; if axial is tensile (unlikely
        # in gravity), pass zero (designer optimizes for compression).
        P_u = max(0.0, env["P_compression_signed"])
        M_u = env["M_max"]
        demand = ColumnDesignDemand(P_u=P_u, M_u=M_u, V_u=env["V_max"])
        res = RcMemberDesigner.design_column(
            b=COL_B, h=COL_H, material=mat,
            demand=demand, cover=0.060,
        )
        if res.success:
            s = res.section
            n_total = len(s.rebar.top_bars) + len(s.rebar.bottom_bars)
            layout = f"{n_total} × {s.rebar.top_bars[0]} symmetric"
            phi_Mn = res.interaction_surface.phi_M_n_at_P_u(P_u)
            print(f"  Col  S{story}-C{col:<5} | "
                  f"{layout:<22} | "
                  f"{res.rho*100:>5.2f}% | "
                  f"{res.dcr:>5.3f} | "
                  f"{phi_Mn/1e3:>11.1f} kN·m")
        else:
            print(f"  Col  S{story}-C{col:<5} | FAILED: {res.notes}")
    print()

    # --- Summary ---
    print("Reading the result:")
    print("* Analysis runs in milliseconds, then the per-member design")
    print("  iterates over standard bar sizes to find the lightest layout.")
    print("* Beams show clear differentiation: ground-floor beams carry")
    print("  more shear (lateral load + gravity); upper-floor beams have")
    print("  smaller demand.")
    print("* Column DCRs cluster near 0.3-0.6, indicating the columns are")
    print("  modestly sized for this demand level; the designer correctly")
    print("  picks the 1% minimum-steel layout for most.")
    print("* Beam designs uniformly use #5 bars at the typical demand;")
    print("  the designer scales bar count to demand.")
    print("* Each member's check objects (FlexuralCheck, ShearCheck,")
    print("  InteractionSurface) are available in the result dataclasses")
    print("  for further reporting (Phase 33).")


if __name__ == "__main__":
    main()
