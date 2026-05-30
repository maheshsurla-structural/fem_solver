"""Phase 30.8 -- end-to-end steel frame design per AISC 360-22.

Capstone for Phase 30 (Steel Design). Builds a 3-story 2-bay steel
moment frame, analyses it under a factored gravity + lateral
combination, extracts per-member force envelopes (P, M, V), runs
the AISC 360-22 DCR check for an initial section assignment, then
**auto-sizes** each member from the embedded W-shapes catalog to
find the lightest section satisfying every limit state.

Pipeline
--------
1. Build a 3 x 2 steel moment frame using BeamColumn2D elements with
   ElasticIsotropic steel (E = 200 GPa) and initial sections
   (W14x90 columns, W18x60 beams).
2. Apply factored gravity UDL (30 kN/m on each beam) + lateral nodal
   loads (50 kN per floor) -- a single LRFD combination, simulating
   1.2 D + 1.0 W.
3. Run LinearStaticAnalysis; extract per-member end forces.
4. For each member, compute the demand envelope (max |P|, max |M|,
   max |V|) and:
   - Run ``SteelMemberDesigner.check_member`` on the initial section
     to get the current DCR.
   - Run ``SteelMemberDesigner.auto_size`` to find the lightest
     section satisfying all checks.
5. Print a summary table comparing initial vs auto-sized layouts.

Run::

    python examples/50_steel_frame_design.py
"""
from __future__ import annotations

import numpy as np

from femsolver import (
    BeamColumn2D,
    ElasticIsotropic,
    Model,
)
from femsolver.analysis.linear_static import LinearStaticAnalysis
from femsolver.design.steel import (
    SteelMemberDemand,
    SteelMemberDesigner,
    astm_a992,
    get_section,
    w_series,
)


# ============================================================ model

N_STORY = 3
N_BAY = 2
H_STORY = 3.5         # m
L_BAY = 6.0           # m

# Initial trial sections (a structural engineer's first guess)
INITIAL_COLUMN = "W14x90"
INITIAL_BEAM = "W18x60"

# Steel: A992
STEEL_MAT = astm_a992()
E_STEEL = STEEL_MAT.E

# Loads (factored, 1.2D + 1.0W proxy)
W_BEAM_DEAD = 30e3            # 30 kN/m factored gravity UDL on each beam
F_LATERAL_PER_FLOOR = 50e3    # 50 kN factored lateral at each floor


def build_frame():
    """Build the 3-story 2-bay steel moment frame."""
    sec_col = get_section(INITIAL_COLUMN)
    sec_beam = get_section(INITIAL_BEAM)
    mat = ElasticIsotropic(1, E=E_STEEL, nu=0.30, rho=7850.0)
    m = Model(ndm=2, ndf=3); m.add_material(mat)

    n_col = N_BAY + 1
    for j in range(N_STORY + 1):
        for i in range(n_col):
            tag = j * n_col + i + 1
            m.add_node(tag, i * L_BAY, j * H_STORY)

    etag = 1

    # Columns -- one per column line per story
    col_tags = []
    for j in range(N_STORY):
        for i in range(n_col):
            n_bot = j * n_col + i + 1
            n_top = (j + 1) * n_col + i + 1
            beam = BeamColumn2D(
                etag, (n_bot, n_top), mat, sec_col.A, sec_col.Ix,
            )
            m.add_element(beam)
            col_tags.append((etag, j + 1, i + 1, INITIAL_COLUMN))
            etag += 1

    # Beams -- bay-by-bay at each level
    beam_tags = []
    for j in range(1, N_STORY + 1):
        for i in range(N_BAY):
            n_L = j * n_col + i + 1
            n_R = j * n_col + i + 2
            beam = BeamColumn2D(
                etag, (n_L, n_R), mat, sec_beam.A, sec_beam.Ix,
            )
            beam.add_uniform_load(-W_BEAM_DEAD)
            m.add_element(beam)
            beam_tags.append((etag, j, i + 1, INITIAL_BEAM))
            etag += 1

    # Fix the base entirely
    for i in range(n_col):
        m.fix(i + 1, [1, 1, 1])

    # Lateral nodal loads at each floor
    for j in range(1, N_STORY + 1):
        F_each = F_LATERAL_PER_FLOOR / n_col
        for i in range(n_col):
            tag = j * n_col + i + 1
            m.add_nodal_load(tag, [F_each, 0.0, 0.0])

    return m, col_tags, beam_tags


# ============================================================ helpers

def member_envelope(element) -> dict:
    """Return (P, V, M) envelope of a BeamColumn2D element."""
    ef = element.end_forces_local
    F_xi, F_yi, M_zi, F_xj, F_yj, M_zj = ef
    sf = element.section_forces       # shape (n_int, 2): [N, M]
    N_max = max(abs(F_xi), abs(F_xj), float(np.max(np.abs(sf[:, 0]))))
    V_max = max(abs(F_yi), abs(F_yj))
    M_max = max(abs(M_zi), abs(M_zj), float(np.max(np.abs(sf[:, 1]))))
    # Column convention: compression positive = pushing UP at the top
    P_signed = -F_xj
    return {"N_max": N_max, "V_max": V_max, "M_max": M_max,
            "P_compression_signed": P_signed}


# ============================================================ main

def main() -> None:
    print("Phase 30.8 -- End-to-End Steel Frame Design Example")
    print("=" * 78)
    print(f"  Frame: {N_STORY} stories x {N_BAY} bays "
          f"({H_STORY:.1f} m x {L_BAY:.1f} m)")
    print(f"  Initial beams: {INITIAL_BEAM}")
    print(f"  Initial columns: {INITIAL_COLUMN}")
    print(f"  Steel: A992 (Fy=50 ksi)")
    print(f"  Gravity UDL: {W_BEAM_DEAD/1e3:.0f} kN/m per beam")
    print(f"  Lateral: {F_LATERAL_PER_FLOOR/1e3:.0f} kN per floor")
    print()

    # --- Analysis ---
    model, col_tags, beam_tags = build_frame()
    LinearStaticAnalysis(model).run()

    # --- Force envelopes ---
    print("Member force envelopes (from linear-static analysis):")
    print(f"  {'Member':<14} | {'M_max':>10} | {'V_max':>10} | "
          f"{'P (compr)':>12}")
    print("  " + "-" * 56)
    beam_envelopes = {}
    col_envelopes = {}
    for etag, level, bay, sect in beam_tags:
        el = model.elements[etag]
        env = member_envelope(el)
        beam_envelopes[etag] = env
        print(f"  Beam L{level}-B{bay:<5} | "
              f"{env['M_max']/1e3:>7.1f} kN.m | "
              f"{env['V_max']/1e3:>7.1f} kN | "
              f"{env['P_compression_signed']/1e3:>+7.1f} kN")
    for etag, story, col, sect in col_tags:
        el = model.elements[etag]
        env = member_envelope(el)
        col_envelopes[etag] = env
        print(f"  Col  S{story}-C{col:<5} | "
              f"{env['M_max']/1e3:>7.1f} kN.m | "
              f"{env['V_max']/1e3:>7.1f} kN | "
              f"{env['P_compression_signed']/1e3:>+7.1f} kN")
    print()

    # --- DCR + auto-size for each beam ---
    print("BEAM design (AISC 360-22):")
    print(f"  {'Member':<14} | {'Initial':<10} | {'Init DCR':>9} | "
          f"{'Auto':<10} | {'Auto DCR':>9} | {'Wt change':>11}")
    print("  " + "-" * 88)
    for etag, level, bay, init_sect_name in beam_tags:
        env = beam_envelopes[etag]
        demand = SteelMemberDemand(
            M_ux=env["M_max"], V_u=env["V_max"], P_u=0.0,
        )
        # Beam unbraced length: full span (no lateral bracing
        # assumed)
        L_member = L_BAY
        # Check the initial section
        init_chk = SteelMemberDesigner.check_member(
            get_section(init_sect_name), STEEL_MAT, demand,
            L=L_member, L_b=L_member, C_b=1.0,
        )
        # Auto-size from the full W-shapes catalog
        opt = SteelMemberDesigner.auto_size(
            STEEL_MAT, demand,
            L=L_member, L_b=L_member, C_b=1.0,
        )
        init_w = get_section(init_sect_name).weight_per_length
        if opt.best is not None:
            picked = opt.best.section.designation
            opt_dcr = opt.best.governing_DCR
            opt_w = opt.best.weight_per_length
            pct = (opt_w - init_w) / init_w * 100
            change_str = f"{pct:+.0f}%"
        else:
            picked = "FAILED"
            opt_dcr = float("nan")
            change_str = "n/a"
        flag_init = " " if init_chk.passes else "X"
        print(f"  Beam L{level}-B{bay:<5} | "
              f"{init_sect_name:<10} | "
              f"{init_chk.governing_DCR:>7.3f}{flag_init} | "
              f"{picked:<10} | "
              f"{opt_dcr:>7.3f}  | "
              f"{change_str:>11}")
    print()

    # --- DCR + auto-size for each column ---
    print("COLUMN design (AISC 360-22):")
    print(f"  {'Member':<14} | {'Initial':<10} | {'Init DCR':>9} | "
          f"{'Auto':<10} | {'Auto DCR':>9} | {'Wt change':>11}")
    print("  " + "-" * 88)
    for etag, story, col, init_sect_name in col_tags:
        env = col_envelopes[etag]
        # Use compression-positive P_u for axial demand
        P_u = max(0.0, env["P_compression_signed"])
        demand = SteelMemberDemand(
            P_u=P_u, M_ux=env["M_max"], V_u=env["V_max"],
        )
        L_member = H_STORY
        init_chk = SteelMemberDesigner.check_member(
            get_section(init_sect_name), STEEL_MAT, demand,
            L=L_member, L_b=L_member, C_b=1.0,
        )
        # Auto-size, restrict to W-series typically used for columns
        candidates = (w_series("W10") + w_series("W12")
                       + w_series("W14"))
        opt = SteelMemberDesigner.auto_size(
            STEEL_MAT, demand,
            L=L_member, L_b=L_member, C_b=1.0,
            candidates=candidates,
        )
        init_w = get_section(init_sect_name).weight_per_length
        if opt.best is not None:
            picked = opt.best.section.designation
            opt_dcr = opt.best.governing_DCR
            opt_w = opt.best.weight_per_length
            pct = (opt_w - init_w) / init_w * 100
            change_str = f"{pct:+.0f}%"
        else:
            picked = "FAILED"
            opt_dcr = float("nan")
            change_str = "n/a"
        flag_init = " " if init_chk.passes else "X"
        print(f"  Col  S{story}-C{col:<5} | "
              f"{init_sect_name:<10} | "
              f"{init_chk.governing_DCR:>7.3f}{flag_init} | "
              f"{picked:<10} | "
              f"{opt_dcr:>7.3f}  | "
              f"{change_str:>11}")
    print()

    # --- Summary ---
    print("Reading the result:")
    print("* Each member is analysed once, then run through the AISC")
    print("  360-22 LRFD checks (Ch. E/F/G/H combined into a single DCR).")
    print("* 'Init DCR' is the demand-capacity ratio for the initial")
    print("  trial section; an 'X' flag means the initial section failed")
    print("  (DCR > 1.0).")
    print("* 'Auto' is the lightest W-shape from the embedded catalog")
    print("  that satisfies every check; 'Wt change' compares its weight")
    print("  per metre against the initial section.")
    print("* Column auto-sizing is restricted to W10/W12/W14 families")
    print("  (typical column depths); beam auto-sizing uses the full")
    print("  catalog.")
    print("* Each member's full check objects (CompressionCheck,")
    print("  FlexureCheck, ShearCheck, CombinedForceCheck) are available")
    print("  in the result dataclasses for further reporting (Phase 33).")


if __name__ == "__main__":
    main()
