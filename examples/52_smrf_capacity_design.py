"""Phase 32.5 -- Special Moment Frame (SMF) capacity-design walkthrough.

A 3-story 2-bay RC SMF is designed in three layers, each building on
the previous:

1. **Base design** (Phase 29): for the demand envelope from a single
   factored combination, design beam flexure + shear and column P-M
   reinforcement.
2. **Seismic detailing** (Phase 32):
   * SCWB ratio check at every interior beam-column joint (ACI 18.7.3).
   * Capacity-design beam shear ``V_e`` from probable moments ``M_pr``
     using the designed reinforcement (ACI 18.6.5). If ``V_e > V_u``
     the stirrups have to be sized for the larger shear.
   * Confined-concrete reinforcement detailing at column ends
     (ACI 18.7.5).

The output table contrasts the base design against the seismic-
detailing modifications -- showing where the SMF requirements bite.

Run::

    python examples/52_smrf_capacity_design.py
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
    beam_flexural_strength,
    column_interaction_surface,
    design_stirrup_spacing,
)
from femsolver.design.seismic import (
    capacity_design_shear,
    confined_concrete_detailing,
    scwb_check,
)


# ============================================================ model

N_STORY = 3
N_BAY = 2
H_STORY = 3.5
L_BAY = 6.0

COL_B = 0.50; COL_H = 0.50
BEAM_B = 0.30; BEAM_H = 0.60

FC_PRIME = 28e6
FY = 420e6
E_CONC = 4700.0 * (FC_PRIME / 1e6) ** 0.5 * 1e6

W_BEAM_DEAD = 40e3       # 40 kN/m factored gravity UDL
F_LATERAL_PER_FLOOR = 80e3   # 80 kN factored lateral (EQ)


def build_frame():
    """Build the SMF for analysis."""
    mat = ElasticIsotropic(1, E=E_CONC, nu=0.20, rho=2400.0)
    m = Model(ndm=2, ndf=3); m.add_material(mat)
    n_col = N_BAY + 1
    for j in range(N_STORY + 1):
        for i in range(n_col):
            m.add_node(j * n_col + i + 1, i * L_BAY, j * H_STORY)
    etag = 1
    col_tags = []
    for j in range(N_STORY):
        for i in range(n_col):
            n_b = j * n_col + i + 1
            n_t = (j + 1) * n_col + i + 1
            m.add_element(BeamColumn2D(
                etag, (n_b, n_t), mat, COL_B * COL_H,
                COL_B * COL_H ** 3 / 12.0,
            ))
            col_tags.append((etag, j + 1, i + 1))
            etag += 1
    beam_tags = []
    for j in range(1, N_STORY + 1):
        for i in range(N_BAY):
            n_L = j * n_col + i + 1
            n_R = j * n_col + i + 2
            b = BeamColumn2D(
                etag, (n_L, n_R), mat,
                BEAM_B * BEAM_H,
                BEAM_B * BEAM_H ** 3 / 12.0,
            )
            b.add_uniform_load(-W_BEAM_DEAD)
            m.add_element(b)
            beam_tags.append((etag, j, i + 1))
            etag += 1
    for i in range(n_col):
        m.fix(i + 1, [1, 1, 1])
    for j in range(1, N_STORY + 1):
        F_each = F_LATERAL_PER_FLOOR / n_col
        for i in range(n_col):
            m.add_nodal_load(j * n_col + i + 1, [F_each, 0, 0])
    return m, col_tags, beam_tags


def member_envelope(element):
    ef = element.end_forces_local
    sf = element.section_forces
    return {
        "M_max": max(abs(ef[2]), abs(ef[5]),
                     float(np.max(np.abs(sf[:, 1])))),
        "V_max": max(abs(ef[1]), abs(ef[4])),
        "P_compr": -ef[3],
    }


# ============================================================ main

def main() -> None:
    print("Phase 32.5 -- SMF Capacity-Design Walkthrough")
    print("=" * 78)
    print(f"  Frame: {N_STORY}-story x {N_BAY}-bay RC SMF")
    print(f"  Beams: {BEAM_B*1000:.0f} x {BEAM_H*1000:.0f} mm")
    print(f"  Columns: {COL_B*1000:.0f} x {COL_H*1000:.0f} mm")
    print(f"  fc' = {FC_PRIME/1e6:.0f} MPa, fy = {FY/1e6:.0f} MPa")
    print(f"  Gravity UDL: {W_BEAM_DEAD/1e3:.0f} kN/m")
    print(f"  Lateral (factored EQ): {F_LATERAL_PER_FLOOR/1e3:.0f} kN/floor")
    print()

    # --- Analysis ---
    model, col_tags, beam_tags = build_frame()
    LinearStaticAnalysis(model).run()
    mat_design = ConcreteMaterial(fc_prime=FC_PRIME, fy=FY)

    # --- BEAM base design + capacity-design shear ---
    print("=" * 78)
    print("BEAM design: base flexure/shear + ACI 18.6.5 capacity-design shear")
    print("=" * 78)
    print(f"  {'Member':<14} | {'M_u':>9} | {'V_u':>8} | {'M_pr':>9} | "
          f"{'V_e':>9} | {'V_design':>9} | {'Stirrup':>14}")
    print("  " + "-" * 90)

    beam_sections = {}        # for SCWB later
    for etag, level, bay in beam_tags:
        env = member_envelope(model.elements[etag])
        demand = BeamDesignDemand(
            M_u_positive=env["M_max"],
            M_u_negative=env["M_max"],     # symmetric for SMF
            V_u=env["V_max"],
        )
        base = RcMemberDesigner.design_beam(
            b=BEAM_B, h=BEAM_H, material=mat_design,
            demand=demand, cover=0.050,
        )
        if not base.success:
            print(f"  Beam L{level}-B{bay:<5} | DESIGN FAILED: {base.notes}")
            continue
        beam_sec = base.section
        beam_sections[etag] = beam_sec
        # Capacity-design shear
        cs = capacity_design_shear(
            beam_sec, beam_sec,
            L_n=L_BAY - COL_B,         # clear span
            w_u=W_BEAM_DEAD,
            V_u_analysis=env["V_max"],
        )
        # Re-design stirrups for the capacity-design shear
        from femsolver.design.concrete import design_stirrup_spacing
        new_shear = design_stirrup_spacing(beam_sec, V_u=cs.V_design)
        print(f"  Beam L{level}-B{bay:<5} | "
              f"{env['M_max']/1e3:>6.1f}kNm | "
              f"{env['V_max']/1e3:>5.1f}kN | "
              f"{cs.M_pr_left/1e3:>6.1f}kNm | "
              f"{cs.V_e/1e3:>6.1f}kN | "
              f"{cs.V_design/1e3:>6.1f}kN | "
              f"#3@{new_shear.s_recommended*1000:>4.0f}mm")
    print()

    # --- COLUMN base design ---
    print("=" * 78)
    print("COLUMN base design + confinement detailing (ACI 18.7.5)")
    print("=" * 78)
    print(f"  {'Member':<14} | {'P_u':>9} | {'M_u':>9} | {'rho':>5} | "
          f"{'l_o':>6} | {'s_o':>7} | {'Conf OK':>8}")
    print("  " + "-" * 75)
    col_sections = {}
    for etag, story, col in col_tags:
        env = member_envelope(model.elements[etag])
        P_u = max(0.0, env["P_compr"])
        demand = ColumnDesignDemand(
            P_u=P_u, M_u=env["M_max"], V_u=env["V_max"],
        )
        base = RcMemberDesigner.design_column(
            b=COL_B, h=COL_H, material=mat_design,
            demand=demand, cover=0.060,
        )
        if not base.success:
            print(f"  Col  S{story}-C{col:<5} | DESIGN FAILED")
            continue
        col_sec = base.section
        col_sections[etag] = col_sec
        # Confinement detailing
        cd = confined_concrete_detailing(
            col_sec, column_clear_height=H_STORY - BEAM_H,
        )
        ok_flag = "ok" if cd.passes else "FAIL"
        print(f"  Col  S{story}-C{col:<5} | "
              f"{P_u/1e3:>6.1f}kN | "
              f"{env['M_max']/1e3:>6.1f}kNm | "
              f"{base.rho*100:>4.2f}% | "
              f"{cd.l_o*1000:>4.0f}mm | "
              f"{cd.s_o_required*1000:>4.0f}mm | "
              f"{ok_flag:>8}")
    print()

    # --- SCWB check at every interior joint ---
    print("=" * 78)
    print("SCWB check at each interior joint (ACI 18.7.3.2: ratio >= 6/5)")
    print("=" * 78)
    print(f"  {'Joint':<10} | {'sum Mn_col':>11} | {'sum Mn_beam':>12} | "
          f"{'ratio':>6} | {'req':>5} | {'pass':>5}")
    print("  " + "-" * 72)
    # Build lookups
    col_by_story_col = {(s, c): tag for tag, s, c in col_tags}
    beam_by_level_bay = {(l, b): tag for tag, l, b in beam_tags}
    # Loop over interior joints (level 1 to N_STORY, col 1 to N_BAY+1)
    n_col = N_BAY + 1
    for level in range(1, N_STORY + 1):
        for col in range(1, n_col + 1):
            # Columns at this joint: below (story=level) and above (story=level+1 if exists)
            col_etags_here = []
            if (level, col) in col_by_story_col:
                col_etags_here.append(col_by_story_col[(level, col)])
            if (level + 1, col) in col_by_story_col:
                col_etags_here.append(col_by_story_col[(level + 1, col)])
            if not col_etags_here:
                continue
            # Beams at this joint: bay (col-1) and bay (col)
            beam_etags_here = []
            if (level, col - 1) in beam_by_level_bay:
                beam_etags_here.append(beam_by_level_bay[(level, col - 1)])
            if (level, col) in beam_by_level_bay:
                beam_etags_here.append(beam_by_level_bay[(level, col)])
            # Compute M_n for each
            col_M_n = []
            for cet in col_etags_here:
                if cet not in col_sections:
                    continue
                env_c = member_envelope(model.elements[cet])
                P_u_c = max(0.0, env_c["P_compr"])
                surface = column_interaction_surface(col_sections[cet])
                # Use the M_n at the P=P_u line (more refined than just M_o)
                M_n_at_P = surface.phi_M_n_at_P_u(P_u_c) / 0.65 if surface.phi_M_n_at_P_u(P_u_c) > 0 else surface.M_o / 0.65
                # Approximate -- phi varies; use simpler M_n estimate from
                # nominal interaction max M_n (M_o / phi at flex-controlled)
                col_M_n.append(abs(M_n_at_P))
            beam_M_n = []
            for bet in beam_etags_here:
                if bet not in beam_sections:
                    continue
                fc = beam_flexural_strength(beam_sections[bet])
                beam_M_n.append(fc.M_n)
            if not beam_M_n:
                continue
            sb = scwb_check(
                column_M_n=col_M_n, beam_M_n=beam_M_n, code="ACI",
            )
            pass_flag = "ok" if sb.passes else "FAIL"
            print(f"  L{level}-C{col:<7} | "
                  f"{sb.sum_M_nc/1e3:>8.1f}kNm | "
                  f"{sb.sum_M_nb/1e3:>9.1f}kNm | "
                  f"{sb.ratio:>5.2f}  | "
                  f"{sb.ratio_required:>4.2f} | "
                  f"{pass_flag:>5}")
    print()

    print("Reading the result:")
    print("* Base member design (Phase 29) sizes beam + column rebar for")
    print("  flexural + axial demand from the analysis envelope.")
    print("* The capacity-design shear V_e (Phase 32.2) usually exceeds")
    print("  the analysis V_u for SMF beams, so stirrup spacing tightens.")
    print("* Confinement at column ends (Phase 32.3) constrains hoop")
    print("  spacing within l_o (the largest of column depth, H/6, 450 mm).")
    print("* SCWB (Phase 32.1) -- the joint-level mechanism check that")
    print("  prevents column hinging. If a joint fails, enlarge columns or")
    print("  reduce beam steel. For this small frame upper-story joints")
    print("  often fail since columns terminate; in practice they're")
    print("  exempted (top-story joint allowed per ACI 18.7.3.2 exception).")


if __name__ == "__main__":
    main()
