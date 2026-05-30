"""Phase 31.5 -- Load combinations + envelope + ASCE 7-22 drift check.

A 3-story 2-bay steel moment frame is loaded with four distinct
load patterns (Dead, Live, Wind, Earthquake_x). The ASCE 7-22 LRFD
basic combinations are enumerated and applied via
:class:`EnvelopeAnalysis`, which:

* runs LinearStaticAnalysis for each combination,
* records per-member end forces for every combo,
* assembles per-member max/min envelopes (with the governing combo
  name for each).

Then a multi-combo drift check (:func:`drift_check_worst_combo`)
loops over the same combos and reports the worst-case story drift
per ASCE 7-22 §12.12, amplified by C_d / I_e and compared against
the Risk-Category-II limit of 0.020·h_sx.

This ties together the analysis side (Phase 1-23) with the load-
combo / envelope / drift machinery (Phase 31), feeding directly into
the design drivers (Phase 29 + 30 ACI/AISC).

Run::

    python examples/51_envelope_drift_check.py
"""
from __future__ import annotations

import numpy as np

from femsolver import (
    BeamColumn2D,
    ElasticIsotropic,
    EnvelopeAnalysis,
    LinearStaticAnalysis,
    LoadCombination,
    LoadPattern,
    Model,
    asce7_lrfd_combinations,
    drift_check_worst_combo,
)


# ============================================================ model

N_STORY = 3
N_BAY = 2
H_STORY = 3.5         # m
L_BAY = 6.0           # m

# Steel W14x90 columns (Ix = 415e-6), W18x60 beams (Ix = 410e-6)
COL_A = 0.0171
COL_I = 415.0e-6
BEAM_A = 0.0114
BEAM_I = 410.0e-6
E = 200.0e9


def build_frame() -> tuple[Model, list, list, list]:
    """Return (model, beam_etags, col_etags, story_node_tags)."""
    mat = ElasticIsotropic(1, E=E, nu=0.30, rho=7850.0)
    m = Model(ndm=2, ndf=3); m.add_material(mat)
    n_col = N_BAY + 1
    for j in range(N_STORY + 1):
        for i in range(n_col):
            tag = j * n_col + i + 1
            m.add_node(tag, i * L_BAY, j * H_STORY)
    etag = 1
    col_etags: list[int] = []
    for j in range(N_STORY):
        for i in range(n_col):
            n_b = j * n_col + i + 1
            n_t = (j + 1) * n_col + i + 1
            m.add_element(BeamColumn2D(etag, (n_b, n_t), mat, COL_A, COL_I))
            col_etags.append(etag); etag += 1
    beam_etags: list[int] = []
    for j in range(1, N_STORY + 1):
        for i in range(N_BAY):
            n_L = j * n_col + i + 1
            n_R = j * n_col + i + 2
            m.add_element(BeamColumn2D(etag, (n_L, n_R), mat, BEAM_A, BEAM_I))
            beam_etags.append(etag); etag += 1
    for i in range(n_col):
        m.fix(i + 1, [1, 1, 1])
    # Story representative nodes (leftmost column line)
    story_tags = [j * n_col + 1 for j in range(1, N_STORY + 1)]
    return m, beam_etags, col_etags, story_tags


# ============================================================ load patterns

# Define each pattern as a closure capturing the model. The
# `factor` argument scales the loads for combination application.

def dead(model, factor=1.0):
    """Floor gravity: 25 kN/m UDL on every beam."""
    for el in model.elements.values():
        # All BeamColumn2D in this model that have add_uniform_load
        # (we apply to all -- columns will have zero by their geometry;
        # really we only want beams)
        pass
    for etag in (10, 11, 12, 13, 14, 15):    # the 6 beam etags
        if etag in model.elements:
            model.elements[etag].add_uniform_load(-25e3 * factor)


def live(model, factor=1.0):
    """Reduced-live floor load: 12 kN/m on every beam."""
    for etag in (10, 11, 12, 13, 14, 15):
        if etag in model.elements:
            model.elements[etag].add_uniform_load(-12e3 * factor)


def wind(model, factor=1.0):
    """Wind: 60 kN per floor pushing in +x."""
    n_col = 3
    for j in range(1, 4):
        F_per_node = 60e3 / n_col * factor
        for i in range(n_col):
            tag = j * n_col + i + 1
            model.add_nodal_load(tag, [F_per_node, 0, 0])


def earthquake_x(model, factor=1.0):
    """Equivalent-static EQ in +x: 80 kN per floor."""
    n_col = 3
    for j in range(1, 4):
        F_per_node = 80e3 / n_col * factor
        for i in range(n_col):
            tag = j * n_col + i + 1
            model.add_nodal_load(tag, [F_per_node, 0, 0])


# ============================================================ main

def main() -> None:
    print("Phase 31.5 -- Load combinations + envelope + ASCE 7 drift")
    print("=" * 72)
    model, beam_etags, col_etags, story_tags = build_frame()
    print(f"  Frame: {N_STORY}-story x {N_BAY}-bay, total "
          f"{len(model.elements)} members")
    print(f"  Patterns: D, L, W, E (earthquake in +x)")
    print()

    patterns = {
        "D": LoadPattern("D", dead),
        "L": LoadPattern("L", live),
        "W": LoadPattern("W", wind),
        "E": LoadPattern("E", earthquake_x),
    }

    # Enumerate the basic ASCE 7-22 LRFD strength combinations
    combos = asce7_lrfd_combinations(include_snow_rain_roof_live=False)
    print(f"  Running {len(combos)} ASCE 7-22 LRFD combinations:")
    for c in combos:
        active = ", ".join(f"{f:+.1f}{n}" for n, f in c.factors.items())
        print(f"    {c.name:<35} -> {active}")
    print()

    # --- Envelope analysis ---
    env = EnvelopeAnalysis(model, patterns, combos).run()
    print(f"Envelope: {len(env.member_envelopes)} members tracked across "
          f"{len(env.combinations)} combinations")
    print()

    # --- Report governing M and V per beam ---
    print("BEAM force envelopes (worst-case |M| and |V| across all combos):")
    print(f"  {'Member':<7} | {'|M|_max':>12} | {'|V|_max':>12} | "
          f"{'M governing combo':<35}")
    print("  " + "-" * 75)
    # BeamColumn2D end_forces_local indices: 0,1,2 = F_xi, F_yi, M_zi
    #                                          3,4,5 = F_xj, F_yj, M_zj
    for etag in beam_etags:
        fe = env.member_envelopes[etag]
        # Worst-case absolute moment: max of |M_zi|, |M_zj|
        abs_max = fe.abs_max_per_component
        M_abs = max(abs_max[2], abs_max[5])
        V_abs = max(abs_max[1], abs_max[4])
        # Governing combo for max-abs M (pick whichever of M_zi/M_zj is larger)
        if abs_max[2] >= abs_max[5]:
            comp_idx = 2
        else:
            comp_idx = 5
        # The largest |M| could come from max_values or min_values
        if abs(fe.max_values[comp_idx]) >= abs(fe.min_values[comp_idx]):
            gov_combo = fe.max_combos[comp_idx]
        else:
            gov_combo = fe.min_combos[comp_idx]
        print(f"  {etag:<7} | {M_abs/1e3:>9.1f} kN.m | "
              f"{V_abs/1e3:>9.1f} kN | {gov_combo:<35}")
    print()

    # --- Report governing P per column ---
    print("COLUMN axial-force envelopes:")
    print(f"  {'Member':<7} | {'P_max (compr)':>14} | {'P_min':>14} | "
          f"{'governing':<35}")
    print("  " + "-" * 75)
    for etag in col_etags:
        fe = env.member_envelopes[etag]
        # F_xj is the axial at the upper node; for "compression"
        # we want the most-negative F_xj (member-end pulling toward
        # the column = pushing into the node from above).
        # The convention here: max(F_xj) is most tensile in member,
        # min(F_xj) is most compressive. Report compression as |min(F_xj)|.
        F_xj_max = fe.max_values[3]
        F_xj_min = fe.min_values[3]
        gov = fe.min_combos[3]    # most-compressive combo
        # Convert to compression-positive convention: -F_xj
        P_compr_max = max(-F_xj_min, -F_xj_max)
        P_compr_min = min(-F_xj_min, -F_xj_max)
        print(f"  {etag:<7} | {P_compr_max/1e3:>11.1f} kN | "
              f"{P_compr_min/1e3:>+11.1f} kN | {gov:<35}")
    print()

    # --- Drift check across all combos ---
    print("ASCE 7-22 sec 12.12 drift check (worst combo per story):")
    print("  C_d = 5.5 (SMF), I_e = 1.0 (Risk Cat II), Delta_a = 0.020 * h_sx")
    print()
    dc = drift_check_worst_combo(
        model, patterns, combos, story_tags,
        direction=0, base_node_tag=1,
        C_d=5.5, I_e=1.0, risk_category="II",
    )
    print(f"  {'Story':<6} | {'h_sx (m)':>9} | {'D_elastic (mm)':>15} | "
          f"{'D_amp (mm)':>11} | {'ratio':>7} | {'limit':>7} | "
          f"{'OK':<3} | {'governing':<25}")
    print("  " + "-" * 100)
    for i in range(len(dc.story_index)):
        flag = " " if dc.passes_per_story[i] else "X"
        print(f"  {dc.story_index[i]:<6} | "
              f"{dc.story_height[i]:>9.2f} | "
              f"{dc.delta_elastic[i]*1000:>+13.3f}   | "
              f"{dc.delta_amplified[i]*1000:>+9.3f}   | "
              f"{dc.drift_ratio[i]:>7.4f} | "
              f"{dc.drift_limit:>7.4f} | "
              f"{flag:<3} | {dc.governing_combo[i]:<25}")
    print()
    print(f"  Overall drift check: {'PASS' if dc.passes else 'FAIL'}")
    print()

    print("Reading the result:")
    print("* Each ASCE 7-22 LRFD combination is applied, analysed, and")
    print("  recorded. The envelope keeps the max/min of every end-force")
    print("  component plus the governing combination name for each.")
    print("* Beam moment envelopes are typically governed by 1.2D + 1.6L")
    print("  (high gravity) or by 1.2D + 1.0E + L (sway). Wind and EQ")
    print("  combinations differ in their factor on lateral.")
    print("* The drift check amplifies the elastic interstory drift by")
    print("  C_d / I_e (per ASCE 7-22 Table 12.2-1 and §12.12.1.1) and")
    print("  checks the ratio against the Risk-Category-II limit 0.020.")
    print("* These envelopes are exactly what RcMemberDesigner or")
    print("  SteelMemberDesigner consume -- so this example completes")
    print("  the analysis-to-design pipeline.")


if __name__ == "__main__":
    main()
