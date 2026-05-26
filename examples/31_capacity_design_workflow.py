"""Phase 19 -- complete performance-based seismic design (PBSD) workflow.

A 5-story shear-stick frame is taken through the full ASCE 41 / EN 1998
capacity-design procedure:

1. **Eigen analysis** -- modal periods, participation factors.
2. **Pushover-to-target** -- displacement-controlled pushover using
   the first-mode invariant load pattern, recording the capacity curve.
3. **Bilinearization** -- equal-area equivalent bilinear of the
   capacity curve (FEMA 356 K_i / d_y / F_y).
4. **Equivalent SDOF + N2 target displacement** -- convert the MDOF
   pushover to an equivalent SDOF and find the target roof
   displacement under a code design spectrum.
5. **ASCE 41 Coefficient Method** -- second-opinion target
   displacement using ``d_t = C0 C1 C2 Sa (T/2pi)^2``.
6. **Story drifts at the target** -- interstory drift ratios at each
   story under the demanded displacement.
7. **Multi-axis combination** -- ASCE 7 100-30 rule applied to a
   sample (x, y) response.

Run::

    python examples/31_capacity_design_workflow.py
"""
from __future__ import annotations

import math

import numpy as np

from femsolver import (
    BeamColumn2D,
    EigenAnalysis,
    ElasticIsotropic,
    Model,
    PushoverToTarget,
    ResponseSpectrum,
    bilinearize_capacity_curve,
    coefficient_method_target,
    equivalent_sdof,
    n2_target_displacement,
    seismic_combination,
    story_drifts,
)


def build_5_story_frame(*, n_story: int = 5, L: float = 3.0) -> Model:
    """A 5-story shear-stick: BeamColumn2D segments stacked
    vertically. Heavier section to give a realistic seismic period."""
    E = 2.0e10
    A = 1.0e-2
    Iz = 1.0e-3            # increased -> shorter period
    rho = 7850.0
    mat = ElasticIsotropic(1, E=E, nu=0.3, rho=rho)
    m = Model(ndm=2, ndf=3); m.add_material(mat)
    for i in range(n_story + 1):
        m.add_node(i + 1, 0.0, i * L)
    for i in range(n_story):
        m.add_element(BeamColumn2D(i + 1, (i + 1, i + 2), mat, A, Iz))
    m.fix(1, [1, 1, 1])
    return m


def main() -> None:
    n_story = 5
    L_story = 3.0
    print(f"\n5-story shear-stick frame -- capacity design workflow")
    print(f"  {n_story} stories @ {L_story} m each (total height {n_story*L_story} m)")
    print()

    # ============================================================
    # 1. Eigen analysis -- modal properties
    # ============================================================
    print("1. Eigen analysis")
    m_eig = build_5_story_frame(n_story=n_story, L=L_story)
    eig = EigenAnalysis(m_eig, num_modes=n_story).run()
    periods = eig["periods_s"]
    print(f"   modal periods (s): {[f'{p:.4f}' for p in periods]}")

    # ============================================================
    # 2. Run pushover-to-target -- record capacity curve
    # ============================================================
    print("\n2. Pushover-to-target (invariant load pattern proportional to story * 1 kN)")
    m = build_5_story_frame(n_story=n_story, L=L_story)
    # Inverted-triangle load pattern (mimics first-mode invariant)
    for i in range(1, n_story + 1):
        m.add_nodal_load(i + 1, [float(i) * 1.0e3, 0.0, 0.0])
    roof_node = n_story + 1
    target_drift = 0.20      # m = 200 mm at the roof
    pt = PushoverToTarget(
        m, target_drift=target_drift, track=(roof_node, 0),
        num_steps=40, tol=1e-6,
    )
    res = pt.run()
    print(f"   {len(res['drift'])} steps, {res['total_iterations']} total iterations")
    print(f"   Final roof drift: {res['drift'][-1]*1000:.2f} mm")
    print(f"   Final base shear: {res['force'][-1]*1e-3:.2f} kN")
    print(f"   {'drift (mm)':>12s}  {'V_base (kN)':>12s}")
    # Show first 3 and last 3 points
    for i in (0, 1, 2, -3, -2, -1):
        print(f"   {res['drift'][i]*1000:>12.3f}  {res['force'][i]*1e-3:>12.3f}")

    # ============================================================
    # 3. Bilinearize the capacity curve
    # ============================================================
    print("\n3. Bilinearize the capacity curve (FEMA 356 / EC 8 equal area)")
    bl = bilinearize_capacity_curve(res["drift"], res["force"])
    print(f"   K_i  = {bl.K_i:.3e} N/m")
    print(f"   d_y  = {bl.d_y*1000:.2f} mm, F_y = {bl.F_y*1e-3:.2f} kN")
    print(f"   d_u  = {bl.d_u*1000:.2f} mm, F_u = {bl.F_u*1e-3:.2f} kN")
    print(f"   alpha (post-yield ratio) = {bl.alpha:.4f}")

    # ============================================================
    # 4. Equivalent SDOF + N2 target
    # ============================================================
    print("\n4. Equivalent SDOF + N2-method target displacement")
    # For the inverted-triangle pattern, Γ ≈ 1.25 and m_eff ≈ 0.85 * Σm_i
    # are typical first-mode values for regular frames.
    total_mass = 5.0 * 7850.0 * 1.0e-2 * L_story    # rho A L per story
    Gamma = 1.25
    m_eff = 0.85 * total_mass
    sdof = equivalent_sdof(res["drift"], res["force"],
                              Gamma=Gamma, m_eff=m_eff)
    print(f"   Gamma = {sdof.Gamma:.3f}, m_eff = {sdof.m_eff:.1f} kg")
    print(f"   SDOF effective period T* = {sdof.T_eff:.4f} s")
    # Bilinearize the SDOF curve
    bl_sdof = bilinearize_capacity_curve(sdof.d_star, sdof.F_star)
    print(f"   SDOF d_y* = {bl_sdof.d_y*1000:.2f} mm, F_y* = {bl_sdof.F_y*1e-3:.2f} kN")

    # Code-shape spectrum (simplified EC 8 type-1 ground type B)
    spec_periods = [0.0, 0.05, 0.15, 0.5, 1.0, 2.0, 5.0]
    spec_sa = [2.0, 2.5, 7.5, 7.5, 3.75, 1.875, 0.75]   # m/s^2 (Sa(T) shape)
    spec = ResponseSpectrum(periods=spec_periods[1:], accelerations=spec_sa[1:])

    n2 = n2_target_displacement(spec, sdof, bl_sdof, Tc=0.5)
    print(f"   N2 method:")
    print(f"     Sa(T*) = {n2['Sa_T_star']:.3f} m/s^2")
    print(f"     R = {n2['R']:.3f} (strength reduction factor)")
    print(f"     d_e* (elastic SDOF) = {n2['d_e_star']*1000:.2f} mm")
    print(f"     d_t* (inelastic SDOF) = {n2['d_t_star']*1000:.2f} mm")
    print(f"     d_t_top (MDOF target) = {n2['d_t_top']*1000:.2f} mm")

    # ============================================================
    # 5. ASCE 41 coefficient method
    # ============================================================
    print("\n5. ASCE 41 coefficient method (second opinion)")
    cm = coefficient_method_target(spec, T_eff=sdof.T_eff,
                                      C0=Gamma, C1=1.0, C2=1.0)
    print(f"   d_t_top (Coefficient Method) = {cm['d_t_top']*1000:.2f} mm")

    # ============================================================
    # 6. Story drifts AT the target displacement
    # ============================================================
    print("\n6. Story drifts AT target displacement (re-run pushover to target)")
    target_drift_actual = max(n2["d_t_top"], cm["d_t_top"])
    # Re-run pushover to the chosen target
    m2 = build_5_story_frame(n_story=n_story, L=L_story)
    for i in range(1, n_story + 1):
        m2.add_nodal_load(i + 1, [float(i) * 1.0e3, 0, 0])
    pt2 = PushoverToTarget(
        m2, target_drift=target_drift_actual, track=(roof_node, 0),
        num_steps=30, tol=1e-6,
    )
    pt2.run()
    sd = story_drifts(m2, [2, 3, 4, 5, 6], direction=0)
    print(f"   target roof drift = {target_drift_actual*1000:.2f} mm")
    print(f"   {'story':>6s}  {'disp (mm)':>10s}  {'drift (mm)':>11s}  "
          f"{'ratio %':>9s}")
    for i in range(5):
        print(f"   {sd['story'][i]:>6d}  "
              f"{sd['absolute_disp'][i]*1000:>10.3f}  "
              f"{sd['interstory_drift'][i]*1000:>11.3f}  "
              f"{sd['drift_ratio'][i]*100:>9.4f}")

    # ============================================================
    # 7. Multi-axis combination
    # ============================================================
    print("\n7. Multi-axis seismic combination (ASCE 7 100-30 rule)")
    # Suppose the same structure was analyzed for an orthogonal
    # direction giving a story drift of 0.45 * x value.
    responses_drift = {
        "x": sd["interstory_drift"][2],         # story 3 in x
        "y": 0.45 * sd["interstory_drift"][2],   # story 3 in y
    }
    drift_combined_100_30 = seismic_combination(
        responses_drift, rule="100-30"
    )
    drift_combined_srss = seismic_combination(
        responses_drift, rule="SRSS"
    )
    print(f"   Story-3 interstory drift, x:     "
          f"{responses_drift['x']*1000:.3f} mm")
    print(f"   Story-3 interstory drift, y:     "
          f"{responses_drift['y']*1000:.3f} mm")
    print(f"   100-30 combination:              "
          f"{drift_combined_100_30*1000:.3f} mm")
    print(f"   SRSS combination:                "
          f"{drift_combined_srss*1000:.3f} mm")

    print()
    print("Reading the workflow:")
    print("* The pushover capacity curve captures the structure's lateral")
    print("  force-displacement behavior under monotonic loading.")
    print("* Bilinearization gives K_i, F_y, d_y for the equivalent")
    print("  elastic-perfectly-plastic system.")
    print("* The N2 method maps the MDOF problem to an SDOF problem in")
    print("  the (d*, F*) plane, then uses the elastic spectrum + R-mu-T")
    print("  rules to compute target displacement.")
    print("* The ASCE 41 Coefficient Method gives a second opinion using")
    print("  empirical C-coefficients on the elastic spectrum demand.")
    print("* Story drift ratios at the target are the key acceptance")
    print("  criterion (typical limits: 1-2% for life safety).")
    print("* Multi-axis combination (100-30 or SRSS) computes the worst")
    print("  bidirectional demand for design checks.")


if __name__ == "__main__":
    main()
