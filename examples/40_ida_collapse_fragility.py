"""Phase 25.5 -- IDA + collapse fragility (capstone).

End-to-end Vamvatsikos-Cornell Incremental Dynamic Analysis (IDA)
followed by lognormal collapse-fragility fitting (Baker 2015).

The pipeline:

1. Build a 3-story 2-D shear-stick model with concentrated story mass.
2. Define a small suite of synthetic ground motions (Ricker pulses with
   varied centre-frequency and timing) to stand in for a record set.
3. Sweep PGA from 0.05 g to 1.20 g via :func:`multi_record_ida`, running
   a fresh nonlinear-transient analysis at each (record, IM) cell.
4. Detect collapse per record via the three-criteria detector
   (drift > 10 percent, NLTHA non-convergence, slope flatlining).
5. Fit a lognormal fragility (theta, beta) by both method-of-moments
   (Baker 2015) and censored MLE (FEMA P695 Appx F), and print
   P(collapse | IM) at several IM levels.

Run::

    python examples/40_ida_collapse_fragility.py
"""
from __future__ import annotations

import math

import numpy as np

from femsolver import (
    BeamColumn2D,
    ElasticIsotropic,
    Model,
    RayleighDamping,
    detect_collapse,
    fit_collapse_fragility,
    max_drift_edp,
    multi_record_ida,
    pga_scale_factor,
)
from femsolver.analysis.eigen import EigenAnalysis


# ============================================================ model factory

def make_model_factory(
    *,
    n_story: int = 3,
    story_height: float = 3.0,
    E: float = 2.0e10,
    A: float = 0.5,
    Iz: float = 4.0e-4,
    rho: float = 2500.0,
):
    """Return a zero-argument callable producing a fresh stick model.

    Each call yields a brand-new ``Model`` (no shared state) so each
    IDA point starts from rest -- required by the V-C single-record
    definition.

    Mass is supplied through the material ``rho`` (kg/m^3): each beam's
    consistent mass matrix is assembled by the element. Defaults give
    a soft-stiffness / heavy-mass stick with first-mode period above
    1 s so the swept PGA range is sufficient to drive drift past 10%
    (the V-C collapse threshold used by ``detect_collapse``).
    """
    def factory() -> Model:
        mat = ElasticIsotropic(1, E=E, nu=0.3, rho=rho)
        m = Model(ndm=2, ndf=3)
        m.add_material(mat)
        for i in range(n_story + 1):
            m.add_node(i + 1, 0.0, i * story_height)
        for i in range(n_story):
            m.add_element(BeamColumn2D(i + 1, (i + 1, i + 2),
                                        mat, A, Iz))
        m.fix(1, [1, 1, 1])
        return m

    return factory


# ============================================================ ground motions

def make_ricker(amp: float, fp: float, t0: float):
    """Centred Ricker (Mexican-hat) pulse.

    Peak amplitude is ``amp`` (m/s^2). Centre frequency ``fp`` (Hz),
    centred at ``t0`` (s).
    """
    def a_g(t: float) -> float:
        tau = math.pi * fp * (t - t0)
        return amp * (1.0 - 2.0 * tau ** 2) * math.exp(-tau ** 2)
    return a_g


# ============================================================ main

def main() -> None:
    print("=" * 70)
    print("Phase 25.5 -- IDA + collapse fragility")
    print("=" * 70)

    factory = make_model_factory()

    # ----- Mode-1 period sanity check -------------------------------
    m_eig = factory()
    eig = EigenAnalysis(m_eig, num_modes=3)
    eig.run()
    T1 = 2.0 * math.pi / math.sqrt(eig.eigenvalues[0])
    print(f"\nFundamental period T1 = {T1:.3f} s")

    # ----- Ground-motion suite --------------------------------------
    # 4 synthetic Ricker pulses with different centre-frequencies and
    # timings -- a placeholder for a real record set.
    t_end = 8.0
    dt = 0.01
    # Ricker centre-frequencies near the structure's first-mode
    # frequency (~ 0.5-0.7 Hz) so the pulse efficiently excites the
    # fundamental mode and produces a spread of collapse IMs across
    # the suite.
    suite_specs = [
        ("rec_low_fp",   0.4, 3.0),
        ("rec_match_fp", 0.6, 2.5),
        ("rec_high_fp",  0.8, 2.5),
        ("rec_offset",   0.5, 4.0),
    ]
    records = []
    for name, fp, t0 in suite_specs:
        ag = make_ricker(amp=1.0, fp=fp, t0=t0)   # unit-amp template
        records.append({
            "name": name,
            "accel_function": ag,
            "t_end": t_end,
            "dt": dt,
            "scale_fn": pga_scale_factor(ag, t_end=t_end, dt=dt),
        })
    print(f"Suite: {len(records)} synthetic records, "
          f"t_end = {t_end:.1f} s, dt = {dt:.3f} s")

    # ----- IM sweep -------------------------------------------------
    g = 9.81
    IM_levels = np.array(
        [0.10, 0.25, 0.50, 0.80, 1.20, 1.80, 2.50, 3.50]
    ) * g
    print(f"IM levels (PGA, m/s^2): "
          + ", ".join(f"{im:.2f}" for im in IM_levels))

    # ----- EDP extractor + damping ----------------------------------
    edp = max_drift_edp(story_node_tags=[2, 3, 4], direction=0,
                        base_node_tag=1)
    omega_1 = math.sqrt(eig.eigenvalues[0])
    omega_2 = math.sqrt(eig.eigenvalues[-1])
    damping = RayleighDamping.from_modes(
        omega_1, 0.05, omega_2, 0.05,
    )

    # ----- Run the IDA suite ----------------------------------------
    print("\nRunning IDA ...")

    def on_progress(rec_idx, name, n_total):
        print(f"  [{rec_idx + 1}/{n_total}] {name}")

    summary = multi_record_ida(
        model_factory=factory,
        records=records,
        IM_levels=IM_levels,
        edp_extractor=edp,
        direction="x",
        damping=damping,
        drift_limit=0.10,
        on_progress=on_progress,
    )

    # ----- Per-record collapse IM ------------------------------------
    print("\nPer-record collapse outcomes:")
    print(f"  {'record':<18}{'collapse IM (g)':>18}{'cause':>22}")
    print("  " + "-" * 56)
    for rec, coll in zip(summary.records, summary.collapse_results):
        im_str = (f"{coll.collapse_IM / g:.3f}"
                  if math.isfinite(coll.collapse_IM) else "no-collapse")
        print(f"  {rec.record_name:<18}{im_str:>18}{coll.cause:>22}")

    print(f"\n  records collapsed   : {summary.n_collapsed}/{len(records)}")
    if math.isfinite(summary.median_collapse_IM):
        print(f"  median collapse IM : "
              f"{summary.median_collapse_IM / g:.3f} g")

    # ----- Fragility fit --------------------------------------------
    collapse_IMs = summary.collapse_IMs
    finite_mask = np.isfinite(collapse_IMs)
    nc_IM_max = np.full_like(collapse_IMs, float(IM_levels[-1]))
    nc_IM_max = nc_IM_max[~finite_mask]    # only non-collapsed records

    if int(np.sum(finite_mask)) >= 2:
        fit = fit_collapse_fragility(
            collapse_IMs[finite_mask],
            no_collapse_IM_max=nc_IM_max if nc_IM_max.size else None,
        )
        print(f"\nLognormal fragility fit ({fit.method}):")
        print(f"  theta (median collapse IM) = {fit.theta / g:.3f} g")
        print(f"  beta  (log-std dispersion) = {fit.beta:.3f}")
        print(f"  records used                = "
              f"{fit.n_collapsed} collapsed / {fit.n_records} total")

        print("\nP(collapse | IM):")
        print(f"  {'IM (g)':>10}{'P_col':>12}")
        for q in [0.20, 0.35, 0.55, 0.80, 1.10, 1.50, 2.00]:
            print(f"  {q:>10.2f}{fit.P_collapse(q * g):>12.3f}")
    else:
        print("\nToo few records collapsed for a fragility fit "
              "(need >= 2).")

    print("\n" + "=" * 70)
    print("Phase 25 complete: IDA driver + collapse + fragility OK.")
    print("=" * 70)


if __name__ == "__main__":
    main()
