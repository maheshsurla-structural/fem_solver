"""Phase 26.4 -- Record selection & scaling feeding into IDA.

End-to-end workflow for a Performance-Based Seismic Engineering
record set:

1. Build a target design spectrum (ASCE 7-22 style) and a hypothetical
   GMPE-predicted median spectrum + log-stddev (the inputs a CMS
   computation would normally take from PSHA disaggregation).
2. Compute the **Conditional Mean Spectrum** anchored at the first-mode
   period ``T_1`` for ``epsilon = 1.5`` -- a typical PBSE target shape
   for MCE-level demands.
3. Generate a small suite of synthetic ground-motion records (Ricker
   pulses with varied centre-frequencies).
4. Compute each record's 5%-damped pseudo-acceleration spectrum and
   apply ASCE 7-22 amplitude scaling so the suite's geometric-mean
   spectrum matches the CMS over ``[0.2 T_1, 2.0 T_1]``.
5. Check the §16.2.4.1 acceptance criterion (suite-average never below
   90 percent of target across the period band).
6. Pipe the scaled suite into :func:`multi_record_ida` and detect
   collapse to demonstrate the closed loop.

Run::

    python examples/41_record_selection_to_ida.py
"""
from __future__ import annotations

import math

import numpy as np

from femsolver import (
    BeamColumn2D,
    ElasticIsotropic,
    Model,
    RayleighDamping,
    amplitude_scale_factor,
    baker_jayaram_correlation,
    compute_epsilon,
    compute_sdof_response_spectrum,
    conditional_mean_spectrum,
    detect_collapse,
    fit_collapse_fragility,
    max_drift_edp,
    multi_record_ida,
    pga_scale_factor,
    period_range_mask,
    record_response_spectrum,
    scale_record_suite,
)
from femsolver.analysis.eigen import EigenAnalysis


# ============================================================ model

def make_model_factory():
    """3-story stick cantilever, defaults tuned so T_1 ~ 1.8 s."""
    def factory() -> Model:
        mat = ElasticIsotropic(1, E=2.0e10, nu=0.3, rho=2500.0)
        m = Model(ndm=2, ndf=3)
        m.add_material(mat)
        for i in range(4):
            m.add_node(i + 1, 0.0, i * 3.0)
        for i in range(3):
            m.add_element(BeamColumn2D(i + 1, (i + 1, i + 2),
                                        mat, 0.5, 4.0e-4))
        m.fix(1, [1, 1, 1])
        return m
    return factory


# ============================================================ design spectrum

def asce7_design_spectrum(T: float) -> float:
    """Smoothed ASCE 7-style 2-corner design spectrum (m/s^2).

    Constant-acceleration plateau for T < T_s = 0.6 s, then 1/T decay.
    The peak Sa is ~1.0 g (a moderate MCE_R-class spectrum).
    """
    g = 9.81
    Sds = 1.0 * g           # short-period design Sa (1.0 g)
    Sd1 = 0.6 * g           # 1-second design Sa (0.6 g)
    Ts = Sd1 / Sds          # corner = 0.6 s
    T0 = 0.2 * Ts
    if T < T0:
        return float(0.4 * Sds + 0.6 * Sds * T / T0)
    if T < Ts:
        return float(Sds)
    return float(Sd1 / T)


# ============================================================ ground motions

def make_ricker(amp: float, fp: float, t0: float):
    def a_g(t: float) -> float:
        tau = math.pi * fp * (t - t0)
        return amp * (1.0 - 2.0 * tau ** 2) * math.exp(-tau ** 2)
    return a_g


# ============================================================ main

def main() -> None:
    print("=" * 72)
    print("Phase 26.4 -- Record selection + scaling + IDA pipeline")
    print("=" * 72)

    g = 9.81

    # ---- structure & periods ----------------------------------------
    factory = make_model_factory()
    m_eig = factory()
    eig = EigenAnalysis(m_eig, num_modes=3)
    eig.run()
    T1 = 2.0 * math.pi / math.sqrt(eig.eigenvalues[0])
    print(f"\nFundamental period T1 = {T1:.3f} s")

    # ---- analysis-period grid ---------------------------------------
    periods = np.logspace(math.log10(0.05), math.log10(6.0), 80)
    mask = period_range_mask(periods, T1=T1, low_mult=0.2, high_mult=2.0)
    print(f"Analysis periods: {periods.size}, "
          f"in band [0.2 T1, 2.0 T1] = "
          f"[{0.2*T1:.2f}, {2*T1:.2f}] s -> {int(np.sum(mask))} periods")

    # ---- target spectra ---------------------------------------------
    design_Sa = np.array([asce7_design_spectrum(T) for T in periods])

    # CMS target: GMPE median = 0.7 x design spectrum (a typical
    # under-prediction), sigma_lnSa = 0.6 across the band (a typical
    # NGA-West2 mid-period value). Anchor epsilon = 1.5 at T*.
    mu_Sa_median = 0.7 * design_Sa
    sigma_lnSa = np.full_like(periods, 0.6)
    eps = 1.5
    cms_Sa = conditional_mean_spectrum(
        T_star=T1, epsilon_star=eps,
        periods=periods,
        mu_lnSa=np.log(mu_Sa_median),
        sigma_lnSa=sigma_lnSa,
    )
    print(f"\nCMS conditioned at T* = T1 = {T1:.3f} s, epsilon = {eps}")
    print(f"  Sa target at T1 (design)     = {asce7_design_spectrum(T1)/g:.3f} g")
    print(f"  Sa CMS at T1                  = "
          f"{cms_Sa[np.argmin(np.abs(periods - T1))]/g:.3f} g")

    # ---- ground-motion suite ----------------------------------------
    t_end = 12.0
    dt = 0.01
    suite_specs = [
        ("rec_a_fp0p4",  0.4, 3.0),
        ("rec_b_fp0p55", 0.55, 3.0),
        ("rec_c_fp0p7",  0.7, 3.5),
        ("rec_d_fp0p5",  0.5, 4.0),
        ("rec_e_fp0p8",  0.8, 2.5),
    ]
    record_funcs = [make_ricker(1.0, fp, t0)
                    for (_, fp, t0) in suite_specs]
    record_names = [s[0] for s in suite_specs]
    print(f"\nSuite: {len(record_funcs)} synthetic records")

    # ---- compute per-record response spectra ------------------------
    print("Computing per-record response spectra ...")
    record_spectra = [
        record_response_spectrum(rf, t_end=t_end, dt=dt, periods=periods)
        for rf in record_funcs
    ]

    # ---- scale to CMS over [0.2 T_1, 2 T_1] -------------------------
    result = scale_record_suite(record_spectra, cms_Sa,
                                period_range_mask=mask)
    print("\nPer-record amplitude scale factors (matched to CMS):")
    for name, sf in zip(record_names, result.scale_factors):
        print(f"  {name:<18}  SF = {sf:8.3f}")
    print(f"\nSuite-average / target min ratio over [0.2T1, 2T1] = "
          f"{result.min_ratio:.3f}  "
          f"(ASCE 7-22 requires >= 0.90)")
    print(f"  Suite passes 90% criterion: {result.passes_90pct}")
    if not result.passes_90pct:
        print("  Note: synthetic Ricker pulses have narrow spectra; a real")
        print("        ASCE 7-22 suite would use 11+ broadband records.")

    # ---- feed scaled suite into IDA ---------------------------------
    print("\nFeeding scaled suite into IDA driver ...")
    # The scale factor we just computed brings the record's spectrum
    # to match the CMS *at the design intensity*. For IDA we then
    # sweep PGA explicitly (or Sa(T_1), but use PGA here for
    # simplicity).
    IM_levels = np.array([0.10, 0.25, 0.50, 0.80, 1.20, 1.80,
                           2.50, 3.50]) * g
    records_for_ida = []
    for (name, fp, t0), sf_match in zip(suite_specs, result.scale_factors):
        # The "calibrated" record: original Ricker times match-scale.
        def make_calibrated(fp_=fp, t0_=t0, sf_=sf_match):
            base = make_ricker(1.0, fp_, t0_)
            def a_g(t):
                return sf_ * base(t)
            return a_g
        ag_calibrated = make_calibrated()
        records_for_ida.append({
            "name": name,
            "accel_function": ag_calibrated,
            "t_end": t_end,
            "dt": dt,
            # IDA further scales the calibrated record to each PGA level
            "scale_fn": pga_scale_factor(ag_calibrated,
                                          t_end=t_end, dt=dt),
        })

    omega_1 = math.sqrt(eig.eigenvalues[0])
    omega_2 = math.sqrt(eig.eigenvalues[-1])
    damping = RayleighDamping.from_modes(omega_1, 0.05, omega_2, 0.05)

    def on_progress(rec_idx, name, n_total):
        print(f"  [{rec_idx + 1}/{n_total}] {name}")

    summary = multi_record_ida(
        model_factory=factory,
        records=records_for_ida,
        IM_levels=IM_levels,
        edp_extractor=max_drift_edp(
            story_node_tags=[2, 3, 4], direction=0, base_node_tag=1,
        ),
        direction="x",
        damping=damping,
        drift_limit=0.10,
        on_progress=on_progress,
    )

    print("\nPer-record collapse outcomes:")
    print(f"  {'record':<18}{'collapse IM (g)':>18}{'cause':>22}")
    print("  " + "-" * 56)
    for rec, coll in zip(summary.records, summary.collapse_results):
        im_str = (f"{coll.collapse_IM / g:.3f}"
                  if math.isfinite(coll.collapse_IM) else "no-collapse")
        print(f"  {rec.record_name:<18}{im_str:>18}{coll.cause:>22}")
    print(f"\n  records collapsed   : "
          f"{summary.n_collapsed}/{len(records_for_ida)}")
    if math.isfinite(summary.median_collapse_IM):
        print(f"  median collapse IM : "
              f"{summary.median_collapse_IM / g:.3f} g")

    # ---- fragility fit ---------------------------------------------
    finite_mask = np.isfinite(summary.collapse_IMs)
    nc_IM_max = np.full_like(summary.collapse_IMs, float(IM_levels[-1]))
    nc_IM_max = nc_IM_max[~finite_mask]
    if int(np.sum(finite_mask)) >= 2:
        fit = fit_collapse_fragility(
            summary.collapse_IMs[finite_mask],
            no_collapse_IM_max=nc_IM_max if nc_IM_max.size else None,
        )
        print(f"\nLognormal fragility ({fit.method}):")
        print(f"  theta = {fit.theta / g:.3f} g     beta = {fit.beta:.3f}")
        print(f"  records used: {fit.n_collapsed}/{fit.n_records}")

    print("\n" + "=" * 72)
    print("Phase 26 closed: record selection (CMS) + ASCE 7 scaling")
    print("                + IDA closed loop OK.")
    print("=" * 72)


if __name__ == "__main__":
    main()
