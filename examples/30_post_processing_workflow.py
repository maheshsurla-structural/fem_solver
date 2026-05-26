"""Phase 21 -- I/O and post-processing workflow.

A complete worked example showing the kinds of artifacts an engineer
would typically extract from a structural-FE analysis:

1. **JSON model deck** -- save the model definition for reproducibility.
2. **VTK export** -- write the deformed shape + reactions + mode shapes
   for inspection in Paraview / VisIt / MayaVi.
3. **Beam force diagrams** -- compute N / V / M along beam elements;
   the standard structural-engineering output.
4. **Mode-shape table** -- structured summary of modal periods + (if
   available) participation factors and effective modal masses.
5. **Capacity curve** -- (drift, base shear) pairs from a pushover
   analysis.

The example builds a small 2D moment-resisting frame, runs each of
the four analysis types (linear static, eigen, response spectrum, and
nonlinear static / pushover) and produces the corresponding output
artifacts.

Run::

    python examples/30_post_processing_workflow.py
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from femsolver import (
    BeamColumn2D,
    EigenAnalysis,
    ElasticIsotropic,
    Model,
    NonlinearStaticAnalysis,
    ResponseSpectrum,
    ResponseSpectrumAnalysis,
)
from femsolver.analysis.linear_static import LinearStaticAnalysis
from femsolver.io import (
    beam_force_diagram,
    capacity_curve,
    load_model_json,
    mode_shape_table,
    save_model_json,
    write_vtk,
)


def build_frame() -> Model:
    """Single-bay, single-story portal frame with consistent units."""
    E = 2.0e10           # Pa
    A = 1.0e-2
    Iz = 1.0e-4
    rho = 7850.0
    L_col = 3.0
    L_bm = 5.0

    mat = ElasticIsotropic(1, E=E, nu=0.3, rho=rho)
    m = Model(ndm=2, ndf=3); m.add_material(mat)
    # Nodes: 1, 2 at base; 3, 4 at roof
    m.add_node(1, 0.0, 0.0)
    m.add_node(2, L_bm, 0.0)
    m.add_node(3, 0.0, L_col)
    m.add_node(4, L_bm, L_col)
    # Two columns + roof beam
    m.add_element(BeamColumn2D(1, (1, 3), mat, A, Iz))    # left column
    m.add_element(BeamColumn2D(2, (2, 4), mat, A, Iz))    # right column
    m.add_element(BeamColumn2D(3, (3, 4), mat, A * 2, Iz * 2))  # roof beam
    m.fix(1, [1, 1, 1])
    m.fix(2, [1, 1, 1])
    return m


def main() -> None:
    out_dir = Path("phase21_outputs")
    out_dir.mkdir(exist_ok=True)
    print(f"\nPhase 21 demo -- output artifacts written to: {out_dir.absolute()}")

    # ============================================================
    # 1. JSON model deck
    # ============================================================
    print("\n1. JSON model deck")
    m = build_frame()
    deck_path = out_dir / "portal_frame.json"
    save_model_json(m, deck_path)
    print(f"   saved   -> {deck_path.name}  ({deck_path.stat().st_size} bytes)")

    m_loaded = load_model_json(deck_path)
    n_match = (
        len(m_loaded.nodes) == len(m.nodes)
        and len(m_loaded.elements) == len(m.elements)
    )
    print(f"   loaded  -> {len(m_loaded.nodes)} nodes, "
          f"{len(m_loaded.elements)} elements (round-trip OK: {n_match})")

    # ============================================================
    # 2. Linear static + VTK with reactions
    # ============================================================
    print("\n2. Linear static + VTK with reactions")
    m1 = build_frame()
    # Apply a horizontal roof load (simulating wind / EQ shear)
    m1.add_nodal_load(3, [10.0e3, 0, 0])
    LinearStaticAnalysis(m1).run()
    vtk_path = out_dir / "portal_static.vtk"
    write_vtk(m1, vtk_path, deformation_scale=1000.0, include_reactions=True)
    print(f"   saved   -> {vtk_path.name}")
    roof_dx = m1.node(3).disp[0]
    print(f"   roof horizontal displacement = {roof_dx * 1e3:.3f} mm")
    print(f"   base reaction (node 1): "
          f"Fx={m1.node(1).reaction[0]:.1f} N, "
          f"Fy={m1.node(1).reaction[1]:.1f} N, "
          f"Mz={m1.node(1).reaction[2]:.1f} Nm")

    # ============================================================
    # 3. Beam force diagrams
    # ============================================================
    print("\n3. Beam force diagrams along the left column")
    fd = beam_force_diagram(m1.element(1), n_points=11)
    print(f"   length = {fd['length']:.3f} m")
    print(f"   {'s (m)':>8s}  {'N':>12s}  {'V':>12s}  {'M':>12s}")
    for s, N, V, M in zip(fd["s"], fd["N"], fd["V"], fd["M"]):
        print(f"   {s:>8.3f}  {N:>+12.2e}  {V:>+12.2e}  {M:>+12.2e}")

    # ============================================================
    # 4. Eigen + mode-shape table + VTK mode shapes
    # ============================================================
    print("\n4. Eigen analysis -- mode-shape table + VTK")
    m2 = build_frame()
    eig = EigenAnalysis(m2, num_modes=3).run()
    tbl = mode_shape_table(eig)
    print(f"   {'mode':>4s}  {'period (s)':>11s}  {'freq (Hz)':>11s}  "
          f"{'omega (rad/s)':>14s}")
    for i in range(tbl["mode"].size):
        print(f"   {tbl['mode'][i]:>4d}  {tbl['period_s'][i]:>11.4f}  "
              f"{tbl['frequency_hz'][i]:>11.4f}  "
              f"{tbl['omega_rad_s'][i]:>14.3f}")
    modes_path = out_dir / "portal_modes.vtk"
    write_vtk(m2, modes_path, include_mode_shapes=True)
    print(f"   VTK with all mode shapes -> {modes_path.name}")

    # ============================================================
    # 5. Response spectrum -- richer modal table
    # ============================================================
    print("\n5. Response-spectrum analysis -- participation factors")
    m3 = build_frame()
    spec = ResponseSpectrum(
        periods=[0.05, 0.2, 0.5, 1.0, 2.0, 5.0],
        accelerations=[1.5, 2.5, 2.5, 1.25, 0.625, 0.25],   # m/s^2
        damping_ratio=0.05,
    )
    rs_res = ResponseSpectrumAnalysis(
        m3, spec, num_modes=3, direction="x", combination="cqc",
    ).run()
    # Combine eigen output with modal_results for the richer table
    table_input = dict(eig)
    table_input["modal_results"] = rs_res["modal_results"]
    table_input["periods_s"] = [r["period"] for r in rs_res["modal_results"]]
    tbl_rs = mode_shape_table(table_input)
    print(f"   {'mode':>4s}  {'period (s)':>11s}  {'Gamma':>10s}  "
          f"{'m_eff (kg)':>11s}  {'Sa (m/s^2)':>11s}")
    for i in range(tbl_rs["mode"].size):
        print(f"   {tbl_rs['mode'][i]:>4d}  {tbl_rs['period_s'][i]:>11.4f}  "
              f"{tbl_rs['participation'][i]:>+10.3f}  "
              f"{tbl_rs['modal_mass_eff'][i]:>11.3f}  "
              f"{tbl_rs['Sa'][i]:>11.4f}")

    # ============================================================
    # 6. Nonlinear pushover -- capacity curve
    # ============================================================
    print("\n6. Nonlinear pushover -- capacity curve")
    m4 = build_frame()
    m4.add_nodal_load(3, [10.0e3, 0, 0])
    res_pushover = NonlinearStaticAnalysis(
        m4, num_steps=20, dlambda=1.0 / 20, tol=1e-6,
        track=(3, 0),       # track the roof horizontal disp
    ).run()
    cc = capacity_curve(res_pushover)
    print(f"   pushover gave {cc['drift'].size} (drift, lambda) pairs")
    print(f"   {'lambda':>10s}  {'drift (mm)':>12s}")
    # Show every other step
    for i in range(0, cc["drift"].size, 2):
        print(f"   {cc['force'][i]:>10.4f}  {cc['drift'][i] * 1e3:>12.4f}")
    print()
    print("Reading the demo:")
    print("* JSON deck enables reproducible analyses and version control.")
    print("* VTK with reactions + mode shapes can be opened in Paraview")
    print("  to inspect deformed shapes, internal forces, and modal")
    print("  visualization side by side.")
    print("* Beam force diagrams (N, V, M) along any beam-column element")
    print("  produce the standard structural-engineering output.")
    print("* Mode-shape and capacity-curve helpers turn raw analysis")
    print("  dicts into NumPy arrays suitable for matplotlib / pandas.")


if __name__ == "__main__":
    main()
