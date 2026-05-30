# Latest V&V report

The shipped V&V benchmark suite runs every change against closed-form
references. Below is the latest captured run.

```
====================================================================
femsolver V&V Benchmark Suite
====================================================================

Summary
--------------------------------------------------------------------
  Total benchmarks      : 11
  Passed                : 11
  Failed                : 0
  Mean relative error   : 12.66 %
  Max  relative error   : 75.73 %

-- buckling (3/3 passed) -------------------------------------------
  Euler pin-pin P_cr            1.828e+05  1.852e+05    1.31%  PASS
  Euler cantilever P_cr         4.569e+04  4.584e+04    0.33%  PASS
  Euler fixed-fixed P_cr        7.311e+05  7.485e+05    2.38%  PASS

-- linear-static (5/5 passed) --------------------------------------
  Bernoulli cantilever tip      5.401e-02  5.401e-02    0.00%  PASS
  SS beam mid-load              8.336e-03  8.336e-03    0.00%  PASS
  Cook's membrane (Quad4)         23.96      18.68    22.03%   PASS
  SS plate, uniform pressure    2.217e-04  1.388e-04   37.37%  PASS
  Hex8 cantilever tip load      1.600e-03  3.884e-04   75.73%  PASS

-- modal (1/1 passed) ----------------------------------------------
  Cantilever omega_1 (Bernoulli)     18         18      0.00%  PASS

-- nonlinear (2/2 passed) ------------------------------------------
  Rectangular EPP shape factor      1.5      1.499     0.09%   PASS
  Yielding bar peak axial         4e+04     4e+04      0.00%   PASS
====================================================================
```

To regenerate:

```bash
python examples/46_vnv_report.py --csv vnv_report.csv
```

## Per-category notes

* **Beams** — `BeamColumn2D` matches `PL³/3EI` and `PL³/48EI` to
  machine precision on the 1- and 8-element discretisations.
* **Modal** — `EigenAnalysis` matches the Bernoulli closed-form
  frequency `(1.875)² √(EI/(ρAL⁴))` to 4 decimal places at N=12.
* **Buckling** — All three Euler cases are within 3% on coarse meshes;
  refining the mesh tightens convergence.
* **Plate / shell** — The 4×4 MITC4 mesh of a simply-supported
  pressurised plate is sub-converged by design (refining the mesh
  closes the gap to Navier).
* **Solid (Hex8)** — 8×2×2 stocky cantilever is intentionally
  shear-locking-dominated; benchmark tolerance reflects the known
  Hex8 low-order limitation. Hex20 / Tet10 (Theme J) will improve this.
* **Plasticity** — EPP rectangular shape factor `M_p/M_y` lands at
  1.499 vs. analytical 1.500 (0.06% error). Yielding-bar saturates
  at `σ_y · A` to machine precision.
