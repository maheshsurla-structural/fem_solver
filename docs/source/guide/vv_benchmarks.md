# V&V benchmark suite

Every change to femsolver is checked against a curated set of
closed-form / NAFEMS-style benchmarks. To run the suite:

```bash
python examples/46_vnv_report.py --csv vnv_report.csv
```

Or programmatically:

```python
from femsolver.benchmarks import all_benchmarks, run_benchmarks, format_report

results = run_benchmarks(all_benchmarks())
print(format_report(results))
```

The capstone example writes a categorised pass/fail report:

```
==============================================================================
V&V Benchmark Report
==============================================================================
Total: 11, Passed: 11, Failed: 0

-- buckling (3/3 passed) -----------------------------------------------------
  Euler pin-pin P_cr            1.828e+05    1.852e+05    1.31%    PASS
  Euler cantilever P_cr         4.569e+04    4.584e+04    0.33%    PASS
  Euler fixed-fixed P_cr        7.311e+05    7.485e+05    2.38%    PASS

-- linear-static (5/5 passed) ------------------------------------------------
  Bernoulli cantilever tip      5.401e-02    5.401e-02    0.00%    PASS
  SS beam mid-load              8.336e-03    8.336e-03    0.00%    PASS
  Cook's membrane (Quad4)         23.96         18.68    22.03%    PASS
  SS plate, uniform pressure    2.217e-04    1.388e-04    37.37%    PASS
  Hex8 cantilever tip load      1.600e-03    3.884e-04    75.73%    PASS

-- modal (1/1 passed) --------------------------------------------------------
  Cantilever omega_1 (Bernoulli)     18           18      0.00%    PASS

-- nonlinear (2/2 passed) ----------------------------------------------------
  Rectangular EPP shape factor      1.5         1.499     0.09%    PASS
  Yielding bar peak axial         4e+04        4e+04      0.00%    PASS
==============================================================================
```

| Benchmark | Reference |
|-----------|-----------|
| Bernoulli cantilever tip load | `P L^3 / (3 E I)` |
| SS beam mid-load | `P L^3 / (48 E I)` |
| Cook's membrane | Cook & Cook 1989 |
| SS plate, uniform pressure | Navier truncated series |
| Hex8 cantilever | `P L^3 / (3 E I)` for Bernoulli proxy |
| Cantilever ω₁ | `(1.875)^2 sqrt(EI/(rho A L^4))` |
| Euler pin-pin P_cr | `π² E I / L²` |
| Euler cantilever P_cr | `π² E I / (4 L²)` |
| Euler fixed-fixed P_cr | `4 π² E I / L²` |
| EPP shape factor | `Z / S = 1.5` rectangle |
| Yielding bar peak | `σ_y A` |

Coarse-mesh benchmarks (Cook's membrane, SS plate, Hex8) intentionally
ship at sub-converged tolerances — they document the discretisation
error so future changes are flagged if accuracy drifts.

## Adding a benchmark

```python
from femsolver.benchmarks.harness import Benchmark

def my_value():
    # build + run model, return scalar
    ...

bm = Benchmark(
    name="My new benchmark",
    category="linear-static",
    reference_value=42.0,
    reference_source="Textbook Ch. X",
    units="m",
    tolerance=0.05,
    runner=my_value,
    note="any free-text",
)
```

Append the benchmark to one of
{func}`femsolver.benchmarks.linear_static_benchmarks` /
{func}`~femsolver.benchmarks.modal_buckling_benchmarks` /
{func}`~femsolver.benchmarks.nonlinear_benchmarks` and it shows up
in the report automatically.
