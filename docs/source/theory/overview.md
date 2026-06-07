# Theory manual overview

The full LaTeX theory document lives at `docs/theory/theory.tex`. It
covers eight chapters in roughly 30-40 pages:

1. **Introduction** — architecture, package layout, conventions.
2. **Linear analysis** — equilibrium, element stiffness, mass,
   assembly, MP-constraint transformation handler.
3. **Nonlinear static analysis** — Newton-Raphson, integrators
   (load / displacement / arc-length), convergence tests.
4. **Constitutive models** — J₂ plasticity (radial return),
   Drucker-Prager, the full uniaxial catalogue.
5. **Sections and elements** — Bernoulli vs corotational vs
   force-based beams; MITC4/9/DKMQ4/Tri3/DKT3 shells; fiber sections;
   plane-section kinematics.
6. **Dynamics** — Newmark family, HHT-α / Gen-α dissipation; modal
   analysis; response-spectrum SRSS/CQC; multi-support excitation.
7. **PBE pipeline** — IDA, collapse, lognormal fragility,
   Conditional Mean Spectrum, FEMA P-58.
8. **Design and detailing** — ACI 318 / AISC 360 / IS 456 / IS 800
   member checks; ACI 18 / AISC 341 / IS 13920 seismic detailing;
   panel zone + RBS + PR connections.
9. **Bridges** — influence lines, PT tendon losses, CEB-FIP creep /
   shrinkage / relaxation.
10. **V&V** — closed-form benchmarks, NAFEMS-style tests.

## Building the PDF

```bash
cd docs/theory
pdflatex theory.tex
pdflatex theory.tex          # run twice to resolve TOC + references
```

The output is `docs/theory/theory.pdf` (~40 pages).

## Citing

If you publish work that uses femsolver, the canonical citation form is:

```bibtex
@manual{femsolver,
  title  = {femsolver: a Python finite-element solver for
            structural analysis},
  author = {{femsolver contributors}},
  year   = {2026},
  note   = {Version 0.1.0. Theory manual: docs/theory/theory.tex.
            V\&V report: examples/46_vnv_report.py.},
}
```

## Source-code cross-references

Each chapter of the theory manual cross-references the implementing
module:

| Theory chapter | Implementation |
|----------------|----------------|
| 2. Linear analysis | `femsolver.analysis.linear_static`, `femsolver.analysis.assembler` |
| 3. Nonlinear static | `femsolver.analysis.nonlinear_static`, `.algorithm`, `.integrator`, `.convergence` |
| 4. Constitutive | `femsolver.materials.*`, `femsolver.materials.uniaxial.*` |
| 5. Sections + elements | `femsolver.sections.*`, `femsolver.elements.*` |
| 6. Dynamics | `femsolver.analysis.transient`, `.transient_integrator`, `.eigen`, `.response_spectrum` |
| 7. PBE | `femsolver.performance.ida`, `.ida_collapse`, `.fragility`, `.cms`, `.record_scaling`, `.p58` |
| 8. Design | `femsolver.design.*` |
| 9. Bridges | `femsolver.bridges.*` |
| 10. V&V | `femsolver.benchmarks.*` |
