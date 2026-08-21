"""Example 78 -- Section Designer verification suite (Midas-GSD-comparable).

Runs the standardized Section-Designer verification set and produces the
artefacts needed to check femsolver against a parallel **Midas GSD**
(General Section Designer) model before a release:

1. A console table of every computed quantity (section properties,
   biaxial P-M-M landmarks in AASHTO LRFD 2024 + Eurocode 2, and
   moment-curvature milestones).
2. ``section_designer_verification.csv`` -- the same table with a blank
   ``midas_gsd`` column to fill in from the parallel model. Re-run with
   ``--gsd filled.csv`` to get the percentage differences and a
   pass/fail against each quantity's documented tolerance.
3. ``section_designer_verification.md`` -- a Markdown version for the
   spec sheet / PR.
4. Per-section geometry SVGs (rebar + dimensions) and, if matplotlib is
   installed, P-M interaction and M-phi plots for visual comparison.

The section definitions (exact dimensions, materials, rebar, tendons,
sign conventions, units) live in
``docs/source/section_designer_verification.md`` and in the docstrings
of :mod:`femsolver.benchmarks.section_designer`.

Run::

    python examples/78_section_designer_verification.py
    python examples/78_section_designer_verification.py --gsd my_gsd.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from femsolver.benchmarks.section_designer import (
    ALL_SECTION_BUILDERS,
    all_verification_items,
    export_csv,
    format_console,
    load_gsd_csv,
    to_markdown,
)
from femsolver.sections import section_to_svg


def _write_svgs(outdir: Path) -> list[str]:
    """One geometry SVG (with rebar + dimensions) per section."""
    written = []
    for build in ALL_SECTION_BUILDERS:
        case = build()
        svg = section_to_svg(case.section, show_rebar=True,
                             show_dimensions=True)
        path = outdir / f"{case.section_id}_geometry.svg"
        path.write_text(svg, encoding="utf-8")
        written.append(path.name)
    return written


def _write_plots(outdir: Path) -> list[str]:
    """P-M interaction plots (columns/pier) and M-phi plots (beam/PSC).

    Returns the list of files written, or an empty list if matplotlib
    is not available.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    import numpy as np

    from femsolver.benchmarks.section_designer import E_S
    from femsolver.design.concrete import (
        biaxial_pmm_surface_aashto,
        biaxial_pmm_surface_ec2,
        moment_curvature,
    )
    from femsolver.materials.uniaxial import ConcreteKentPark, UniaxialBilinear

    written: list[str] = []
    for build in ALL_SECTION_BUILDERS:
        case = build()

        if case.kind in ("column", "pier"):
            sa = biaxial_pmm_surface_aashto(
                case.section, f_c_prime=case.f_c_prime, f_y=case.f_y,
                spiral=case.spiral, n_angles=4, n_depths=40)
            se = biaxial_pmm_surface_ec2(
                case.section, f_ck=case.f_c_prime, f_yk=case.f_y,
                n_angles=4, n_depths=40)

            def _curve(surf, design):
                pts = sorted(surf.slice_uniaxial_z(), key=lambda p: p.P_n)
                M = [(p.phi_M_nz if design else p.M_nz) / 1e3 for p in pts]
                P = [(p.phi_P_n if design else p.P_n) / 1e3 for p in pts]
                return M, P

            fig, ax = plt.subplots(figsize=(6, 6))
            M, P = _curve(sa, False)
            ax.plot(M, P, "-", color="C0", label="AASHTO nominal")
            M, P = _curve(sa, True)
            ax.plot(M, P, "--", color="C0", label="AASHTO design (phi)")
            M, P = _curve(se, False)
            ax.plot(M, P, "-", color="C3", label="EC2 design")
            ax.axhline(0, color="k", lw=0.5)
            ax.set_xlabel("M_z  [kN.m]")
            ax.set_ylabel("P  [kN, +compression]")
            ax.set_title(f"{case.section_id} strong-axis P-M interaction")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            path = outdir / f"{case.section_id}_PM_interaction.png"
            fig.tight_layout()
            fig.savefig(path, dpi=110)
            plt.close(fig)
            written.append(path.name)

        if case.kind in ("beam", "psc"):
            import math
            concrete_uni = ConcreteKentPark(
                fpc=case.f_c_prime, eps_c0=0.002,
                fpcu=0.4 * case.f_c_prime, eps_cu=0.0035)
            steel_uni = UniaxialBilinear(E=E_S, sigma_y=case.f_y, b=0.01)
            f_r = 0.62 * math.sqrt(case.f_c_prime / 1e6) * 1e6
            res = moment_curvature(
                case.section, P_target=0.0,
                concrete_uniaxial=concrete_uni, steel_uniaxial=steel_uni,
                kappa_max=0.06, n_steps=60, f_y=case.f_y, E_s=E_S,
                f_rupture=f_r)
            kappa = [p.kappa for p in res.points]
            M = [p.M / 1e3 for p in res.points]
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(kappa, M, "-", color="C2")
            if res.M_cr:
                ax.axhline(res.M_cr / 1e3, color="C1", lw=0.8, ls=":",
                           label=f"M_cr={res.M_cr/1e3:.0f}")
            if res.M_u:
                ax.axhline(res.M_u / 1e3, color="C3", lw=0.8, ls=":",
                           label=f"M_u={res.M_u/1e3:.0f}")
            ax.set_xlabel("curvature kappa  [1/m]")
            ax.set_ylabel("M  [kN.m]")
            ax.set_title(f"{case.section_id} moment-curvature (P=0)")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            path = outdir / f"{case.section_id}_moment_curvature.png"
            fig.tight_layout()
            fig.savefig(path, dpi=110)
            plt.close(fig)
            written.append(path.name)
    return written


def main(outdir: str | None = None, gsd_csv: str | None = None) -> int:
    out = Path(outdir) if outdir else (
        Path(__file__).parent / "section_designer_verification")
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(" Section Designer verification suite (AASHTO LRFD 2024 + EC2)")
    print("=" * 78)

    items = all_verification_items()
    if gsd_csv and Path(gsd_csv).exists():
        load_gsd_csv(items, gsd_csv)
        n_ref = sum(1 for it in items if it.gsd is not None)
        n_pass = sum(1 for it in items if it.passed)
        print(f"\nLoaded {n_ref} Midas GSD values from {gsd_csv}: "
              f"{n_pass}/{n_ref} within tolerance.")

    print(format_console(items))

    csv_path = out / "section_designer_verification.csv"
    md_path = out / "section_designer_verification.md"
    export_csv(items, str(csv_path))
    md_path.write_text(
        "# Section Designer verification -- femsolver vs Midas GSD\n"
        + to_markdown(items), encoding="utf-8")

    svgs = _write_svgs(out)
    plots = _write_plots(out)

    print("-" * 78)
    print(f"Wrote {len(items)} verification items.")
    print(f"  CSV (fill the midas_gsd column) : {csv_path}")
    print(f"  Markdown table                  : {md_path}")
    print(f"  Geometry SVGs                   : {len(svgs)} files")
    if plots:
        print(f"  Interaction / M-phi plots       : {len(plots)} PNGs")
    else:
        print("  (matplotlib not installed -- skipped PNG plots)")
    print(f"  Output directory                : {out}")
    print("\nNext: build the same sections in Midas GSD, fill the "
          "midas_gsd column of the CSV, then re-run with --gsd <file>.")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=None,
                    help="output directory (default: examples/"
                         "section_designer_verification/)")
    ap.add_argument("--gsd", default=None,
                    help="a filled CSV to load Midas GSD values from")
    args = ap.parse_args()
    sys.exit(main(outdir=args.outdir, gsd_csv=args.gsd))
