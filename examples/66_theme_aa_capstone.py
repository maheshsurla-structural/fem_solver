"""Theme AA capstone -- complete cross-platform validation suite.

Runs every cross-platform benchmark (NAFEMS + frames) and produces:

* A printable text table for CI logs.
* An HTML validation report (browser-friendly).
* A CSV export (auditable record for QA / management).

Each benchmark is compared against 2-3 documented references, so a
PASS means femsolver agrees with all the published reference values
within their stated tolerances.
"""
from __future__ import annotations

import os
import sys

from femsolver.benchmarks import (
    all_cross_platform_benchmarks,
    export_validation_csv,
    format_validation_table,
    render_validation_html,
)


def main():
    print("=" * 78)
    print("Theme AA capstone -- cross-platform validation suite")
    print("=" * 78)

    benchmarks = all_cross_platform_benchmarks()
    results = [b.run() for b in benchmarks]
    n_pass = sum(1 for r in results if r.overall_passed)

    print()
    print(format_validation_table(results, width=140))

    print(f"Summary: {n_pass}/{len(results)} benchmarks passed all "
          "reference checks.")
    print()
    print("Per-benchmark detail:")
    for r in results:
        b = r.benchmark
        cat = b.category
        print(f"  [{cat:14s}] {b.name}")
        print(f"      Computed: {r.computed_value:.6e} {b.units}")
        for ref in b.references:
            err, ok = r.per_reference.get(ref.source, (float('nan'), False))
            tag = "OK  " if ok else "FAIL"
            print(f"      [{tag}] {ref.source:<40s} {ref.value:.6e} "
                  f"(tol {ref.tolerance*100:.2f}%, err {err*100:.3f}%)")

    # Save HTML + CSV
    os.makedirs("theme_aa_validation", exist_ok=True)
    html_path = "theme_aa_validation/validation_report.html"
    csv_path = "theme_aa_validation/validation_data.csv"
    with open(html_path, "w", encoding="utf-8") as fh:
        fh.write(render_validation_html(
            results,
            title="femsolver cross-platform validation",
        ))
    export_validation_csv(results, csv_path)
    print()
    print(f"HTML report: {html_path}")
    print(f"CSV data:    {csv_path}")

    print()
    print("Theme AA capstone DONE.")


if __name__ == "__main__":
    sys.exit(main())
