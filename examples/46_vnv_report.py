"""Phase 35.5 -- V&V benchmark report capstone.

Runs the full V&V benchmark suite and produces:

1. A formatted on-screen report (categorised, pass/fail flagged).
2. A CSV export (suitable for sales-grade verification documents,
   audit trails, regression-bot consumption).
3. Aggregate summary statistics (pass count, max relative error,
   mean relative error).

Run::

    python examples/46_vnv_report.py [--csv PATH]
"""
from __future__ import annotations

import argparse
import os

from femsolver.benchmarks import (
    all_benchmarks,
    export_csv,
    format_report,
    run_benchmarks,
    summary_stats,
)


def main(csv_path: str | None = None) -> None:
    print("=" * 96)
    print("femsolver V&V Benchmark Suite")
    print("=" * 96)
    print()
    print("Running benchmarks ...")

    bms = all_benchmarks()
    print(f"  Loaded {len(bms)} benchmarks "
          + ", ".join(sorted({b.category for b in bms})))

    results = run_benchmarks(bms)

    # Aggregate summary
    stats = summary_stats(results)
    print()
    print("Summary")
    print("-" * 96)
    print(f"  Total benchmarks      : {stats['total']}")
    print(f"  Passed                : {stats['passed']}")
    print(f"  Failed                : {stats['failed']}")
    print(f"  Mean relative error   : {stats['mean_error'] * 100:.2f} %")
    print(f"  Max  relative error   : {stats['max_error'] * 100:.2f} %")
    print()

    # Detailed report
    print(format_report(results))

    # CSV export (optional)
    if csv_path:
        export_csv(results, csv_path)
        print(f"CSV written to: {os.path.abspath(csv_path)}")

    # Exit status
    if stats["failed"] > 0:
        print("STATUS: at least one benchmark FAILED its declared tolerance.")
    else:
        print("STATUS: all benchmarks within tolerance.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=None,
                        help="optional CSV output path")
    args = parser.parse_args()
    main(csv_path=args.csv)
