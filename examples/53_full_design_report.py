"""Phase 33.4 -- end-to-end design report on a frame.

Capstone example for Phase 33. A 2-story 2-bay reinforced-concrete
frame is analysed under a single factored combination, every member
is designed via :class:`RcMemberDesigner` (Phase 29), and the full
results are packaged into two deliverables:

* an **HTML report** -- self-contained, one card per member with
  geometry, demand, capacity, DCR, code citations, pass/fail flag
* a **CSV summary** -- one row per member, suitable for spreadsheet
  review

Both outputs are written to a temporary directory (printed to stdout
so you can open / inspect them after the run).

Run::

    python examples/53_full_design_report.py
"""
from __future__ import annotations

import os
import tempfile

import numpy as np

from femsolver import (
    BeamColumn2D,
    ElasticIsotropic,
    Model,
)
from femsolver.analysis.linear_static import LinearStaticAnalysis
from femsolver.design import (
    from_beam_design_result,
    from_column_design_result,
    write_csv_summary,
    write_html_report,
)
from femsolver.design.concrete import (
    BeamDesignDemand,
    ColumnDesignDemand,
    ConcreteMaterial,
    RcMemberDesigner,
)


N_STORY = 2
N_BAY = 2
H_STORY = 3.5
L_BAY = 6.0
COL_B = 0.40; COL_H = 0.40
BEAM_B = 0.30; BEAM_H = 0.55
FC_PRIME = 28e6; FY = 420e6
E_CONC = 4700.0 * (FC_PRIME / 1e6) ** 0.5 * 1e6
W_BEAM_DEAD = 30e3
F_LATERAL_PER_FLOOR = 60e3


def build_frame():
    mat = ElasticIsotropic(1, E=E_CONC, nu=0.20, rho=2400.0)
    m = Model(ndm=2, ndf=3); m.add_material(mat)
    n_col = N_BAY + 1
    for j in range(N_STORY + 1):
        for i in range(n_col):
            m.add_node(j * n_col + i + 1, i * L_BAY, j * H_STORY)
    etag = 1
    col_tags: list[tuple[int, int, int]] = []
    for j in range(N_STORY):
        for i in range(n_col):
            n_b = j * n_col + i + 1
            n_t = (j + 1) * n_col + i + 1
            m.add_element(BeamColumn2D(
                etag, (n_b, n_t), mat, COL_B * COL_H,
                COL_B * COL_H ** 3 / 12.0,
            ))
            col_tags.append((etag, j + 1, i + 1))
            etag += 1
    beam_tags: list[tuple[int, int, int]] = []
    for j in range(1, N_STORY + 1):
        for i in range(N_BAY):
            n_L = j * n_col + i + 1
            n_R = j * n_col + i + 2
            b = BeamColumn2D(
                etag, (n_L, n_R), mat,
                BEAM_B * BEAM_H, BEAM_B * BEAM_H ** 3 / 12.0,
            )
            b.add_uniform_load(-W_BEAM_DEAD)
            m.add_element(b)
            beam_tags.append((etag, j, i + 1))
            etag += 1
    for i in range(n_col):
        m.fix(i + 1, [1, 1, 1])
    for j in range(1, N_STORY + 1):
        F_each = F_LATERAL_PER_FLOOR / n_col
        for i in range(n_col):
            m.add_nodal_load(j * n_col + i + 1, [F_each, 0, 0])
    return m, col_tags, beam_tags


def member_envelope(element) -> dict:
    ef = element.end_forces_local
    sf = element.section_forces
    return {
        "M_max": max(abs(ef[2]), abs(ef[5]),
                     float(np.max(np.abs(sf[:, 1])))),
        "V_max": max(abs(ef[1]), abs(ef[4])),
        "P_compr": -ef[3],
    }


def main() -> None:
    print("Phase 33.4 -- Full Design Report (capstone)")
    print("=" * 60)

    # --- Analyse ---
    model, col_tags, beam_tags = build_frame()
    LinearStaticAnalysis(model).run()
    mat_design = ConcreteMaterial(fc_prime=FC_PRIME, fy=FY)

    # --- Design every member, build report entries ---
    entries = []
    for etag, level, bay in beam_tags:
        env = member_envelope(model.elements[etag])
        demand = BeamDesignDemand(
            M_u_positive=env["M_max"],
            M_u_negative=env["M_max"],
            V_u=env["V_max"],
        )
        res = RcMemberDesigner.design_beam(
            b=BEAM_B, h=BEAM_H, material=mat_design,
            demand=demand, cover=0.050,
        )
        entries.append(from_beam_design_result(
            f"Beam L{level}-B{bay}", res, demand=demand,
        ))
    for etag, story, col in col_tags:
        env = member_envelope(model.elements[etag])
        demand = ColumnDesignDemand(
            P_u=max(0.0, env["P_compr"]),
            M_u=env["M_max"],
            V_u=env["V_max"],
        )
        res = RcMemberDesigner.design_column(
            b=COL_B, h=COL_H, material=mat_design,
            demand=demand, cover=0.060,
        )
        entries.append(from_column_design_result(
            f"Col S{story}-C{col}", res, demand=demand,
        ))

    # --- Console summary table ---
    print(f"  {'Member':<14} | {'Type':<25} | {'DCR':>6} | {'Status':<5}")
    print("  " + "-" * 60)
    for e in entries:
        status = "PASS" if e.passes else "FAIL"
        print(f"  {e.member_tag:<14} | {e.member_type:<25} | "
              f"{e.governing_dcr:>5.3f}  | {status:<5}")
    n_pass = sum(1 for e in entries if e.passes)
    print(f"\n  Summary: {n_pass}/{len(entries)} passing")
    print()

    # --- Write deliverables ---
    out_dir = tempfile.mkdtemp(prefix="femsolver_design_")
    html_path = os.path.join(out_dir, "design_report.html")
    csv_path = os.path.join(out_dir, "design_summary.csv")
    write_html_report(entries, html_path, title="2-Story RC Frame Design")
    write_csv_summary(entries, csv_path)
    print("Deliverables written to:")
    print(f"  HTML: {html_path}")
    print(f"  CSV:  {csv_path}")
    html_size = os.path.getsize(html_path)
    csv_size = os.path.getsize(csv_path)
    print(f"  Sizes: HTML {html_size} bytes, CSV {csv_size} bytes")
    print()
    print("Reading the result:")
    print("* Each member is designed by the appropriate driver, then")
    print("  packaged into a code-neutral MemberReportEntry.")
    print("* The HTML report is self-contained (CSS inline) -- attach")
    print("  directly to a project deliverable email.")
    print("* The CSV is the engineering-review version: import into")
    print("  Excel / Google Sheets for whole-frame triage.")
    print("* Each entry carries the code-clause references that the")
    print("  underlying design check used, so the report is auditable.")


if __name__ == "__main__":
    main()
