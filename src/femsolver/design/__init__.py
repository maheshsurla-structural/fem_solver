"""Design code modules.

Phase 29-33: code-based member design (ACI 318, AISC 360, ASCE 7).

Submodules
----------
* :mod:`concrete` -- ACI 318-19 reinforced concrete (Phase 29, 32)
* future: ``steel`` -- AISC 360-22 (Phase 30, 32)
* future: ``combos`` -- ASCE 7-22 load combinations + drift (Phase 31)
* future: ``reports`` -- HTML / CSV design reports (Phase 33)
"""
from femsolver.design import concrete, seismic, steel
from femsolver.design import is456, is800, is1893, is13920   # Phase 36 (G2)
from femsolver.design import connections                        # Phase 37 (I)
from femsolver.design import ec2, ec3, ec8                       # Phase 46 (G1)
from femsolver.design import punching, two_way_slab, diaphragm   # Phase 53 (W)
from femsolver.design import punching_reinforcement              # Phase HH.8
from femsolver.design import timber                                # Phase D.1.3
from femsolver.design import psc                                   # Phase B.8
from femsolver.design.reports import (
    MemberReportEntry,
    from_beam_design_result,
    from_column_design_result,
    from_steel_member_check,
    make_csv_summary,
    make_html_report,
    write_csv_summary,
    write_html_report,
)

__all__ = [
    "concrete",
    "seismic",
    "steel",
    "is456",
    "is800",
    "is1893",
    "is13920",
    "connections",
    "ec2",
    "ec3",
    "ec8",
    "punching",
    "punching_reinforcement",
    "two_way_slab",
    "diaphragm",
    "MemberReportEntry",
    "from_beam_design_result",
    "from_column_design_result",
    "from_steel_member_check",
    "make_html_report",
    "make_csv_summary",
    "write_html_report",
    "write_csv_summary",
]
