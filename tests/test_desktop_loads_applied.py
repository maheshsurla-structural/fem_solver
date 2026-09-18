"""Loads Applied spec (E3b) — ``Project.apply_loads(model, ("applied", rows))``
and the ``normalize_loads_applied`` coercion helper.

A *Loads Applied* spec is an explicit list of ``(pattern_id, scale)`` rows (the
SAP2000 grid): it lets an analysis case pin exactly which load patterns it
applies and at what scale — e.g. ``1.0 Dead + 0.5 Live`` — without inventing a
combination. Coverage here is engine-level (no Qt): the scaled sum lands on the
model, zero-scale rows drop out, and the normalizer is robust to JSON
round-trips and junk rows.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "desktop"))

from project import (LoadCase, Load, Material, Member,  # noqa: E402
                     Node, Project, Section, normalize_loads_applied)


def _project():
    """A 2-D cantilever: node 2 free, carrying a Dead and a Live nodal load."""
    p = Project(ndm=2, ndf=3)
    p.nodes = [Node(1, 0, 0, supports=(1, 1, 1)), Node(2, 0, 3)]
    p.sections = [Section(id=1, name="s", A=1e-3, Iz=1e-5)]
    p.materials = [Material(1, "m", E=2e11, nu=0.3)]
    p.members = [Member(1, 1, 2, 1, 1)]
    p.load_cases = [LoadCase(id=1, name="Dead", nature="dead"),
                    LoadCase(id=2, name="Live", nature="live")]
    p.loads = [Load(node=2, values=(0.0, -10.0, 0.0), case=1),   # Dead
               Load(node=2, values=(0.0, -4.0, 0.0), case=2)]    # Live
    return p


def _load_on(p, selection):
    m = p.build_model(with_loads=False)
    p.apply_loads(m, selection)
    return list(m.node(2).load)


def test_applied_scales_and_sums_patterns():
    # 1.0 Dead + 0.5 Live = -10 + 0.5*(-4) = -12 on Fy
    assert _load_on(_project(), ("applied", [(1, 1.0), (2, 0.5)])) \
        == [0.0, -12.0, 0.0]


def test_applied_empty_clears_all_loads():
    assert _load_on(_project(), ("applied", [])) == [0.0, 0.0, 0.0]


def test_applied_skips_zero_scale_rows():
    with_zero = _load_on(_project(), ("applied", [(1, 1.0), (2, 0.0)]))
    without = _load_on(_project(), ("applied", [(1, 1.0)]))
    assert with_zero == without == [0.0, -10.0, 0.0]


def test_applied_all_patterns_x1_matches_all_selection():
    # applying every pattern at 1.0 is exactly the default ("all", None) build
    p = _project()
    assert _load_on(p, ("applied", [(1, 1.0), (2, 1.0)])) \
        == _load_on(p, ("all", None))


def test_applied_duplicate_pattern_accumulates():
    # apply_case is additive, so a repeated row stacks (matches SAP's grid)
    assert _load_on(_project(), ("applied", [(1, 1.0), (1, 0.5)])) \
        == [0.0, -15.0, 0.0]


def test_normalize_coerces_json_rows_and_drops_zeros():
    # JSON round-trips tuples to lists; a string scale coerces; a 0 drops out
    assert normalize_loads_applied([[1, "2.5"], [2, 0]]) == [(1, 2.5)]


def test_normalize_drops_missing_ids_and_none():
    assert normalize_loads_applied([[None, 1.0], [3, 1.5]]) == [(3, 1.5)]
    assert normalize_loads_applied(None) == []
    assert normalize_loads_applied([]) == []


def test_normalize_preserves_order_and_duplicates():
    assert normalize_loads_applied([[2, 1.0], [1, 0.5], [2, 0.25]]) \
        == [(2, 1.0), (1, 0.5), (2, 0.25)]
