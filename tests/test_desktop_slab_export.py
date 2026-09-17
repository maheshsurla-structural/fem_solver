"""Slab-modeling — CSV export of per-node slab results (slab plan S9)."""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "desktop"))

from femsolver.analysis.linear_static import LinearStaticAnalysis  # noqa: E402

import model_geometry as mg  # noqa: E402
from project import (Area, AreaLoad, Material, Node, Project,  # noqa: E402
                     ShellSection)


def _solved_slab(mesh=(3, 3)):
    p = Project(ndm=3, ndf=6)
    p.materials.append(Material(id=1, name="C", E=30e9, nu=0.2, rho=2400.0))
    p.shell_sections.append(ShellSection(id=1, name="T", thickness=0.2))
    p.nodes.extend([
        Node(id=1, x=0.0, y=0.0, z=0.0), Node(id=2, x=4.0, y=0.0, z=0.0),
        Node(id=3, x=4.0, y=4.0, z=0.0), Node(id=4, x=0.0, y=4.0, z=0.0),
    ])
    p.areas.append(Area(id=1, nodes=[1, 2, 3, 4], shell_section=1, material=1,
                        mesh=mesh))
    p.area_loads.append(AreaLoad(area=1, kind="selfweight", case=1))
    m = p.build_model(with_loads=True)
    for tag, nd in m.nodes.items():
        x, y, _z = nd.coords
        if x in (0.0, 4.0) or y in (0.0, 4.0):
            m.fix(tag, [1, 1, 1, 1, 1, 1])
    LinearStaticAnalysis(m).run()
    return m


def test_results_table_shape_and_headers():
    m = _solved_slab(mesh=(3, 3))
    headers, rows = mg.slab_results_table(m)
    assert headers[:6] == ["node", "x", "y", "z", "Uz", "|U|"]
    assert {"M11", "M22", "M12", "Vmax"} <= set(headers)
    assert len(rows) == 16                      # (3+1)² grid nodes
    assert len(rows[0]) == len(headers)


def test_results_table_none_without_areas():
    p = Project(ndm=3, ndf=6)
    p.nodes.append(Node(id=1, x=0.0, y=0.0, z=0.0))
    assert mg.slab_results_table(p.build_model(with_loads=False)) is None


def test_write_slab_csv_roundtrip(tmp_path):
    from main_window import _write_slab_csv
    m = _solved_slab(mesh=(2, 2))
    table = mg.slab_results_table(m)
    out = tmp_path / "slab.csv"
    _write_slab_csv(str(out), table)
    with open(out, newline="", encoding="utf-8") as f:
        got = list(csv.reader(f))
    assert got[0] == table[0]                   # header row
    assert len(got) == 1 + len(table[1])        # header + data rows
    # a centre node should have a non-zero downward deflection recorded
    uz_col = table[0].index("Uz")
    assert any(float(r[uz_col]) < 0.0 for r in got[1:])
