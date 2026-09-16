"""Slab-modeling S0 — the surface (shell/plate) data model.

Headless coverage that ``project.Area`` and ``project.ShellSection`` carry the
data contract, that the ``Project`` accessors work, and that both survive a
JSON round-trip — including tuple coercion of ``Area.mesh`` and the
backward-compatible load of pre-slab projects (no ``areas`` key).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "desktop"))

from project import (  # noqa: E402
    SHELL_KINDS,
    Area,
    Node,
    Project,
    ShellSection,
)


def _slab_project() -> Project:
    """A minimal 3-D project with one 4-node quad slab area (4 m × 3 m)."""
    p = Project(ndm=3, ndf=6)
    p.shell_sections.append(
        ShellSection(id=1, name="SLAB200", thickness=0.20, kind="shell-thin",
                     modifiers={"m11": 0.25, "m22": 0.25})
    )
    p.nodes.extend([
        Node(id=1, x=0.0, y=0.0, z=0.0),
        Node(id=2, x=4.0, y=0.0, z=0.0),
        Node(id=3, x=4.0, y=3.0, z=0.0),
        Node(id=4, x=0.0, y=3.0, z=0.0),
    ])
    p.areas.append(
        Area(id=1, nodes=[1, 2, 3, 4], shell_section=1, material=0,
             mesh=(2, 2), local_axis=0.0)
    )
    return p


# ------------------------------------------------------------ ShellSection

def test_shell_section_kind_and_modifier_defaults():
    ss = ShellSection(id=1, name="T150", thickness=0.15)
    assert ss.kind == "shell-thin"
    assert ss.kind in SHELL_KINDS
    # an unset modifier reads back as 1.0; a set one reads its value
    assert ss.modifier("f11") == 1.0
    ss.modifiers["f11"] = 0.5
    assert ss.modifier("f11") == 0.5
    # a garbage modifier value degrades to 1.0 rather than raising
    ss.modifiers["m11"] = None
    assert ss.modifier("m11") == 1.0


# ------------------------------------------------------------------- Area

def test_area_post_init_coerces_nodes_and_mesh():
    a = Area(id=1, nodes=("1", 2, 3.0, 4), shell_section=1, material=0,
             mesh=[3, 4])
    assert a.nodes == [1, 2, 3, 4]
    assert isinstance(a.nodes[0], int)
    assert a.mesh == (3, 4)
    assert isinstance(a.mesh, tuple)


def test_area_mesh_defaults_to_1x1():
    a = Area(id=1, nodes=[1, 2, 3], shell_section=1, material=0)
    assert a.mesh == (1, 1)


def test_area_triangle_allowed():
    a = Area(id=7, nodes=[10, 11, 12], shell_section=1, material=2)
    assert len(a.nodes) == 3


# ------------------------------------------------------- Project accessors

def test_project_area_and_shell_section_accessors():
    p = _slab_project()
    assert p.shell_section(1).name == "SLAB200"
    assert p.shell_section(999) is None
    assert p.area(1).nodes == [1, 2, 3, 4]
    assert p.area(42) is None


def test_project_next_id_allocation():
    p = _slab_project()
    assert p.next_shell_section_id() == 2
    assert p.next_area_id() == 2
    empty = Project(ndm=3, ndf=6)
    assert empty.next_shell_section_id() == 1
    assert empty.next_area_id() == 1


# --------------------------------------------------------- JSON round-trip

def test_round_trip_preserves_areas_and_shell_sections():
    p = _slab_project()
    q = Project.from_json(p.to_json())

    assert len(q.shell_sections) == 1
    ss = q.shell_section(1)
    assert ss.name == "SLAB200"
    assert ss.thickness == 0.20
    assert ss.kind == "shell-thin"
    assert ss.modifiers == {"m11": 0.25, "m22": 0.25}

    assert len(q.areas) == 1
    a = q.area(1)
    assert a.nodes == [1, 2, 3, 4]
    assert a.shell_section == 1
    # mesh survives JSON (list) as a 2-tuple
    assert a.mesh == (2, 2)
    assert isinstance(a.mesh, tuple)


def test_backward_compatible_load_without_slab_keys():
    """A pre-slab project dict (no areas / shell_sections keys) loads with
    empty lists — old .fem files keep opening."""
    legacy = {
        "schema": "femsolver-project/1",
        "name": "legacy",
        "ndm": 2,
        "ndf": 3,
        "nodes": [{"id": 1, "x": 0.0, "y": 0.0}],
        "members": [],
    }
    p = Project.from_dict(legacy)
    assert p.areas == []
    assert p.shell_sections == []
    assert p.name == "legacy"
