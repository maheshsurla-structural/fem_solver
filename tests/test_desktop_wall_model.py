"""Wall-modeling W0 — the wall / pier data model.

Headless coverage that ``project.Area`` carries a structural ``role`` plus
optional ``pier`` / ``spandrel`` labels (ETABS-style), that the labels are
normalized and gated on the wall role, that the ``Project`` wall/pier accessors
group areas correctly, that the whole thing survives a JSON round-trip, and that
older files (no ``role`` key) migrate to a slab.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "desktop"))

from project import (  # noqa: E402
    AREA_ROLES,
    Area,
    Node,
    Project,
    ShellSection,
)


def _wall_project() -> Project:
    """A 3-D project with two stacked wall quads sharing pier label 'P1'."""
    p = Project(ndm=3, ndf=6)
    p.shell_sections.append(
        ShellSection(id=1, name="WALL250", thickness=0.25, kind="shell-thin")
    )
    # a vertical wall in the X-Z plane, two panels stacked in Z (two stories)
    p.nodes.extend([
        Node(id=1, x=0.0, y=0.0, z=0.0),
        Node(id=2, x=4.0, y=0.0, z=0.0),
        Node(id=3, x=4.0, y=0.0, z=3.0),
        Node(id=4, x=0.0, y=0.0, z=3.0),
        Node(id=5, x=4.0, y=0.0, z=6.0),
        Node(id=6, x=0.0, y=0.0, z=6.0),
    ])
    p.areas.append(Area(id=1, nodes=[1, 2, 3, 4], shell_section=1, material=0,
                        role="wall", pier="P1"))
    p.areas.append(Area(id=2, nodes=[4, 3, 5, 6], shell_section=1, material=0,
                        role="wall", pier="P1"))
    return p


# ------------------------------------------------------------- Area role / labels

def test_area_defaults_to_slab_with_no_labels():
    a = Area(id=1, nodes=[1, 2, 3, 4], shell_section=1, material=0)
    assert a.role == "slab"
    assert a.pier is None and a.spandrel is None


def test_unknown_role_falls_back_to_slab():
    a = Area(id=1, nodes=[1, 2, 3, 4], shell_section=1, material=0,
             role="bogus")
    assert a.role == "slab"
    assert set(AREA_ROLES) == {"slab", "wall", "shell"}


def test_wall_keeps_pier_and_spandrel_labels():
    a = Area(id=1, nodes=[1, 2, 3, 4], shell_section=1, material=0,
             role="wall", pier="P1", spandrel="S2")
    assert a.role == "wall"
    assert a.pier == "P1"
    assert a.spandrel == "S2"


def test_labels_are_trimmed_and_blanks_become_none():
    a = Area(id=1, nodes=[1, 2, 3, 4], shell_section=1, material=0,
             role="wall", pier="  P1  ", spandrel="   ")
    assert a.pier == "P1"          # whitespace stripped
    assert a.spandrel is None      # blank -> unset


def test_labels_dropped_when_role_is_not_wall():
    """A pier/spandrel label only means something on a wall — a slab carrying a
    stray label normalizes it away, so 'is this labeled' has one answer."""
    a = Area(id=1, nodes=[1, 2, 3, 4], shell_section=1, material=0,
             role="slab", pier="P1", spandrel="S2")
    assert a.pier is None and a.spandrel is None


# ------------------------------------------------------ Project wall/pier accessors

def test_project_walls_and_pier_accessors():
    p = _wall_project()
    assert {a.id for a in p.walls()} == {1, 2}
    assert p.pier_names() == ["P1"]
    assert p.spandrel_names() == []
    assert {a.id for a in p.areas_in_pier("P1")} == {1, 2}
    assert p.areas_in_pier("nope") == []


def test_walls_exclude_slabs_and_shells():
    p = _wall_project()
    p.nodes.append(Node(id=7, x=0.0, y=0.0, z=0.0))
    p.areas.append(Area(id=3, nodes=[1, 2, 3, 4], shell_section=1, material=0,
                        role="slab"))
    assert {a.id for a in p.walls()} == {1, 2}
    assert p.pier_names() == ["P1"]     # the slab contributes no pier


# ------------------------------------------------------------- JSON round-trip

def test_round_trip_preserves_role_and_labels():
    p = _wall_project()
    p.areas[1].spandrel = "S1"          # give panel 2 a spandrel too
    q = Project.from_json(p.to_json())

    a1, a2 = q.area(1), q.area(2)
    assert a1.role == "wall" and a1.pier == "P1"
    assert a2.role == "wall" and a2.pier == "P1" and a2.spandrel == "S1"
    assert q.pier_names() == ["P1"]
    assert q.spandrel_names() == ["S1"]


def test_legacy_area_without_role_migrates_to_slab():
    """An area dict from before W0 (no role/pier/spandrel keys) loads as a
    slab with no labels — old .fem files keep opening."""
    legacy = {
        "schema": "femsolver-project/1",
        "name": "legacy",
        "ndm": 3,
        "ndf": 6,
        "nodes": [{"id": i, "x": 0.0, "y": 0.0, "z": 0.0} for i in range(1, 5)],
        "shell_sections": [{"id": 1, "name": "T200", "thickness": 0.2}],
        "areas": [{"id": 1, "nodes": [1, 2, 3, 4], "shell_section": 1,
                   "material": 0, "mesh": [1, 1]}],
    }
    p = Project.from_dict(legacy)
    a = p.area(1)
    assert a.role == "slab"
    assert a.pier is None and a.spandrel is None
    assert p.walls() == []
