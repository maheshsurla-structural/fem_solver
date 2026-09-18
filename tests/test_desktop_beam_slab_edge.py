"""Beam-to-slab-edge compatibility — prototype epics BE0-BE2.

A member lying along a meshed slab edge is auto-split at the slab's edge nodes so
beam and slab share those nodes (true displacement compatibility). Covers the
geometric splitter and the shared-node merge; results/design aggregation (BE3) is
separate and not exercised here.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "desktop"))

import pytest  # noqa: E402

from project import (  # noqa: E402
    Area, Material, Member, Node, Project, Section, ShellSection,
    decode_member_id, member_element_tag,
)


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


def _slab_with_edge_beam(mesh=(2, 2), auto=True):
    """A single 4×4 quad slab meshed ``mesh`` with a beam along its n1->n2 edge."""
    p = Project(ndm=3, ndf=6, auto_connect_beams=auto)
    p.materials.append(Material(id=1, name="C", E=30e9, nu=0.2))
    p.sections.append(Section(id=1, name="B", A=0.09, Iz=6.75e-4,
                              Iy=6.75e-4, J=1.0e-3))
    p.shell_sections.append(ShellSection(id=1, name="T", thickness=0.2))
    p.nodes.extend([
        Node(id=1, x=0.0, y=0.0, z=0.0),
        Node(id=2, x=4.0, y=0.0, z=0.0),
        Node(id=3, x=4.0, y=4.0, z=0.0),
        Node(id=4, x=0.0, y=4.0, z=0.0),
    ])
    p.areas.append(Area(id=1, nodes=[1, 2, 3, 4], shell_section=1, material=1,
                        mesh=mesh))
    p.members.append(Member(id=7, n1=1, n2=2, section=1, material=1))
    return p


def _bare_model(p):
    """A model holding only the project nodes — the state ``build_model`` is in
    when it builds the shared node registry, so the internal helpers can be
    exercised in isolation without re-creating already-meshed slab nodes."""
    from femsolver import Model
    m = Model(ndm=p.ndm, ndf=p.ndf)
    for nd in p.nodes:
        m.add_node(nd.id, nd.x, nd.y, nd.z)
    return m


# ------------------------------------------------------ BE1 detection (pure)

def test_split_chain_finds_edge_nodes():
    p = _slab_with_edge_beam(mesh=(2, 2))
    edge = p._register_area_edge_nodes(p._node_registry(_bare_model(p))[1])
    chain = p._member_split_chain(p.members[0], edge)
    # 2×2 mesh → one interior node on the n1->n2 edge → 3-node chain
    assert chain is not None and len(chain) == 3
    assert chain[0] == 1 and chain[-1] == 2


def test_split_chain_none_without_interior_nodes():
    p = _slab_with_edge_beam(mesh=(1, 1))          # edge has no interior node
    edge = p._register_area_edge_nodes(p._node_registry(_bare_model(p))[1])
    assert p._member_split_chain(p.members[0], edge) is None


# ------------------------------------------------ BE2 build-time splitting

def test_member_split_into_subelements():
    p = _slab_with_edge_beam(mesh=(2, 2))
    m = p.build_model(with_loads=False)
    tags = set(m.elements.keys())
    # the member (id 7) is replaced by two sub-elements with split tags…
    assert 7 not in tags
    assert member_element_tag(7, 0) in tags
    assert member_element_tag(7, 1) in tags


def test_beam_and_slab_share_the_edge_node():
    p = _slab_with_edge_beam(mesh=(2, 2))
    m = p.build_model(with_loads=False)
    # the beam sub-elements' shared node is the mid-edge node at (2, 0, 0)
    e0 = m.elements[member_element_tag(7, 0)]
    e1 = m.elements[member_element_tag(7, 1)]
    shared = set(e0.node_tags) & set(e1.node_tags)
    assert len(shared) == 1
    mid = shared.pop()
    assert tuple(round(c, 6) for c in m.node(mid).coords[:3]) == (2.0, 0.0, 0.0)
    # …and a shell element references that very node (one merged node, not two)
    shell_tags = [set(e.node_tags) for t, e in m.elements.items()
                  if t >= 2_000_000]
    assert any(mid in nt for nt in shell_tags)


def test_finer_mesh_splits_into_more_elements():
    p = _slab_with_edge_beam(mesh=(4, 2))          # 3 interior nodes on the edge
    m = p.build_model(with_loads=False)
    sub = [t for t in m.elements if member_element_tag(7, 0) <= t
           < member_element_tag(7, 0) + 1000]
    assert len(sub) == 4                           # 3 interior nodes → 4 spans


# ------------------------------------------------------ toggle + regression

def test_toggle_off_keeps_single_element():
    p = _slab_with_edge_beam(mesh=(2, 2), auto=False)
    m = p.build_model(with_loads=False)
    assert 7 in m.elements                         # unsplit, keyed by member id
    assert member_element_tag(7, 0) not in m.elements


def test_edge_beam_carries_slab_load():
    """Physics: a slab supported only by an edge beam (beam ends pinned) must
    solve without singularity and deflect downward. The interior slab edge node
    is unsupported except through the shared beam node — so a finite, downward
    solution proves beam↔slab compatibility is live, not just topological."""
    from femsolver import LinearStaticAnalysis
    p = _slab_with_edge_beam(mesh=(2, 2))
    # pin the beam's two end nodes (the slab's other edge is free)
    for nd in p.nodes:
        if nd.id in (1, 2):
            nd.supports = [1, 1, 1, 1, 1, 1]
    m = p.build_model(with_loads=False)
    # the mid-edge slab node, shared by the beam's two sub-elements
    mid = None
    for t, e in m.elements.items():
        if member_element_tag(7, 0) <= t < member_element_tag(7, 0) + 1000:
            mid = (set(e.node_tags) - {1, 2}).pop()
    assert mid is not None
    m.add_nodal_load(mid, [0.0, 0.0, -1.0e4, 0.0, 0.0, 0.0])
    LinearStaticAnalysis(m).run()
    uz = m.node(mid).disp[2]
    assert uz < 0.0 and abs(uz) < 1.0             # downward and finite


def test_pure_frame_unaffected():
    p = Project(ndm=3, ndf=6)
    p.materials.append(Material(id=1, name="C", E=30e9, nu=0.2))
    p.sections.append(Section(id=1, name="B", A=0.09, Iz=6.75e-4,
                              Iy=6.75e-4, J=1.0e-3))
    p.nodes.extend([Node(id=1, x=0.0, y=0.0, z=0.0),
                    Node(id=2, x=4.0, y=0.0, z=0.0)])
    p.members.append(Member(id=7, n1=1, n2=2, section=1, material=1))
    m = p.build_model(with_loads=False)
    assert set(m.elements.keys()) == {7}           # no areas → no splitting


# =================================================== BE3: member↔element map

def test_member_element_tags_map():
    p = _slab_with_edge_beam(mesh=(2, 2))
    tags = p.member_element_tags(p.members[0])
    assert tags == [member_element_tag(7, 0), member_element_tag(7, 1)]
    # the map agrees with what the build actually emitted
    m = p.build_model(with_loads=False)
    assert all(t in m.elements for t in tags)


def test_member_element_tags_unsplit():
    # no interior edge node on a 1×1 mesh → single element keyed by member id
    p = _slab_with_edge_beam(mesh=(1, 1))
    assert p.member_element_tags(p.members[0]) == [7]
    # toggle off → never split, even with interior edge nodes present
    q = _slab_with_edge_beam(mesh=(2, 2), auto=False)
    assert q.member_element_tags(q.members[0]) == [7]


def test_decode_member_id_ranges():
    assert decode_member_id(7) == 7                # unsplit member id == tag
    assert decode_member_id(member_element_tag(7, 3)) == 7   # split sub-element
    from project import area_element_tag
    assert decode_member_id(area_element_tag(1, 0)) is None  # area/shell tag


# =================================================== BE3: design over a split

def test_design_covers_split_beam():
    """A steel edge beam split at the slab edge must be design-checked over its
    whole length — every sub-element resolves to member 7 and gets a DCR."""
    from design import design_all
    from femsolver import LinearStaticAnalysis
    p = _slab_with_edge_beam(mesh=(2, 2))
    p.sections[0].shape = "W12x65"                 # steel §H1 path
    p.materials[0] = Material(id=1, name="A992", E=200e9, nu=0.3)
    for nd in p.nodes:                             # pin the two beam ends
        if nd.id in (1, 2):
            nd.supports = [1, 1, 1, 1, 1, 1]
    m = p.build_model(with_loads=False)
    mid = (set(m.elements[member_element_tag(7, 0)].node_tags)
           & set(m.elements[member_element_tag(7, 1)].node_tags)).pop()
    m.add_nodal_load(mid, [0.0, 0.0, -5.0e4, 0.0, 0.0, 0.0])
    LinearStaticAnalysis(m).run()
    dcrs = design_all(m, p)
    # both sub-elements of member 7 carry a (finite) DCR — not grey
    for j in (0, 1):
        d = dcrs[member_element_tag(7, j)]
        assert d is not None and d > 0.0


# =================================================== BE3: selection decode

# =================================================== BE5: hinged / cable

def test_hinged_member_not_split():
    """A member carrying a (lumped) plastic hinge keeps its identity — a lumped
    hinge assumes one element end-to-end, so it is never split at a slab edge."""
    p = _slab_with_edge_beam(mesh=(2, 2))
    p.members[0].hinge = 1                          # assign a hinge property id
    assert p.member_element_tags(p.members[0]) == [7]
    m = p.build_model(with_loads=False)
    assert 7 in m.elements                          # single element, id == tag
    assert member_element_tag(7, 0) not in m.elements
    # removing the hinge lets it split again (the gate is the hinge, not luck)
    p.members[0].hinge = None
    assert len(p.member_element_tags(p.members[0])) == 2


def test_cable_member_not_split():
    p = _slab_with_edge_beam(mesh=(2, 2))
    p.members[0].kind = "cable"
    assert p.member_element_tags(p.members[0]) == [7]
    m = p.build_model(with_loads=False)
    assert 7 in m.elements                          # single truss, not split


# =================================================== BE4: line-load spread

def test_member_udl_distributed_to_subelements():
    from project import MemberLoad
    p = _slab_with_edge_beam(mesh=(2, 2))
    p.member_loads.append(MemberLoad(member=7, wy=-1000.0, wz=-200.0, case=1))
    m = p.build_model(with_loads=False)
    p.apply_case(m, 1, 1.0)
    for j in (0, 1):                               # every sub-element gets the
        el = m.element(member_element_tag(7, j))   # same intensity (N/m)
        assert abs(el._wy_local - (-1000.0)) < 1e-9
        assert abs(el._wz_local - (-200.0)) < 1e-9


def test_member_udl_total_equivalent_load_preserved():
    """The split must not change the physics: the total equivalent transverse
    load over the sub-elements equals the whole member's UDL (wy·L)."""
    from project import MemberLoad
    wy = -1000.0
    p = _slab_with_edge_beam(mesh=(4, 2))          # beam split into 4
    p.member_loads.append(MemberLoad(member=7, wy=wy, case=1))
    m = p.build_model(with_loads=False)
    p.apply_case(m, 1, 1.0)
    total = 0.0
    for t in m.elements:
        if member_element_tag(7, 0) <= t < member_element_tag(7, 0) + 1000:
            f = m.element(t).f_eq_local()          # 3-D: f[1]+f[7] = wy·L_sub
            total += f[1] + f[7]
    assert abs(total - wy * 4.0) < 1e-6            # L_total = 4.0 m


def test_member_udl_factor_scales():
    from project import MemberLoad
    p = _slab_with_edge_beam(mesh=(2, 2))
    p.member_loads.append(MemberLoad(member=7, wy=-1000.0, case=1))
    m = p.build_model(with_loads=False)
    p.apply_case(m, 1, 1.4)                         # combination factor
    assert abs(m.element(member_element_tag(7, 0))._wy_local - (-1400.0)) < 1e-9


def test_member_udl_unsplit_still_applies():
    from project import MemberLoad
    p = _slab_with_edge_beam(mesh=(1, 1))          # not split
    p.member_loads.append(MemberLoad(member=7, wy=-500.0, case=1))
    m = p.build_model(with_loads=False)
    p.apply_case(m, 1, 1.0)
    assert abs(m.element(7)._wy_local - (-500.0)) < 1e-9


def test_window_select_reports_member_not_subtag(qapp):
    p = _slab_with_edge_beam(mesh=(2, 2))
    m = p.build_model(with_loads=False)
    # every node of the beam edge (y == 0) is "inside" the window
    inside = {t: (abs(m.node(t).coords[1]) < 1e-9) for t in m.nodes}
    from model_view import ModelView
    v = ModelView()
    v.set_model(m)
    refs = v._members_inside(inside)
    assert ("member", 7) in refs                   # project id, not a 4M+ tag
    assert all(mid < 1_000_000 for _k, mid in refs)
