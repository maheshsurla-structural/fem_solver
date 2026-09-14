"""Unit-aware input dialogs (plan U3) — the model must store SI regardless of
the display units the user types in, and seed the widgets back in display units.

Complements the existing dialog tests (which run under the default N/m, where
every conversion is the identity) by exercising a genuinely non-SI project.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "desktop"))

from project import Load, Member, MemberLoad, Node, Project, Section  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


def _project(force="kN", length="mm", ndm=2, ndf=3) -> Project:
    p = Project(ndm=ndm, ndf=ndf, force_unit=force, length_unit=length)
    p.nodes = [Node(id=1, x=0.0, y=0.0), Node(id=2, x=6.0, y=0.0)]
    p.sections = [Section(id=1, name="S", A=1.0, Iz=1.0)]
    p.materials = []
    p.members = [Member(id=1, n1=1, n2=2, section=1, material=None)]
    return p


# --------------------------------------------------------------- coordinates
def test_node_coords_parsed_to_si(qapp):
    from editing import NodeDialog
    p = _project(length="mm")
    dlg = NodeDialog(None, p)
    dlg.x.setValue(2000.0)          # 2000 mm typed
    dlg.y.setValue(-500.0)
    n = dlg.data()
    assert n.x == pytest.approx(2.0)     # stored as 2 m
    assert n.y == pytest.approx(-0.5)


def test_node_coords_seeded_in_display(qapp):
    from editing import NodeDialog
    p = _project(length="mm")
    dlg = NodeDialog(None, p, node=Node(id=1, x=3.0, y=0.0))  # 3 m stored
    assert dlg.x.value() == pytest.approx(3000.0)             # shown as 3000 mm
    assert dlg.x.unit_label() == "mm"


# ---------------------------------------------------------- forces & moments
def test_load_force_and_moment_convert_independently(qapp):
    """kN × mm: a force DOF scales by 1000, a moment DOF (kN·mm) by 1 — the two
    must not share one factor."""
    from editing import LoadDialog
    p = _project(force="kN", length="mm", ndm=2, ndf=3)   # DOFs: Dx, Dy, Rz
    dlg = LoadDialog(None, p)
    assert len(dlg.vals) == 3
    dlg.vals[0].setValue(5.0)     # 5 kN
    dlg.vals[2].setValue(7.0)     # 7 kN·mm (moment)
    ld = dlg.data()
    assert ld.values[0] == pytest.approx(5000.0)   # N
    assert ld.values[2] == pytest.approx(7.0)      # N·m  (7 kN·mm = 7 N·m)


def test_load_moment_label_is_force_length(qapp):
    from editing import LoadDialog
    p = _project(force="kN", length="m", ndm=2, ndf=3)
    dlg = LoadDialog(None, p)
    assert dlg.vals[0].unit_label() == "kN"        # Dx force
    assert dlg.vals[2].unit_label() == "kN·m"      # Rz moment


# ------------------------------------------------------------- line loads
def test_member_udl_parsed_to_si(qapp):
    from member_load_dialog import MemberLoadDialog
    p = _project(force="kN", length="m")           # kN/m = 1000 N/m
    dlg = MemberLoadDialog(None, p)
    dlg.wy.setValue(3.0)                            # 3 kN/m
    ml = dlg.data()
    assert ml.wy == pytest.approx(3000.0)          # N/m
    assert dlg.wy.unit_label() == "kN/m"


# ------------------------------------------------------------ load generator
def test_loadgen_magnitude_parsed_to_si(qapp):
    from editing import LoadGenDialog
    p = _project(force="kN", length="m")
    p.nodes = [Node(id=1, x=0.0, y=0.0, supports=(1, 1, 1)),
               Node(id=2, x=0.0, y=3.0)]
    dlg = LoadGenDialog(None, p)
    dlg.mag.setValue(50.0)                          # 50 kN
    _, mag = dlg.params()
    assert mag == pytest.approx(50000.0)            # N


# ---------------------------------------------------------- geometric moves
def test_move_offset_parsed_to_si(qapp):
    from editing import MoveDialog
    p = _project(length="mm")
    dlg = MoveDialog(None, p)
    dlg.dx.setValue(1500.0)                         # 1500 mm
    dx, dy, dz = dlg.data()
    assert dx == pytest.approx(1.5)                 # m


# ----------------------------------------------------------- imperial (U5)
def test_node_coords_imperial_ft(qapp):
    from editing import NodeDialog
    p = _project(force="kip", length="ft")
    dlg = NodeDialog(None, p)
    dlg.x.setValue(10.0)                            # 10 ft
    n = dlg.data()
    assert n.x == pytest.approx(3.048)              # m
    assert dlg.x.unit_label() == "ft"


def test_load_kip_and_kip_in_moment(qapp):
    from editing import LoadDialog
    p = _project(force="kip", length="in", ndm=2, ndf=3)
    dlg = LoadDialog(None, p)
    dlg.vals[0].setValue(2.0)                       # 2 kip
    dlg.vals[2].setValue(1.0)                       # 1 kip·in
    ld = dlg.data()
    assert ld.values[0] == pytest.approx(8896.443230521)     # N
    assert ld.values[2] == pytest.approx(4448.2216152605 * 0.0254)  # N·m
    assert dlg.vals[2].unit_label() == "kip·in"


# ------------------------------------------------ section properties (U6)
def test_section_area_inertia_convert(qapp):
    from editing import SectionDialog
    p = _project(length="mm")                       # AREA mm², INERTIA mm⁴
    dlg = SectionDialog(None, p, section=Section(id=1, name="S",
                                                 A=6.0e-3, Iz=2.0e-4))
    assert dlg.A.unit_label() == "mm²"
    assert dlg.A.value() == pytest.approx(6000.0)   # 6e-3 m² shown as mm²
    dlg.A.setValue(5000.0)                           # type 5000 mm²
    s = dlg.data()
    assert s.A == pytest.approx(5.0e-3)             # stored m²


def test_section_aisc_autofill_stays_si(qapp):
    from editing import SectionDialog, _designations
    p = _project(length="mm")
    dlg = SectionDialog(None, p)
    desigs = _designations()
    if desigs:                                       # pick a catalog shape
        i = [dlg.shape.itemData(k) for k in range(dlg.shape.count())].index(
            desigs[0])
        dlg.shape.setCurrentIndex(i)
        s = dlg.data()
        assert 0.0 < s.A < 1.0                       # a real area in m² (SI)


# --------------------------------------- live Properties panel (screenshot UI)
def _spins(widget):
    from unit_widgets import UnitSpin
    return widget.findChildren(UnitSpin)


def test_properties_node_form_converts(qapp):
    from properties import PropertiesPanel
    from PySide6.QtWidgets import QPushButton
    p = _project(length="mm")
    applied = {}
    panel = PropertiesPanel(lambda k, key, item: applied.update(item=item),
                            lambda *a: None)
    panel.show_item(p, "node", 1)                   # node 1 at x=0
    spins = _spins(panel._content)
    spins[0].setValue(2500.0)                       # x = 2500 mm
    panel._content.findChild(QPushButton).click()   # Apply
    assert applied["item"].x == pytest.approx(2.5)  # stored as 2.5 m


def test_properties_load_form_converts_moment(qapp):
    from properties import PropertiesPanel
    from PySide6.QtWidgets import QPushButton
    p = _project(force="kN", length="mm", ndm=2, ndf=3)
    p.loads = [Load(node=1, values=(0.0, 0.0, 0.0), case=1)]
    applied = {}
    panel = PropertiesPanel(lambda k, key, item: applied.update(item=item),
                            lambda *a: None)
    panel.show_item(p, "load", 0)
    spins = _spins(panel._content)                  # [Dx, Dy, Rz]
    spins[0].setValue(4.0)                           # 4 kN
    spins[2].setValue(9.0)                           # 9 kN·mm
    panel._content.findChild(QPushButton).click()
    assert applied["item"].values[0] == pytest.approx(4000.0)   # N
    assert applied["item"].values[2] == pytest.approx(9.0)      # N·m
