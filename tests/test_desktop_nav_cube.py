"""Navigation cube (orientation gizmo) — ``desktop/nav_cube.py`` +
``ModelView`` wiring.

Covers the font-independent contract headlessly: the camera basis, which faces
are visible, click hit-testing, and — the key regression — that each cube face
is paired with the ``set_view`` name that actually points the camera *at that
face*, so "click TOP → the top view" stays self-consistent.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "desktop"))


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    pytest.importorskip("pyvistaqt")
    from PySide6.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


# --- pure geometry (no Qt widget) -------------------------------------------
def test_camera_basis_top_view():
    import nav_cube as nc
    # camera above the origin looking straight down, +Y up → right is +X,
    # forward is -Z (into the scene).
    r, u, f = nc.camera_basis_from((0, 0, 10), (0, 0, 0), (0, 1, 0))
    assert np.allclose(r, (1, 0, 0), atol=1e-9)
    assert np.allclose(u, (0, 1, 0), atol=1e-9)
    assert np.allclose(f, (0, 0, -1), atol=1e-9)


def test_camera_basis_degenerate_returns_none():
    import nav_cube as nc
    assert nc.camera_basis_from((0, 0, 0), (0, 0, 0), (0, 0, 1)) is None


def test_visible_faces_are_three_and_non_opposite():
    import nav_cube as nc
    b = nc.camera_basis_from((1, -1, 0.8), (0, 0, 0), (0, 0, 1))
    vis = nc.visible_faces(b)
    assert len(vis) == 3                       # an iso pose shows exactly three
    opposite = {"left": "right", "right": "left", "front": "back",
                "back": "front", "top": "bottom", "bottom": "top"}
    for name in vis:                           # never a face and its opposite
        assert opposite[name] not in vis


def test_top_view_shows_only_the_top_face():
    import nav_cube as nc
    b = nc.camera_basis_from((0, 0, 10), (0, 0, 0), (0, 1, 0))
    assert nc.visible_faces(b) == ["top"]


def test_hit_test_face_centroids_and_corner(qapp):
    import nav_cube as nc
    b = nc.camera_basis_from((1, -1, 0.8), (0, 0, 0), (0, 0, 1))
    center, radius = (66.0, 66.0), 132 * 0.28
    faces = {n: c for n, _nm, c in nc._FACES}
    for name in nc.visible_faces(b):
        cs = faces[name]
        cen = [sum(c[i] for c in cs) / 4.0 for i in range(3)]
        sx, sy, _d = nc.project(cen, b)
        px, py = center[0] + sx * radius, center[1] - sy * radius
        assert nc.hit_test(px, py, b, center, radius) == name
    # a near corner (all components toward the camera) → isometric
    _r, _u, f = b
    near = min(nc._CORNERS, key=lambda c: np.asarray(c, float) @ f)
    sx, sy, _d = nc.project(near, b)
    px, py = center[0] + sx * radius, center[1] - sy * radius
    assert nc.hit_test(px, py, b, center, radius) == "iso"
    # empty space well outside the cube → nothing
    assert nc.hit_test(2.0, 2.0, b, center, radius) is None


# --- ModelView wiring -------------------------------------------------------
def test_view_face_pairing_is_self_consistent(qapp):
    """The heart of it: for every named ortho view, the cube face turned most
    head-on to the camera is the face bearing that view's name."""
    import model_view as mv
    import nav_cube as nc
    from demo_model import demo_project
    v = mv.ModelView()
    v.set_model(demo_project().build_model())
    for name in ("top", "bottom", "front", "back", "left", "right"):
        v.set_view(name)
        _r, _u, f = v.camera_basis()
        head_on = min(nc._FACES, key=lambda fc: np.asarray(fc[1]) @ f)
        assert head_on[0] == name


def test_modelview_has_positioned_nav_cube(qapp):
    import model_view as mv
    v = mv.ModelView()
    v.resize(900, 600)
    v._position_nav_cube()
    cube = v._nav_cube
    # pinned to the top-right, inside the viewport
    assert cube.x() + cube.width() <= v.width()
    assert cube.x() > v.width() / 2
    assert cube.y() < v.height() / 2


def test_orbit_changes_the_camera_without_error(qapp):
    import model_view as mv
    from demo_model import demo_project
    v = mv.ModelView()
    v.set_model(demo_project().build_model())
    v.set_view("front")
    before = v.camera_basis()
    v.orbit(90.0, 0.0)
    after = v.camera_basis()
    assert not np.allclose(np.vstack(before), np.vstack(after), atol=1e-3)


# --- widget interaction -----------------------------------------------------
class _FakeView:
    """Minimal stand-in recording the cube's callbacks (a fixed iso basis)."""
    def __init__(self):
        import nav_cube as nc
        self._b = nc.camera_basis_from((1, -1, 0.8), (0, 0, 0), (0, 0, 1))
        self.views, self.orbits = [], []

    def camera_basis(self):
        return self._b

    def set_view(self, name):
        self.views.append(name)

    def orbit(self, az, el):
        self.orbits.append((az, el))


def _press(cube, x, y):
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent
    ev = QMouseEvent(QEvent.Type.MouseButtonPress, QPointF(x, y),
                     Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
                     Qt.KeyboardModifier.NoModifier)
    cube.mousePressEvent(ev)


def test_clicking_a_face_calls_set_view(qapp):
    import nav_cube as nc
    from PySide6.QtWidgets import QWidget
    host = QWidget()
    fake = _FakeView()
    cube = nc.NavCube(fake, host)
    faces = {n: c for n, _nm, c in nc._FACES}
    name = nc.visible_faces(fake.camera_basis())[0]
    cs = faces[name]
    cen = [sum(c[i] for c in cs) / 4.0 for i in range(3)]
    sx, sy, _d = nc.project(cen, fake.camera_basis())
    px = cube._center[0] + sx * cube._radius
    py = cube._center[1] - sy * cube._radius
    _press(cube, px, py)
    assert fake.views == [name]


def test_clicking_an_arrow_calls_orbit(qapp):
    import nav_cube as nc
    from PySide6.QtWidgets import QWidget
    host = QWidget()
    fake = _FakeView()
    cube = nc.NavCube(fake, host)
    tri = cube._arrows["az_right"]
    c = tri.boundingRect().center()
    _press(cube, c.x(), c.y())
    assert fake.orbits and fake.orbits[0][0] == 90.0


def test_nav_cube_grab_is_non_null(qapp):
    import nav_cube as nc
    from PySide6.QtWidgets import QWidget
    host = QWidget()
    cube = nc.NavCube(_FakeView(), host)
    pm = cube.grab()
    assert not pm.isNull()
    assert pm.width() == nc.NavCube.SIZE
