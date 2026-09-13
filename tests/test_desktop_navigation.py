"""Viewport navigation controls — the Midas/SAP-style camera scheme plus the
on-viewport toolbar (``desktop/nav_toolbar.py``) and ``ModelView`` wiring.

Camera *translation* math (pan / zoom-to-cursor) needs a real render-window
size and can't be exercised headlessly (the offscreen window is 0×0, same as
the existing selection projection), so those are covered by no-raise + state
checks; the orbit / zoom-scale / lock / tool-sync logic is fully asserted.
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


def _view():
    import model_view as mv
    from demo_model import demo_project
    v = mv.ModelView()
    v.resize(900, 600)
    v.set_model(demo_project().build_model())
    return v


# --- tools & toolbar --------------------------------------------------------
def _tool_btn(v, cid):
    return v._nav_bar._buttons_by_id[cid]


def test_toolbar_has_navigation_tools(qapp):
    v = _view()
    for cid in ("select", "orbit", "pan", "zoomwin"):
        assert v._nav_bar._buttons_by_id[cid].isCheckable()


def test_setting_a_tool_syncs_the_toolbar(qapp):
    v = _view()
    v.set_mode("pan")
    assert _tool_btn(v, "pan").isChecked()
    assert not _tool_btn(v, "select").isChecked()
    v.set_mode("select")
    assert _tool_btn(v, "select").isChecked()


def test_ribbon_only_tool_leaves_no_nav_tool_pressed(qapp):
    v = _view()
    v.set_mode("orbit")
    assert _tool_btn(v, "orbit").isChecked()
    v.set_mode("window")                       # a ribbon selection tool
    for cid in ("select", "orbit", "pan", "zoomwin"):
        assert not _tool_btn(v, cid).isChecked()


# --- 2-D rotation lock ------------------------------------------------------
def test_planar_model_locks_rotation_by_default(qapp):
    v = _view()                                # demo frame is ndm == 2
    assert v.rotation_locked() is True
    assert _tool_btn(v, "orbit").isEnabled() is False


def test_orbit_is_blocked_while_locked_then_works_unlocked(qapp):
    v = _view()
    b0 = v.camera_basis()
    v._orbit_pixels(40, 10)                    # locked → no change
    assert np.allclose(np.vstack(b0), np.vstack(v.camera_basis()))
    v.set_rotation_locked(False)
    assert _tool_btn(v, "orbit").isEnabled() is True
    v._orbit_pixels(40, 10)                    # now it rotates
    assert not np.allclose(np.vstack(b0), np.vstack(v.camera_basis()))


def test_lock_button_reflects_and_drives_state(qapp):
    v = _view()
    v.set_rotation_locked(False)
    lock = v._nav_bar._buttons_by_id["lock"]
    assert lock.isChecked() is False
    lock.setChecked(True)                      # user clicks the lock
    assert v.rotation_locked() is True


# --- zoom -------------------------------------------------------------------
def test_zoom_changes_parallel_scale_both_ways(qapp):
    v = _view()
    s0 = v.camera.parallel_scale
    v.zoom_in()
    assert v.camera.parallel_scale < s0        # smaller scale = closer
    s1 = v.camera.parallel_scale
    v.zoom_out()
    assert v.camera.parallel_scale > s1


def test_wheel_event_zooms(qapp):
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QWheelEvent
    v = _view()
    s0 = v.camera.parallel_scale
    ev = QWheelEvent(QPointF(450, 300), QPointF(450, 300), QPoint(0, 0),
                     QPoint(0, 120), Qt.MouseButton.NoButton,
                     Qt.KeyboardModifier.NoModifier,
                     Qt.ScrollPhase.NoScrollPhase, False)
    v.wheelEvent(ev)
    assert v.camera.parallel_scale < s0        # wheel up → zoom in


def test_orbit_keeps_model_within_clipping_planes(qapp):
    """Regression: a hand-driven orbit must refresh the near/far clipping range
    (a flat 2-D model starts with a razor-thin range that would slice it once
    rotated). Assert every scene-bounds corner lies between the planes."""
    v = _view()
    v.set_rotation_locked(False)
    for _ in range(15):
        v._orbit_pixels(15, 8)
    cam = v.camera
    pos = np.asarray(cam.position, float)
    fwd = np.asarray(cam.focal_point, float) - pos
    fwd = fwd / float(np.linalg.norm(fwd))
    b = v.renderer.ComputeVisiblePropBounds()   # xmin,xmax,ymin,ymax,zmin,zmax
    corners = np.array([[b[ix], b[iy], b[iz]]
                        for ix in (0, 1) for iy in (2, 3) for iz in (4, 5)])
    dists = (corners - pos) @ fwd
    near, far = cam.GetClippingRange()
    assert near <= dists.min() + 1e-6
    assert dists.max() <= far + 1e-6


# --- no-raise camera moves (translation degenerate offscreen) ---------------
def test_pan_and_fit_selected_do_not_raise(qapp):
    from PySide6.QtCore import QPointF
    v = _view()
    v._pan_pixels(QPointF(400, 300), QPointF(460, 330))   # no exception
    v.highlight([("node", 1), ("member", 1)])
    v.fit_selected()
    v.highlight([])
    v.fit_selected()                                       # empty → fit-all path
