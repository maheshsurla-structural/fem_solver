"""The 3-D model viewport — a PyVista/VTK widget that renders a Model.

Subclasses ``pyvistaqt.QtInteractor`` (itself a Qt widget) so it drops
straight into the main window as the central widget. ``set_model`` is the
one entry point: clear, draw members / nodes / supports, frame the camera.
"""
from __future__ import annotations

import math
from functools import partial

import numpy as np
from PySide6.QtCore import QPoint, QRect, QSize, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPolygon
from PySide6.QtWidgets import QApplication, QLabel, QRubberBand, QWidget
from pyvistaqt import QtInteractor

import model_geometry as mg
import style
from nav_cube import NavCube, camera_basis_from
from nav_toolbar import NavToolbar


# Model-entity inks live in the theme (``style.V_*``) and are read at render
# time, so the viewport restyles with the rest of the app on a theme switch.
# Diagram colour keys off the component (N / V / M):
def _diagram_color(kind: str) -> str:
    return {"N": style.V_DIAG_N, "V": style.V_DIAG_V,
            "M": style.V_DIAG_M}.get(kind, style.V_DIAG_N)


def _nice_step(span: float) -> float:
    """A tidy grid spacing ≈ span/10, snapped to 1 / 2 / 5 × 10ⁿ."""
    if span <= 0:
        return 1.0
    raw = span / 10.0
    mag = 10.0 ** math.floor(math.log10(raw))
    for m in (1.0, 2.0, 5.0):
        if m * mag >= raw:
            return m * mag
    return 10.0 * mag


def _build_ground_grid(model):
    """A CAD ground grid — line segments in the model's ground plane (z = base),
    spanning the model bounds at a 'nice' step. This is the professional floor
    grid that replaces PyVista's plot-style bounds box. Returns a lines
    ``PolyData``, or ``None`` for an empty model."""
    import pyvista as pv
    if model is None or not len(getattr(model, "nodes", {})):
        return None
    _t, pts, _i = mg.node_points(model)
    pts = np.asarray(pts, dtype=float)
    if not len(pts):
        return None
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    span = float(max(hi[0] - lo[0], hi[1] - lo[1], 1.0))
    step = _nice_step(span)
    x0 = math.floor((lo[0] - step) / step) * step
    x1 = math.ceil((hi[0] + step) / step) * step
    y0 = math.floor((lo[1] - step) / step) * step
    y1 = math.ceil((hi[1] + step) / step) * step
    z = float(lo[2]) - 0.01 * span               # a hair behind the model plane
    verts, lines = [], []

    def _seg(a, b):
        i = len(verts)
        verts.append(a)
        verts.append(b)
        lines.extend((2, i, i + 1))

    n = x0
    while n <= x1 + step * 0.5:                   # lines of constant x
        _seg((n, y0, z), (n, y1, z))
        n += step
    n = y0
    while n <= y1 + step * 0.5:                   # lines of constant y
        _seg((x0, n, z), (x1, n, z))
        n += step
    poly = pv.PolyData(np.asarray(verts, dtype=float))
    poly.lines = np.asarray(lines, dtype=np.int64)
    return poly


def _deformed_point(node, scale: float) -> np.ndarray:
    """A node's deformed position (3-vec, z=0 for 2-D) = coords + scale·disp."""
    c = np.asarray(node.coords, dtype=float).ravel()
    d = np.asarray(node.disp, dtype=float).ravel()
    x = float(c[0]) + scale * float(d[0])
    y = (float(c[1]) + scale * float(d[1])) if c.size > 1 else 0.0
    z = float(c[2]) if c.size > 2 else 0.0
    return np.array([x, y, z], dtype=float)


class _PolygonOverlay(QWidget):
    """Transparent overlay for lasso (polygon) selection — click vertices,
    double-click / right-click to close. Composites over the OpenGL viewport."""

    def __init__(self, parent, on_polygon):
        super().__init__(parent)
        self._on_polygon = on_polygon
        self._pts = []
        self._cur = None
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setMouseTracking(True)

    def reset(self) -> None:
        self._pts, self._cur = [], None
        self.update()

    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            self._pts.append(ev.position().toPoint())
            self.update()
        elif ev.button() == Qt.MouseButton.RightButton:
            self._finish()

    def mouseMoveEvent(self, ev):
        if self._pts:
            self._cur = ev.position().toPoint()
            self.update()

    def mouseDoubleClickEvent(self, ev):
        self._finish()

    def _finish(self) -> None:
        pts = list(self._pts)
        self.reset()
        if len(pts) >= 3:
            self._on_polygon(pts)

    def paintEvent(self, ev):
        if not self._pts:
            return
        sel = QColor(style.V_SELECTION)
        p = QPainter(self)
        p.setPen(QPen(sel, 1.5, Qt.PenStyle.DashLine))
        p.setBrush(QColor(sel.red(), sel.green(), sel.blue(), 40))
        p.drawPolygon(QPolygon(self._pts + ([self._cur] if self._cur else [])))
        p.setBrush(sel)
        for pt in self._pts:
            p.drawEllipse(pt, 2, 2)


class ModelView(QtInteractor):
    # emitted when the interaction tool changes, so the shell can keep the
    # ribbon + the on-viewport toolbar in sync.
    mode_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.set_background(style.VIEW_BG)
        self.enable_parallel_projection()   # orthographic — CAD-style elevations
        self._model = None
        self._replay = None                  # redraws the last scene on restyle
        self._pick_cb = None
        self._add_node_cb = None
        self._add_member_cb = None
        self._add_area_cb = None
        self._show_area_axes = False           # local-axis triads overlay (S6)
        self._show_node_labels = False         # node-number labels overlay
        self._show_elem_labels = False         # element-number labels overlay
        self._mode = "select"
        self._member_start = None
        self._area_pick: list = []            # nodes collected in draw_area mode
        self._wall_plan_pts: list = []        # plan points in draw_wall_plan (W7)
        self._wall_plan_cb = None
        self._snap_on = True
        self._snap_grid = 0.5
        self._show_story_grid = True          # named grid + story-level overlay
        self._story_project = None            # project whose grid/stories we draw
        self._snap_xs: list = []              # named-grid snap targets (W4b)
        self._snap_ys: list = []
        self._snap_zs: list = []              # story-elevation snap targets (W1b)
        self._work_plane = "xy"               # draw work plane: xy | xz | yz | 3pt
        self._work_offset = 0.0               # its fixed-axis coordinate (SI)
        self._plane_origin = None             # 3-pt plane origin (np 3-vec)
        self._plane_normal = None             # 3-pt plane unit normal
        self._region_cb = None
        self._coord_cb = None
        self._rubber = None
        self._band_origin = None
        self._band_additive = False
        self._highlight = []
        self._nav = None                     # active drag: (kind, last QPointF)
        self._zoom_anchor = None             # fixed anchor for a right-drag zoom
        self._poly_overlay = _PolygonOverlay(self, self._polygon_select)
        self._poly_overlay.hide()
        # orientation cube — a CAD-style navigation gizmo pinned top-right; it
        # reads ``camera_basis`` and drives ``set_view`` / ``orbit`` (see nav_cube).
        self._nav_cube = NavCube(self, self)
        # viewport tool strip (orbit / pan / zoom-window / fit / selection …).
        # It drives this view's tools + camera helpers, but is *docked* by the
        # shell into its own toolbar band below the ribbon (re-homed there), so
        # it never overlaps the canvas. Created here (parentless) so the view is
        # usable standalone; hidden until the shell shows it.
        self._nav_bar = NavToolbar(self)
        self._nav_bar.hide()
        self._position_nav_cube()
        # empty-state hint — shown (centred) whenever there is no model to draw,
        # instead of a blank canvas (charter §F). Themed via the #canvasHint QSS.
        self._hint = QLabel(
            "No model yet — draw a node, or open a project  (Ctrl+O)", self)
        self._hint.setObjectName("canvasHint")
        self._hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._hint.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._sync_hint()
        try:
            self.enable_point_picking(callback=self._on_point_picked,
                                      left_clicking=True, show_message=False,
                                      show_point=False)
        except Exception:                    # picking optional; never block init
            pass

    def set_pick_callback(self, fn) -> None:
        self._pick_cb = fn

    def set_add_node_callback(self, fn) -> None:
        self._add_node_cb = fn

    def set_add_member_callback(self, fn) -> None:
        self._add_member_cb = fn

    def set_wall_plan_callback(self, fn) -> None:
        """``fn((x1,y1),(x2,y2))`` is called with the two plan points of a wall
        drawn in ``draw_wall_plan`` mode (wall plan W7)."""
        self._wall_plan_cb = fn

    def set_add_area_callback(self, fn) -> None:
        self._add_area_cb = fn

    def set_area_axes(self, on: bool) -> None:
        """Toggle the per-area local-axis triads (slab plan S6) and redraw."""
        self._show_area_axes = bool(on)
        if self._replay is not None:
            try:
                self._replay()
            except Exception:
                pass

    def _draw_area_axes(self, model) -> None:
        triads = mg.area_local_axes(model)
        if triads is None:
            return
        colors = ("#d64545", "#3faf5a", "#3b7dd8")   # e1 red, e2 green, e3 blue
        for poly, color, name in zip(triads, colors,
                                     ("axis1", "axis2", "axis3")):
            if poly is not None:
                self.add_mesh(poly, color=color, line_width=3, name=name)

    def set_node_labels(self, on: bool) -> None:
        """Toggle node-number labels in the viewport and redraw."""
        self._show_node_labels = bool(on)
        self._replay_scene()

    def set_element_labels(self, on: bool) -> None:
        """Toggle element-number labels in the viewport and redraw."""
        self._show_elem_labels = bool(on)
        self._replay_scene()

    def _replay_scene(self) -> None:
        """Redraw whatever the view last rendered (model / deformed / …)."""
        if self._replay is not None:
            try:
                self._replay()
            except Exception:
                pass

    def _draw_id_labels(self, model) -> None:
        """Overlay node- and/or element-number labels per the active toggles."""
        if self._show_node_labels:
            tags, pts, _index = mg.node_points(model)
            if len(pts):
                try:
                    self.add_point_labels(
                        pts, [str(t) for t in tags], font_size=12,
                        text_color=style.V_NODE, shape_opacity=0.15,
                        always_visible=True, name="node_labels")
                except Exception:
                    pass
        if self._show_elem_labels:
            tags, cents = mg.element_centroids(model)
            if len(cents):
                try:
                    self.add_point_labels(
                        cents, [str(t) for t in tags], font_size=12,
                        text_color=style.V_MEMBER, shape_opacity=0.15,
                        always_visible=True, name="elem_labels")
                except Exception:
                    pass

    def set_snap(self, on: bool, grid: float) -> None:
        self._snap_on = bool(on)
        self._snap_grid = max(float(grid), 1e-3)

    def set_region_callback(self, fn) -> None:
        self._region_cb = fn

    def set_coord_callback(self, fn) -> None:
        """``fn(x, y)`` receives the world coordinate under the cursor on the
        model's ground plane (z = 0), for a live status-bar readout."""
        self._coord_cb = fn

    # --- navigation & selection input (Midas/SAP-style scheme) --------------
    #     All camera motion is handled here at the Qt level so we own the
    #     bindings (middle = pan · wheel = zoom-to-cursor · right = dynamic
    #     zoom · left = the active tool). Native VTK move events are never
    #     forwarded, so the trackball never rotates behind our back; left
    #     press/release is forwarded only for select/draw so the picker fires.
    def _start_band(self, pos, ev) -> None:
        self._band_origin = pos.toPoint()
        self._band_additive = bool(ev.modifiers() & (
            Qt.KeyboardModifier.ShiftModifier
            | Qt.KeyboardModifier.ControlModifier))
        if self._rubber is None:
            self._rubber = QRubberBand(QRubberBand.Shape.Rectangle, self)
        self._rubber.setGeometry(QRect(self._band_origin, QSize()))
        self._rubber.show()

    def mousePressEvent(self, ev):
        pos = ev.position()
        b = ev.button()
        if b == Qt.MouseButton.MiddleButton or (
                b == Qt.MouseButton.LeftButton and self._mode == "pan"):
            self._nav = ("pan", pos)                      # pan (middle / Pan tool)
            return
        if b == Qt.MouseButton.RightButton:
            self._nav = ("zoom", pos)                     # dynamic zoom (drag)
            self._zoom_anchor = pos
            return
        if b == Qt.MouseButton.LeftButton and self._mode == "orbit":
            self._nav = ("orbit", pos)
            return
        if b == Qt.MouseButton.LeftButton and self._mode in ("window", "zoomwin"):
            self._start_band(pos, ev)
            return
        if b == Qt.MouseButton.LeftButton:                # select / draw → picker
            super().mousePressEvent(ev)
            return
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        pos = ev.position()
        if self._coord_cb is not None:
            wc = self._world_on_ground(pos)
            if wc is not None:
                self._coord_cb(*wc)
        if self._nav is not None:
            kind, last = self._nav
            if kind == "pan":
                self._pan_pixels(last, pos)
            elif kind == "orbit":
                self._orbit_pixels(pos.x() - last.x(), pos.y() - last.y())
            elif kind == "zoom":
                a = self._zoom_anchor or pos
                self._zoom(1.0 - (pos.y() - last.y()) * 0.01, a.x(), a.y())
            self._nav = (kind, pos)
            return
        if self._band_origin is not None and self._mode in ("window", "zoomwin"):
            self._rubber.setGeometry(
                QRect(self._band_origin, pos.toPoint()).normalized())
            return
        # swallow every other move — never let VTK's trackball rotate/pan

    def mouseReleaseEvent(self, ev):
        if self._nav is not None:
            self._nav = None
            return
        if (self._band_origin is not None
                and ev.button() == Qt.MouseButton.LeftButton):
            rect = QRect(self._band_origin,
                         ev.position().toPoint()).normalized()
            mode = self._mode
            self._band_origin = None
            if self._rubber is not None:
                self._rubber.hide()
            if rect.width() > 3 and rect.height() > 3:
                if mode == "zoomwin":
                    self._zoom_to_box(rect)
                else:
                    self._window_select(rect, self._band_additive)
            return
        if ev.button() == Qt.MouseButton.LeftButton:      # complete a pick
            super().mouseReleaseEvent(ev)
            return
        super().mouseReleaseEvent(ev)

    def wheelEvent(self, ev):
        """Zoom toward the cursor (replaces VTK's zoom-about-centre)."""
        d = ev.angleDelta().y()
        if d == 0:
            return
        pos = ev.position()
        self._zoom(1.15 if d > 0 else 1.0 / 1.15, pos.x(), pos.y())
        ev.accept()

    # --- camera navigation helpers -----------------------------------------
    def _focal_world(self, x_px, y_px):
        """Widget pixel → world point on the camera's focal plane (parallel
        projection). Panning / zoom-to-cursor use it to keep the point under
        the cursor fixed while the camera moves."""
        ren = self.renderer
        dpr = self.devicePixelRatioF() or 1.0
        fp = self.camera.focal_point
        ren.SetWorldPoint(float(fp[0]), float(fp[1]), float(fp[2]), 1.0)
        ren.WorldToDisplay()
        fz = ren.GetDisplayPoint()[2]
        ren.SetDisplayPoint(x_px * dpr, (self.height() - y_px) * dpr, fz)
        ren.DisplayToWorld()
        w = ren.GetWorldPoint()
        h = w[3] if w[3] else 1.0
        return np.array([w[0] / h, w[1] / h, w[2] / h])

    def _translate_camera(self, delta) -> None:
        cam = self.camera
        cam.SetPosition(*(np.asarray(cam.position, float) + delta))
        cam.SetFocalPoint(*(np.asarray(cam.focal_point, float) + delta))

    def _render_nav(self) -> None:
        """Recompute the near/far clipping planes for the whole scene, then
        render. Camera moves done by hand (orbit / pan / zoom) don't refresh
        the clipping range on their own, so without this a rotated model gets
        sliced by a stale near plane (part of it disappears)."""
        try:
            self.renderer.ResetCameraClippingRange()
        except Exception:
            pass
        self.render()

    def _pan_pixels(self, last, cur) -> None:
        d = (self._focal_world(last.x(), last.y())
             - self._focal_world(cur.x(), cur.y()))
        self._translate_camera(d)
        self._render_nav()

    def _orbit_pixels(self, dx, dy) -> None:
        cam = self.camera
        cam.Azimuth(-dx * 0.35)
        cam.Elevation(dy * 0.35)
        cam.OrthogonalizeViewUp()
        self._render_nav()

    def _zoom(self, factor, ax=None, ay=None) -> None:
        """Parallel-projection zoom by ``factor`` (>1 zooms in) about the pixel
        (ax, ay), which is held fixed under the cursor."""
        factor = max(1e-3, float(factor))
        cam = self.camera
        if ax is None:
            ax, ay = self.width() / 2.0, self.height() / 2.0
        before = self._focal_world(ax, ay)
        cam.SetParallelScale(cam.parallel_scale / factor)
        after = self._focal_world(ax, ay)
        self._translate_camera(before - after)
        self._render_nav()

    def _zoom_to_box(self, rect) -> None:
        center = self._focal_world(rect.center().x(), rect.center().y())
        cam = self.camera
        frac = max(rect.width() / max(self.width(), 1),
                   rect.height() / max(self.height(), 1), 1e-3)
        self._translate_camera(center - np.asarray(cam.focal_point, float))
        cam.SetParallelScale(cam.parallel_scale * frac)
        self._render_nav()

    def zoom_in(self) -> None:
        self._zoom(1.25)

    def zoom_out(self) -> None:
        self._zoom(1.0 / 1.25)

    def fit_selected(self) -> None:
        """Frame the current selection (falls back to fit-all when nothing is
        selected)."""
        pts = []
        if self._model is not None:
            for kind, ident in self._highlight:
                if kind == "node" and ident in self._model.nodes:
                    pts.append(mg.to_xyz(self._model.nodes[ident].coords))
                elif kind == "member":
                    for tag in self._member_element_tags(ident):
                        for c in self._model.element(tag).node_coords():
                            pts.append(mg.to_xyz(c))
        if not pts:
            self.reset_camera()
            self.render()
            return
        p = np.asarray(pts, float)
        lo, hi = p.min(axis=0), p.max(axis=0)
        pad = 0.1 * (float(np.linalg.norm(hi - lo)) or 1.0)
        self.renderer.ResetCamera(lo[0] - pad, hi[0] + pad, lo[1] - pad,
                                  hi[1] + pad, lo[2] - pad, hi[2] + pad)
        self.render()

    def current_mode(self) -> str:
        return self._mode

    def _project(self, world):
        """World coords -> widget (logical) pixels, y down."""
        ren = self.renderer
        ren.SetWorldPoint(float(world[0]), float(world[1]), float(world[2]), 1.0)
        ren.WorldToDisplay()
        dx, dy, _dz = ren.GetDisplayPoint()
        dpr = self.devicePixelRatioF() or 1.0
        return dx / dpr, self.height() - dy / dpr

    def _world_on_ground(self, pos):
        """Widget pixel → world (x, y) on the z = 0 ground plane, by intersecting
        the cursor ray with that plane. Works for ortho and iso views; returns
        None if the ray is parallel to the plane or the transform is unready."""
        try:
            ren = self.renderer
            dpr = self.devicePixelRatioF() or 1.0
            dx = pos.x() * dpr
            dy = (self.height() - pos.y()) * dpr
            ray = []
            for zd in (0.0, 1.0):                # near + far display points
                ren.SetDisplayPoint(dx, dy, zd)
                ren.DisplayToWorld()
                w = ren.GetWorldPoint()
                h = w[3] if w[3] else 1.0
                ray.append((w[0] / h, w[1] / h, w[2] / h))
            (x0, y0, z0), (x1, y1, z1) = ray
            dz = z1 - z0
            if abs(dz) < 1e-12:
                return None
            t = -z0 / dz
            return x0 + t * (x1 - x0), y0 + t * (y1 - y0)
        except Exception:
            return None

    def _window_select(self, rect, additive=False) -> None:
        if self._model is None or self._region_cb is None:
            return
        inside = {}
        for tag, n in self._model.nodes.items():
            x, y = self._project(mg.to_xyz(n.coords))
            inside[tag] = rect.contains(int(x), int(y))
        refs = [("node", t) for t, v in inside.items() if v]
        refs += self._members_inside(inside)
        self._region_cb(refs, additive)

    def _polygon_select(self, pts) -> None:
        if self._model is None or self._region_cb is None:
            return
        poly = QPolygon(pts)
        additive = bool(QApplication.keyboardModifiers() & (
            Qt.KeyboardModifier.ShiftModifier
            | Qt.KeyboardModifier.ControlModifier))
        inside = {}
        for tag, n in self._model.nodes.items():
            x, y = self._project(mg.to_xyz(n.coords))
            inside[tag] = poly.containsPoint(QPoint(int(x), int(y)),
                                             Qt.FillRule.OddEvenFill)
        refs = [("node", t) for t, v in inside.items() if v]
        refs += self._members_inside(inside)
        self._region_cb(refs, additive)

    def _members_inside(self, inside: dict) -> list:
        """('member', id) refs for every member both of whose end nodes are in
        the ``inside`` map. A beam split at a slab edge (BE2) has several 2-node
        sub-elements; each is decoded back to its project member id and the
        member is reported once (BE3)."""
        from project import decode_member_id
        refs, seen = [], set()
        for tag, e in self._model.elements.items():
            nt = e.node_tags
            if len(nt) == 2 and inside.get(nt[0]) and inside.get(nt[1]):
                mid = decode_member_id(tag)
                if mid is not None and mid not in seen:
                    seen.add(mid)
                    refs.append(("member", mid))
        return refs

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if self._poly_overlay.isVisible():
            self._poly_overlay.setGeometry(self.rect())
        if self._hint.isVisible():
            self._hint.setGeometry(self.rect())
        self._position_nav_cube()

    def _position_nav_cube(self) -> None:
        """Pin the orientation cube to the top-right corner of the viewport."""
        cube = getattr(self, "_nav_cube", None)
        if cube is None:
            return
        margin = 12
        cube.move(self.width() - cube.width() - margin, margin)
        cube.raise_()

    def camera_basis(self):
        """Screen basis (right, up, forward) as unit world vectors for the
        active camera — consumed by the navigation cube. ``None`` if unready."""
        try:
            cam = self.camera
            return camera_basis_from(cam.position, cam.focal_point, cam.up)
        except Exception:
            return None

    def orbit(self, d_azimuth: float, d_elevation: float) -> None:
        """Rotate the camera about the model by the given degrees (azimuth /
        elevation), keep the up-vector sane, then re-fit — used by the cube's
        orbit chevrons to bring a hidden face round to the front."""
        try:
            cam = self.camera
            if d_azimuth:
                cam.Azimuth(float(d_azimuth))
            if d_elevation:
                cam.Elevation(float(d_elevation))
            cam.OrthogonalizeViewUp()
        except Exception:
            return
        self.reset_camera()
        self.render()

    def _sync_hint(self) -> None:
        """Show the centred empty-state hint when there is no model to draw."""
        empty = self._model is None or not len(getattr(self._model, "nodes", {}))
        self._hint.setVisible(empty)
        if empty:
            self._hint.setGeometry(self.rect())
            self._hint.raise_()
        self._position_nav_cube()             # keep the cube above the hint

    _TOOL_CURSORS = {
        "pan": Qt.CursorShape.OpenHandCursor,
        "orbit": Qt.CursorShape.SizeAllCursor,
        "zoomwin": Qt.CursorShape.CrossCursor,
        "window": Qt.CursorShape.CrossCursor,
        "draw_node": Qt.CursorShape.CrossCursor,
        "draw_member": Qt.CursorShape.CrossCursor,
        "draw_area": Qt.CursorShape.CrossCursor,
        "draw_wall_plan": Qt.CursorShape.CrossCursor,
    }

    def set_mode(self, mode: str) -> None:
        self._mode = mode
        self._member_start = None
        self._area_pick = []
        self._wall_plan_pts = []
        self._nav = None
        self.remove_actor("groundplane", render=False)
        if mode in ("draw_node", "draw_wall_plan"):
            self._add_ground_plane()
        if mode == "polygon":
            self._poly_overlay.setGeometry(self.rect())
            self._poly_overlay.reset()
            self._poly_overlay.raise_()
            self._poly_overlay.show()
        else:
            self._poly_overlay.hide()
        self.setCursor(self._TOOL_CURSORS.get(mode, Qt.CursorShape.ArrowCursor))
        self.render()
        self.mode_changed.emit(mode)

    # draw work plane (wall plan W1b): 'xy' horizontal (normal Z), 'xz' elevation
    # (normal Y), 'yz' elevation (normal X); the fixed axis sits at _work_offset.
    _PLANE_NORMAL = {"xy": (0.0, 0.0, 1.0), "xz": (0.0, 1.0, 0.0),
                     "yz": (1.0, 0.0, 0.0)}
    _PLANE_AXIS = {"xy": 2, "xz": 1, "yz": 0}     # index of the fixed coordinate

    def set_work_plane(self, kind: str, offset: float = 0.0) -> None:
        """Set the plane new draw-node clicks land on: 'xy' (horizontal), 'xz'
        or 'yz' (vertical elevations), at fixed-axis coordinate ``offset``.
        Redraws the plane while the draw-node tool is active (wall plan W1b)."""
        self._work_plane = kind if kind in self._PLANE_NORMAL else "xy"
        self._work_offset = float(offset)
        self._redraw_work_plane()

    def set_work_plane_3pt(self, p0, p1, p2) -> bool:
        """Set an arbitrary work plane through three points (wall plan W1c). The
        plane origin is ``p0`` and its normal is ``(p1−p0)×(p2−p0)``. Returns
        False (leaving the plane unchanged) if the three points are collinear."""
        p0, p1, p2 = (np.asarray(p, dtype=float) for p in (p0, p1, p2))
        n = np.cross(p1 - p0, p2 - p0)
        if np.linalg.norm(n) < 1e-9:
            return False
        self._plane_origin = p0
        self._plane_normal = n / np.linalg.norm(n)
        self._work_plane = "3pt"
        self._redraw_work_plane()
        return True

    def _snap_plane_xy(self, p):
        """Snap a raw click to the XY work plane's (x, y): spacing grid, then a
        named grid line when close (wall plan W7 plan draw)."""
        x = _snap(p[0], self._snap_grid) if self._snap_on else round(float(p[0]), 3)
        y = _snap(p[1], self._snap_grid) if self._snap_on else round(float(p[1]), 3)
        if self._snap_xs or self._snap_ys:
            tol = max(self._snap_grid * 0.5, 1e-3)
            x, y, _z = mg.snap_to_grid(x, y, 0.0, self._snap_xs, self._snap_ys,
                                       [], tol)
        return x, y

    def _redraw_work_plane(self) -> None:
        if self._mode == "draw_node":
            self.remove_actor("groundplane", render=False)
            self._add_ground_plane()
            self.render()

    def _work_plane_origin_normal(self):
        """(origin, unit-normal) of the active work plane as np 3-vectors."""
        if self._work_plane == "3pt" and self._plane_origin is not None:
            return self._plane_origin, self._plane_normal
        ax = self._PLANE_AXIS.get(self._work_plane, 2)
        o = np.zeros(3)
        o[ax] = self._work_offset
        return o, np.asarray(self._PLANE_NORMAL.get(self._work_plane,
                                                    (0.0, 0.0, 1.0)), float)

    def _add_ground_plane(self) -> None:
        import pyvista as pv
        if self._model is not None and len(self._model.nodes):
            _t, pts, _i = mg.node_points(self._model)
            lo, hi = pts.min(axis=0), pts.max(axis=0)
            ctr = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2]
            size = float(np.max(hi - lo)) * 2.0 + 10.0
        else:
            ctr = [0.0, 0.0, 0.0]
            size = 20.0
        origin, normal = self._work_plane_origin_normal()
        if self._work_plane == "3pt":
            ctr = list(origin)                    # centre the plane at its origin
        else:
            ctr[self._PLANE_AXIS[self._work_plane]] = self._work_offset
        plane = pv.Plane(center=tuple(ctr), direction=tuple(normal),
                         i_size=size, j_size=size)
        self.add_mesh(plane, color=style.ACCENT, opacity=0.12, name="groundplane",
                      pickable=True)

    def _on_point_picked(self, *args) -> None:
        if self._model is None or not args:
            return
        try:
            p = np.asarray(args[0], dtype=float).ravel()
        except Exception:
            return
        if p.size < 3:
            return
        p = p[:3]
        if self._mode == "draw_node":
            if self._add_node_cb is not None:
                if self._work_plane == "3pt":
                    # arbitrary plane (W1c): project the click onto it; no
                    # axis-aligned snapping applies to a tilted plane
                    o, n = self._work_plane_origin_normal()
                    q = p - float(np.dot(p - o, n)) * n
                    self._add_node_cb(round(float(q[0]), 4),
                                      round(float(q[1]), 4), round(float(q[2]), 4))
                    return
                # axis-aligned plane (W1b): pin the fixed axis to the offset,
                # spacing-snap the two in-plane axes.
                fixed = self._PLANE_AXIS[self._work_plane]
                xyz = [float(p[0]), float(p[1]), float(p[2])]
                xyz[fixed] = self._work_offset          # exact on the plane
                for i in range(3):
                    if i == fixed:
                        continue
                    xyz[i] = (_snap(xyz[i], self._snap_grid) if self._snap_on
                              else round(xyz[i], 3))
                # a named grid line / story level (W4b) wins over the spacing
                # grid when the click lands within half a spacing step of it
                if self._snap_xs or self._snap_ys or self._snap_zs:
                    tol = max(self._snap_grid * 0.5, 1e-3)
                    sx, sy, sz = mg.snap_to_grid(
                        xyz[0], xyz[1], xyz[2], self._snap_xs, self._snap_ys,
                        self._snap_zs, tol)
                    snapped = [sx, sy, sz]
                    for i in range(3):              # never move the fixed axis
                        if i != fixed:
                            xyz[i] = snapped[i]
                self._add_node_cb(xyz[0], xyz[1], xyz[2])
            return
        if self._mode == "draw_wall_plan":
            # collect two plan points on the (XY) work plane; on the second,
            # hand both to the callback to build the wall / stack (W7)
            if self._wall_plan_cb is not None:
                x, y = self._snap_plane_xy(p)
                self._wall_plan_pts.append((x, y))
                if len(self._wall_plan_pts) >= 2:
                    p1, p2 = self._wall_plan_pts[:2]
                    self._wall_plan_pts = []
                    self._wall_plan_cb(p1, p2)
            return
        tol = max(mg.model_span(self._model) * 0.05, 0.15)
        sel = mg.nearest_item(self._model, p, tol)
        if sel is None:
            return
        if self._mode == "draw_member":
            if sel[0] != "node":
                return
            if self._member_start is None:
                self._member_start = sel[1]
                self.highlight([("node", sel[1])])
            elif sel[1] != self._member_start:
                if self._add_member_cb is not None:
                    self._add_member_cb(self._member_start, sel[1])
                self._member_start = None
            return
        if self._mode == "draw_area":
            if sel[0] != "node":
                return
            nid = sel[1]
            # click the first corner again to close the loop (tri / quad /
            # polygon — any ≥3 corners); re-clicking any other corner is ignored
            if self._area_pick and nid == self._area_pick[0] \
                    and len(self._area_pick) >= 3:
                self._finish_area()
                return
            if nid in self._area_pick:
                return                            # ignore repeats mid-loop
            self._area_pick.append(nid)
            self.highlight([("node", n) for n in self._area_pick])
            return
        if self._pick_cb is not None:
            self._pick_cb(*sel)

    def _finish_area(self) -> None:
        nodes = list(self._area_pick)
        self._area_pick = []
        self.clear_highlight()
        if len(nodes) >= 3 and self._add_area_cb is not None:
            self._add_area_cb(nodes)

    def highlight(self, items) -> None:
        """Highlight a set of [(kind, id), …] — all selected nodes + members."""
        self._highlight = [tuple(it) for it in items]
        self._draw_highlight()

    def clear_highlight(self) -> None:
        self._highlight = []
        self.remove_actor("selection", render=False)
        self.remove_actor("selection_areas", render=False)
        self.remove_actor("selection_nodes", render=True)

    def _member_element_tags(self, member_id) -> list:
        """Engine element tags in the displayed model that belong to project
        member ``member_id`` — its own tag when unsplit, or every sub-element of
        a beam split at a slab edge (BE2/BE3). Lets highlight/framing work off a
        project member id even though the built model no longer carries it."""
        if self._model is None:
            return []
        if member_id in self._model.elements:
            return [member_id]                    # fast path: unsplit member
        from project import decode_member_id
        return [tag for tag in self._model.elements
                if decode_member_id(tag) == member_id]

    def _draw_highlight(self) -> None:
        self.remove_actor("selection", render=False)
        self.remove_actor("selection_nodes", render=False)
        self.remove_actor("selection_areas", render=False)
        if not self._highlight or self._model is None:
            self.render()
            return
        import pyvista as pv
        span = mg.model_span(self._model)
        pts, lines, area_faces = [], [], []
        for kind, ident in self._highlight:
            if kind == "node" and ident in self._model.nodes:
                pts.append(mg.to_xyz(self._model.nodes[ident].coords))
            elif kind == "member":
                for tag in self._member_element_tags(ident):
                    line = mg.element_line(self._model.element(tag))
                    if line is not None:
                        lines.append(line)
            elif kind == "area":
                faces = mg.area_faces_mesh(self._model, ident)
                if faces is not None:
                    area_faces.append(faces)
        if lines:
            merged = lines[0] if len(lines) == 1 else pv.merge(lines)
            self.add_mesh(merged.tube(radius=max(span * 0.006, 2e-3)),
                          color=style.V_SELECTION, name="selection")
        if area_faces:
            merged = (area_faces[0] if len(area_faces) == 1
                      else pv.merge(area_faces))
            self.add_mesh(merged, color=style.V_SELECTION, opacity=0.5,
                          show_edges=True, edge_color=style.V_SELECTION,
                          name="selection_areas")
        if pts:
            self.add_points(np.asarray(pts, dtype=float), color=style.V_SELECTION,
                            render_points_as_spheres=True, point_size=20,
                            name="selection_nodes")
        self.render()

    def _draw_grid(self) -> None:
        """A CAD ground grid in the model plane plus a corner orientation triad —
        the professional replacement for PyVista's plot-style bounds box (the old
        'X Axis / Y Axis' frame). Colours come from the palette."""
        self.remove_actor("groundgrid", render=False)
        try:
            grid = _build_ground_grid(self._model)
        except Exception:
            grid = None
        if grid is not None:
            self.add_mesh(grid, color=style.VIEW_GRID, line_width=1,
                          name="groundgrid", pickable=False)
        try:                                 # corner XYZ orientation gizmo
            self.add_axes(color=style.TEXT, line_width=2)
        except Exception:
            pass

    def set_story_grid(self, project) -> None:
        """Draw the named grid lines + story levels overlay and register the
        grid coordinates as snap targets (wall plan W4b). Call after
        ``set_model`` (which clears the scene). Safe to call with no stories or
        grid — it simply removes any stale overlay."""
        self._story_project = project
        try:
            self._snap_xs, self._snap_ys, self._snap_zs = mg.snap_targets(project)
        except Exception:
            self._snap_xs, self._snap_ys, self._snap_zs = [], [], []
        self.remove_actor("storygrid", render=False)
        if not self._show_story_grid or project is None or self._model is None:
            self.render()
            return
        try:
            poly = mg.grid_story_mesh(project, self._model)
        except Exception:
            poly = None
        if poly is not None:
            self.add_mesh(poly, color=style.ACCENT, line_width=1,
                          opacity=0.6, name="storygrid", pickable=False)
        self.remove_actor("gridbubbles", render=False)
        try:
            bpts, blab = mg.grid_bubble_labels(project, self._model)
        except Exception:
            bpts, blab = [], []
        if bpts:
            try:
                self.add_point_labels(
                    np.asarray(bpts, dtype=float), blab, font_size=12,
                    text_color=style.ACCENT, shape="rounded_rect",
                    shape_opacity=0.25, always_visible=True, pickable=False,
                    name="gridbubbles")
            except Exception:
                pass
        self.render()

    def show_story_grid(self, on: bool) -> None:
        """Toggle the grid/story overlay; redraws from the stored project."""
        self._show_story_grid = bool(on)
        if self._story_project is not None:
            self.set_story_grid(self._story_project)

    def apply_theme(self) -> None:
        """Re-read the palette (background + entity inks) and repaint the current
        scene in place, keeping the camera. Mirrors ``SectionCanvas.apply_theme``
        so the shell can restyle every canvas the same way on a theme switch."""
        self.set_background(style.VIEW_BG)
        try:
            cpos = self.camera_position
        except Exception:
            cpos = None
        if self._replay is not None:
            try:
                self._replay()               # redraw meshes with fresh tokens
            except Exception:
                pass
        if cpos is not None:                 # _replay reframes; restore the view
            try:
                self.camera_position = cpos
            except Exception:
                pass
        self.render()
        try:
            self._nav_cube.apply_theme()     # restyle the cube with the app
            self._nav_bar.apply_theme()
        except Exception:
            pass

    def set_model(self, model) -> None:
        self._model = model
        self._replay = partial(self.set_model, model)
        self.clear()

        span = mg.model_span(model)
        # surface (slab/shell) fill first, so the member tubes overlay its edges
        areas = mg.areas_mesh(model)
        if areas is not None:
            self.add_mesh(areas, color=style.V_AREA, opacity=0.35,
                          show_edges=True, edge_color=style.V_MEMBER,
                          name="areas")
            if self._show_area_axes:
                self._draw_area_axes(model)
        mesh = mg.members_mesh(model)
        if mesh is not None:
            self.add_mesh(mesh.tube(radius=max(span * 0.004, 1e-3)),
                          color=style.V_MEMBER, name="members")

        _tags, pts, _index = mg.node_points(model)
        if len(pts):
            self.add_points(pts, color=style.V_NODE, render_points_as_spheres=True,
                            point_size=14, name="nodes")

        supports = mg.support_points(model)
        if len(supports):
            self.add_points(supports, color=style.V_SUPPORT,
                            render_points_as_spheres=True, point_size=22,
                            name="supports")

        self._draw_id_labels(model)
        self._draw_grid()
        self._frame(model)
        self._draw_highlight()
        if self._mode == "draw_node":
            self._add_ground_plane()
        self._sync_hint()

    def mark_hinges(self, project) -> None:
        """Overlay a magenta marker inside each end of every member that has a
        fiber plastic hinge assigned (``Member.hinge``), placed at the midpoint
        of the resolved hinge region. Call after :meth:`set_model` (which clears
        the scene). No-op when nothing is hinged."""
        coords = {n.id: np.array([n.x, n.y, getattr(n, "z", 0.0)], float)
                  for n in project.nodes}
        pts = []
        for mb in project.members:
            hid = getattr(mb, "hinge", None)
            hinge = project.hinge(hid) if hid else None
            if hinge is None or mb.n1 not in coords or mb.n2 not in coords:
                continue
            a, b = coords[mb.n1], coords[mb.n2]
            L = float(np.linalg.norm(b - a))
            if L <= 0.0:
                continue
            u = (b - a) / L
            lp_i, lp_j = project.resolve_hinge_lengths(mb, hinge)
            pts.append(a + u * (0.5 * lp_i))
            pts.append(b - u * (0.5 * lp_j))
        if pts:
            self.add_points(np.asarray(pts, dtype=float), color=style.V_HINGE,
                            render_points_as_spheres=True, point_size=20,
                            name="hinges")

    def show_deformed(self, model, scale: float) -> None:
        """Draw the deformed shape (coral) over a grey ghost of the model."""
        self._model = model
        self._replay = partial(self.show_deformed, model, scale)
        self.clear()
        span = mg.model_span(model)

        ref = mg.members_mesh(model)
        if ref is not None:
            self.add_mesh(ref.tube(radius=max(span * 0.0025, 1e-3)),
                          color=style.V_REFERENCE, name="reference")

        mesh = mg.deformed_members_mesh(model, scale)
        if mesh is not None:
            self.add_mesh(mesh.tube(radius=max(span * 0.004, 1e-3)),
                          color=style.V_DEFORMED, name="deformed")

        pts = mg.deformed_points(model, scale)
        if len(pts):
            self.add_points(pts, color=style.V_DEFORMED_NODE,
                            render_points_as_spheres=True, point_size=12,
                            name="deformed_nodes")

        supports = mg.support_points(model)
        if len(supports):
            self.add_points(supports, color=style.V_SUPPORT,
                            render_points_as_spheres=True, point_size=20,
                            name="supports")
        self._draw_grid()
        self._frame(model)

    def show_nl_step(self, model, node_disp, member_damage, scale: float,
                     *, member_state=None, color_mode: str = "strain") -> None:
        """Render one nonlinear-analysis step on the main view (plan G-S1):
        the model's deformed shape at ``scale``, each member coloured by its
        hinge state, over a grey ghost of the undeformed model. ``node_disp``
        maps node id -> (dx, dy); ``member_damage`` maps element id -> peak
        |fiber strain|; ``member_state`` maps element id -> ASCE 41 level
        (0..3), all from ``NonlinearResults.step(k)``.

        ``color_mode`` = ``"strain"`` colours by the continuous peak fiber
        strain (YlOrRd); ``"acceptance"`` colours by the discrete IO/LS/CP
        performance level (§16 C5) when ``member_state`` is supplied."""
        self._replay = partial(self.show_nl_step, model, node_disp,
                               member_damage, scale, member_state=member_state,
                               color_mode=color_mode)
        import pyvista as pv
        from matplotlib import colormaps
        from matplotlib.colors import Normalize

        from femsolver.performance.acceptance import LEVEL_COLORS

        # write this step's displacements onto the live model
        for nid, n in model.nodes.items():
            disp = (node_disp or {}).get(nid) or ()
            for i, v in enumerate(disp):          # 2-D (dx,dy) or 3-D (dx,dy,dz)
                if i < len(n.disp):
                    n.disp[i] = float(v)

        self.clear()
        span = mg.model_span(model)
        ref = mg.members_mesh(model)
        if ref is not None:
            self.add_mesh(ref.tube(radius=max(span * 0.0025, 1e-3)),
                          color=style.V_REFERENCE, name="reference")

        dmg = member_damage or {}
        state = member_state or {}
        by_state = color_mode == "acceptance" and bool(state)
        vmax = max([abs(v) for v in dmg.values()] + [1e-9])
        norm = Normalize(0.0, vmax)
        cmap = colormaps["YlOrRd"]
        radius = max(span * 0.004, 1e-3)
        for eid, el in model.elements.items():
            tags = getattr(el, "node_tags", None)
            if not tags or len(tags) < 2:
                continue
            na, nb = model.nodes[tags[0]], model.nodes[tags[-1]]
            pa = _deformed_point(na, scale)
            pb = _deformed_point(nb, scale)
            if by_state:
                lvl = int(state.get(eid, 0))
                rgb = LEVEL_COLORS[max(0, min(lvl, len(LEVEL_COLORS) - 1))]
            elif eid in dmg:
                rgb = cmap(norm(dmg[eid]))[:3]
            else:
                rgb = (0.5, 0.5, 0.5)
            line = pv.PolyData(np.array([pa, pb]),
                               lines=np.array([2, 0, 1], dtype=np.int64))
            self.add_mesh(line.tube(radius=radius), color=rgb, name=f"nlmem{eid}")

        pts = mg.deformed_points(model, scale)
        if len(pts):
            self.add_points(pts, color=style.V_DEFORMED_NODE,
                            render_points_as_spheres=True, point_size=10,
                            name="nl_nodes")
        supports = mg.support_points(model)
        if len(supports):
            self.add_points(supports, color=style.V_SUPPORT,
                            render_points_as_spheres=True, point_size=18,
                            name="supports")
        self._draw_grid()
        self._frame(model)

    def show_area_contour(self, model, quantity: str = "Umag"):
        """Colour-map a nodal displacement quantity over the surface (slab)
        elements (slab plan S7), over a grey ghost of the members. Returns the
        peak |value|, or 0.0 when the model has no areas."""
        self._model = model
        self._replay = partial(self.show_area_contour, model, quantity)
        self.clear()
        span = mg.model_span(model)

        ref = mg.members_mesh(model)
        if ref is not None:
            self.add_mesh(ref.tube(radius=max(span * 0.0025, 1e-3)),
                          color=style.V_REFERENCE, name="members")

        poly = mg.areas_contour_mesh(model, quantity)
        if poly is None or poly.n_points == 0:
            self._draw_grid()
            self._frame(model)
            return 0.0
        vmax = float(np.max(np.abs(poly.point_data["value"])))
        title = mg.AREA_CONTOUR_QUANTITIES.get(quantity, quantity)
        self.add_mesh(poly, scalars="value", cmap="viridis", show_edges=True,
                      edge_color=style.V_MEMBER, name="area_contour",
                      scalar_bar_args={"title": title})

        supports = mg.support_points(model)
        if len(supports):
            self.add_points(supports, color=style.V_SUPPORT,
                            render_points_as_spheres=True, point_size=18,
                            name="supports")
        self._draw_grid()
        self._frame(model)
        return vmax

    def show_area_result(self, model, quantity: str = "M11"):
        """Colour-map a shell stress resultant (moment/membrane/shear) over the
        surface elements (slab plan S7 rich), over a grey member ghost. Signed
        quantities use a diverging map centred at zero. Returns peak |value|."""
        self._model = model
        self._replay = partial(self.show_area_result, model, quantity)
        self.clear()
        span = mg.model_span(model)

        ref = mg.members_mesh(model)
        if ref is not None:
            self.add_mesh(ref.tube(radius=max(span * 0.0025, 1e-3)),
                          color=style.V_REFERENCE, name="members")

        poly = mg.areas_result_mesh(model, quantity)
        if poly is None or poly.n_points == 0:
            self._draw_grid()
            self._frame(model)
            return 0.0
        vals = poly.point_data["value"]
        vmax = float(np.max(np.abs(vals))) if vals.size else 0.0
        label, _unit, signed = mg.AREA_RESULT_QUANTITIES.get(
            quantity, (quantity, "", True))
        kw = dict(scalars="value", show_edges=True, edge_color=style.V_MEMBER,
                  name="area_result", scalar_bar_args={"title": label})
        if signed and vmax > 0:
            kw.update(cmap="coolwarm", clim=(-vmax, vmax))   # centred at 0
        else:
            kw.update(cmap="viridis")
        self.add_mesh(poly, **kw)

        supports = mg.support_points(model)
        if len(supports):
            self.add_points(supports, color=style.V_SUPPORT,
                            render_points_as_spheres=True, point_size=18,
                            name="supports")
        self._draw_grid()
        self._frame(model)
        return vmax

    def show_area_reinforcement(self, model, quantity: str = "As_x_bot", *,
                                cover: float = 0.025, fy: float = 420e6,
                                fc: float = 30e6):
        """Colour-map the required steel area per width (mm²/m) for a Wood-Armer
        design moment over the slab elements (slab plan S9). Returns peak As."""
        self._model = model
        self._replay = partial(self.show_area_reinforcement, model, quantity,
                               cover=cover, fy=fy, fc=fc)
        self.clear()
        span = mg.model_span(model)
        ref = mg.members_mesh(model)
        if ref is not None:
            self.add_mesh(ref.tube(radius=max(span * 0.0025, 1e-3)),
                          color=style.V_REFERENCE, name="members")
        poly = mg.areas_reinforcement_mesh(model, quantity, cover=cover,
                                           fy=fy, fc=fc)
        if poly is None or poly.n_points == 0:
            self._draw_grid()
            self._frame(model)
            return 0.0
        vals = poly.point_data["value"]
        vmax = float(np.nanmax(vals)) if vals.size else 0.0
        label = mg.AREA_REBAR_QUANTITIES.get(
            quantity, (quantity, ""))[0]
        self.add_mesh(poly, scalars="value", cmap="viridis", show_edges=True,
                      edge_color=style.V_MEMBER, name="area_rebar",
                      scalar_bar_args={"title": f"{label}  [mm²/m]"})
        supports = mg.support_points(model)
        if len(supports):
            self.add_points(supports, color=style.V_SUPPORT,
                            render_points_as_spheres=True, point_size=18,
                            name="supports")
        self._draw_grid()
        self._frame(model)
        return vmax

    def show_cut_line(self, p0, p1) -> None:
        """Overlay a bright line for a section cut (slab plan S7 feedback)."""
        import pyvista as pv
        a = np.asarray(mg.to_xyz(p0), dtype=float)
        b = np.asarray(mg.to_xyz(p1), dtype=float)
        line = pv.PolyData(np.array([a, b]),
                           lines=np.array([2, 0, 1], dtype=np.int64))
        span = mg.model_span(self._model) if self._model is not None else 1.0
        self.add_mesh(line.tube(radius=max(span * 0.004, 2e-3)),
                      color=style.V_SELECTION, name="section_cut")
        self.render()

    def show_diagram(self, model, kind: str):
        """Draw the N / V / M diagram over grey members; return max |value|."""
        self._replay = partial(self.show_diagram, model, kind)
        self.clear()
        span = mg.model_span(model)
        ref = mg.members_mesh(model)
        if ref is not None:
            self.add_mesh(ref.tube(radius=max(span * 0.003, 1e-3)),
                          color=style.V_REFERENCE, name="members")
        vmax = mg.diagram_extreme(model, kind)
        scale = (0.16 * span / vmax) if vmax > 0 else 0.0
        fill, outline = mg.diagram_meshes(model, kind, scale)
        color = _diagram_color(kind)
        if fill is not None:
            self.add_mesh(fill, color=color, opacity=0.35, name="diagram_fill")
        if outline is not None:
            self.add_mesh(outline, color=color, line_width=2, name="diagram_outline")
        supports = mg.support_points(model)
        if len(supports):
            self.add_points(supports, color=style.V_SUPPORT,
                            render_points_as_spheres=True, point_size=18,
                            name="supports")
        self._draw_grid()
        self._frame(model)
        return vmax

    def show_design(self, model, dcrs) -> None:
        """Colour each member by its AISC demand/capacity ratio and label it
        with the value (grey / '—' = no steel shape assigned)."""
        self._replay = partial(self.show_design, model, dcrs)
        self.clear()
        span = mg.model_span(model)
        radius = max(span * 0.004, 1e-3)
        lab_pts, lab_txt = [], []
        for tag, e in model.elements.items():
            line = mg.element_line(e)
            if line is None:
                continue
            dcr = dcrs.get(tag)
            self.add_mesh(line.tube(radius=radius), color=_dcr_color(dcr),
                          name=f"member_{tag}")
            c = e.node_coords()
            lab_pts.append((np.asarray(mg.to_xyz(c[0]))
                            + np.asarray(mg.to_xyz(c[1]))) / 2.0)
            lab_txt.append(f"{dcr:.2f}" if dcr is not None else "—")
        _tags, pts, _index = mg.node_points(model)
        if len(pts):
            self.add_points(pts, color=style.V_NODE, render_points_as_spheres=True,
                            point_size=10, name="nodes")
        supports = mg.support_points(model)
        if len(supports):
            self.add_points(supports, color=style.V_SUPPORT,
                            render_points_as_spheres=True, point_size=20,
                            name="supports")
        if lab_pts:
            try:
                self.add_point_labels(np.array(lab_pts), lab_txt, font_size=12,
                                      text_color=style.TEXT, shape_opacity=0.15,
                                      always_visible=True, name="dcr_labels")
            except Exception:
                pass
        try:
            self.add_legend(
                [["DCR <= 0.50", "#2f9e44"], ["0.50 - 0.90", "#f59e0b"],
                 ["0.90 - 1.00", "#ea580c"], ["> 1.00  fail", "#dc2626"],
                 ["no section", "#9aa0a6"]],
                bcolor=style.PANEL, size=(0.24, 0.26), loc="upper right")
        except Exception:
            pass
        self._draw_grid()
        self._frame(model)

    def _frame(self, model) -> None:
        self.view_xy() if getattr(model, "ndm", 3) == 2 else self.view_isometric()
        self.reset_camera()

    def fit(self) -> None:
        self.reset_camera()

    def set_view(self, name: str) -> None:
        """Standard named views (isometric / top / bottom / front / back /
        left / right), then fit."""
        views = {
            "iso": self.view_isometric,
            "top": self.view_xy,
            "bottom": lambda: self.view_xy(negative=True),
            "front": self.view_xz,
            "back": lambda: self.view_xz(negative=True),
            "left": lambda: self.view_yz(negative=True),
            "right": self.view_yz,
        }
        fn = views.get(name)
        if fn is None:
            return
        try:
            fn()
        except Exception:
            self.view_isometric()
        self.reset_camera()
        self.render()


def _snap(v, grid: float = 0.5) -> float:
    return round(v / grid) * grid


def _dcr_color(dcr) -> str:
    if dcr is None:
        return "#9aa0a6"          # grey — no design section
    if dcr <= 0.5:
        return "#2f9e44"          # green (distinct from the base member green)
    if dcr <= 0.9:
        return "#f59e0b"          # amber
    if dcr <= 1.0:
        return "#ea580c"          # orange
    return "#dc2626"              # red — over capacity
