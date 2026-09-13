"""A CAD-style navigation cube — a small orientation gizmo overlaid on the
top-right of the 3-D viewport.

It draws a live cube that **rotates in sync with the camera**, so its labelled
faces always show the model's true orientation. Interaction:

* click a visible **face**  → snap to that orthographic view (top / bottom /
  front / back / left / right) and fit the model to the screen;
* click a **corner**        → isometric view + fit;
* click a **chevron arrow** → orbit the camera 90° (to bring a hidden face
  round to the front).

The cube is decoupled from VTK: it asks its host view for the camera basis
(``camera_basis``) and calls back into it (``set_view`` / ``orbit``). The pure
geometry + hit-testing helpers at module scope are unit-tested headlessly.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QWidget

import style

# --- cube topology ----------------------------------------------------------
# Every face pairs an outward world-axis normal with the ``set_view`` name that
# points the camera *at that face*. Because the cube tracks the real camera,
# this pairing is what makes "click TOP → the top view" self-consistent (a
# regression test pins each pairing against ``ModelView.set_view``).
_S = 1.0
_FACES = (
    ("right",  (1.0, 0.0, 0.0),
     ((1, -1, -1), (1, 1, -1), (1, 1, 1), (1, -1, 1))),
    ("left",   (-1.0, 0.0, 0.0),
     ((-1, 1, -1), (-1, -1, -1), (-1, -1, 1), (-1, 1, 1))),
    ("back",   (0.0, 1.0, 0.0),
     ((1, 1, -1), (-1, 1, -1), (-1, 1, 1), (1, 1, 1))),
    ("front",  (0.0, -1.0, 0.0),
     ((-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1))),
    ("top",    (0.0, 0.0, 1.0),
     ((-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1))),
    ("bottom", (0.0, 0.0, -1.0),
     ((-1, 1, -1), (1, 1, -1), (1, -1, -1), (-1, -1, -1))),
)
_LABELS = {"front": "FRONT", "back": "BACK", "left": "LEFT", "right": "RIGHT",
           "top": "TOP", "bottom": "BOT"}
_CORNERS = tuple((sx, sy, sz) for sx in (-1, 1) for sy in (-1, 1)
                 for sz in (-1, 1))


def _norm(v):
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


def camera_basis_from(position, focal_point, up):
    """Screen basis (right, up, forward) as unit world vectors, from a camera's
    position / focal point / view-up. ``forward`` points into the scene (away
    from the eye). Returns ``None`` if the camera is degenerate."""
    pos = np.asarray(position, dtype=float)
    fp = np.asarray(focal_point, dtype=float)
    fwd = fp - pos
    if float(np.linalg.norm(fwd)) < 1e-9:
        return None
    f = _norm(fwd)
    r = _norm(np.cross(f, _norm(up)))
    if float(np.linalg.norm(r)) < 1e-9:
        return None
    u = _norm(np.cross(r, f))
    return r, u, f


def project(vec, basis):
    """A world vector → cube-local screen coords (sx, sy up-positive, depth)."""
    r, u, f = basis
    v = np.asarray(vec, dtype=float)
    return float(v @ r), float(v @ u), float(v @ f)


def visible_faces(basis):
    """Names of the faces turned toward the camera (outward normal · forward
    < 0), front-most first — the only faces drawn and hit-tested."""
    out = []
    _r, _u, f = basis
    for name, normal, _c in _FACES:
        d = float(np.asarray(normal) @ f)
        if d < -1e-6:
            out.append((d, name))
    out.sort()                       # most negative (most head-on) first
    return [name for _d, name in out]


def _face_polygon(corners, basis, center, radius):
    pts = []
    for c in corners:
        sx, sy, _d = project(tuple(_S * k for k in c), basis)
        pts.append(QPointF(center[0] + sx * radius, center[1] - sy * radius))
    return pts


def hit_test(local_x, local_y, basis, center, radius):
    """What a click at cube-local (x, y) pixels lands on: a face name, ``"iso"``
    for a near corner, or ``None``. Corners take priority (small hotspots)."""
    # near-side corners first — small circular hotspots → isometric
    _r, _u, f = basis
    corner_r = radius * 0.34
    for c in _CORNERS:
        if float(np.asarray(c, dtype=float) @ f) >= 0.0:
            continue                 # far corner, hidden
        sx, sy, _d = project(tuple(_S * k for k in c), basis)
        px = center[0] + sx * radius
        py = center[1] - sy * radius
        if (local_x - px) ** 2 + (local_y - py) ** 2 <= corner_r ** 2:
            return "iso"
    faces = {n: c for n, _nm, c in _FACES}
    for name in visible_faces(basis):
        poly = QPolygonF(_face_polygon(faces[name], basis, center, radius))
        if poly.containsPoint(QPointF(local_x, local_y), Qt.FillRule.OddEvenFill):
            return name
    return None


def _shade(hexcolor: str, factor: float) -> QColor:
    """Lighten (>1) / darken (<1) a colour for cheap face shading."""
    c = QColor(hexcolor)
    f = max(0.0, factor)
    return QColor(min(255, int(c.red() * f)), min(255, int(c.green() * f)),
                  min(255, int(c.blue() * f)), c.alpha())


class NavCube(QWidget):
    """The orientation-cube overlay. Give it the host :class:`ModelView`; it
    reads ``view.camera_basis()`` and calls ``view.set_view`` / ``view.orbit``.
    Sizes itself; the host positions it (top-right) in ``resizeEvent``."""

    SIZE = 132

    def __init__(self, view, parent=None):
        super().__init__(parent or view)
        self._view = view
        self.setFixedSize(self.SIZE, self.SIZE)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setToolTip("Navigation cube — click a face, corner or arrow to "
                        "change the view")
        self._hover = None                     # hovered region name / "iso"
        self._last_basis = None
        # geometry: cube centred, leaving a margin for the orbit chevrons
        self._radius = self.SIZE * 0.28
        self._center = (self.SIZE / 2.0, self.SIZE / 2.0)
        self._arrows = self._build_arrows()
        # poll the camera so the cube stays in sync while the user mouse-orbits
        self._timer = QTimer(self)
        self._timer.setInterval(80)
        self._timer.timeout.connect(self._sync)
        self._timer.start()

    # -- camera sync ---------------------------------------------------------
    def _basis(self):
        try:
            b = self._view.camera_basis()
        except Exception:
            b = None
        if b is None:                          # fallback: a pleasant iso pose
            b = camera_basis_from((1, -1, 0.8), (0, 0, 0), (0, 0, 1))
        return b

    def _sync(self) -> None:
        if not self.isVisible():
            return
        b = self._basis()
        if b is None:
            return
        if (self._last_basis is None
                or not np.allclose(np.vstack(b), np.vstack(self._last_basis),
                                   atol=1e-4)):
            self._last_basis = b
            self.update()

    def apply_theme(self) -> None:
        self.update()

    # -- interaction ---------------------------------------------------------
    def _region_at(self, pos):
        cx, cy = pos.x(), pos.y()
        for name, tri in self._arrows.items():
            if tri.containsPoint(QPointF(cx, cy), Qt.FillRule.OddEvenFill):
                return name
        return hit_test(cx, cy, self._basis(), self._center, self._radius)

    def mouseMoveEvent(self, ev):
        hit = self._region_at(ev.position())
        if hit != self._hover:
            self._hover = hit
            self.update()

    def leaveEvent(self, ev):
        if self._hover is not None:
            self._hover = None
            self.update()

    def mousePressEvent(self, ev):
        if ev.button() != Qt.MouseButton.LeftButton:
            return
        hit = self._region_at(ev.position())
        if hit is None:
            return
        if hit in ("az_left", "az_right", "el_up", "el_down"):
            steps = {"az_left": (-90.0, 0.0), "az_right": (90.0, 0.0),
                     "el_up": (0.0, 90.0), "el_down": (0.0, -90.0)}[hit]
            try:
                self._view.orbit(*steps)
            except Exception:
                pass
        else:                                  # a face name or "iso"
            try:
                self._view.set_view(hit)
            except Exception:
                pass
        self._sync()

    # -- painting ------------------------------------------------------------
    def _build_arrows(self):
        """Four small orbit chevrons, one at each mid-edge of the widget."""
        w = float(self.SIZE)
        m, s = 3.0, 9.0                        # edge margin, half-width
        mid = w / 2.0
        tri = {
            "el_up":    [(mid, m), (mid - s, m + s * 1.3), (mid + s, m + s * 1.3)],
            "el_down":  [(mid, w - m), (mid - s, w - m - s * 1.3),
                         (mid + s, w - m - s * 1.3)],
            "az_left":  [(m, mid), (m + s * 1.3, mid - s), (m + s * 1.3, mid + s)],
            "az_right": [(w - m, mid), (w - m - s * 1.3, mid - s),
                         (w - m - s * 1.3, mid + s)],
        }
        return {k: QPolygonF([QPointF(x, y) for x, y in v])
                for k, v in tri.items()}

    def paintEvent(self, ev):
        basis = self._basis()
        if basis is None:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self._paint_arrows(p)
        self._paint_cube(p, basis)
        p.end()

    def _paint_arrows(self, p: QPainter) -> None:
        for name, tri in self._arrows.items():
            hot = self._hover == name
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QColor(style.ACCENT if hot else style.MUTED))
            p.drawPolygon(tri)

    def _paint_cube(self, p: QPainter, basis) -> None:
        _r, u, _f = basis
        faces = {n: (nm, c) for n, nm, c in _FACES}
        edge = QColor(style.BORDER_STRONG)
        for name in reversed(visible_faces(basis)):    # far → near
            normal, corners = faces[name]
            poly = QPolygonF(_face_polygon(corners, basis, self._center,
                                           self._radius))
            hot = self._hover == name
            if hot:
                fill = QColor(style.ACCENT_SOFT)
                edge_c = QColor(style.ACCENT)
                txt = QColor(style.ACCENT)
            else:
                # cheap top-lit shading: faces pointing up read brighter
                vert = float(np.asarray(normal) @ u)
                fill = _shade(style.PANEL, 1.0 + 0.10 * vert)
                edge_c = edge
                txt = QColor(style.MUTED)
            p.setBrush(fill)
            p.setPen(QPen(edge_c, 1.4))
            p.drawPolygon(poly)
            self._paint_label(p, name, corners, basis, txt)

    def _paint_label(self, p, name, corners, basis, color) -> None:
        cx = sum(c[0] for c in corners) / 4.0
        cy = sum(c[1] for c in corners) / 4.0
        cz = sum(c[2] for c in corners) / 4.0
        sx, sy, _d = project((_S * cx, _S * cy, _S * cz), basis)
        px = self._center[0] + sx * self._radius
        py = self._center[1] - sy * self._radius
        font = QFont()
        font.setFamilies(["Segoe UI", "Arial", "sans-serif"])
        font.setPointSizeF(7.5)
        font.setBold(True)
        p.setFont(font)
        p.setPen(color)
        box = QRectF(px - 22, py - 8, 44, 16)  # centred on the face centroid
        p.drawText(box, Qt.AlignmentFlag.AlignCenter, _LABELS.get(name, name))
