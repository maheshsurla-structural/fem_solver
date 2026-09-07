"""Interactive cross-section canvas for the desktop Section Designer.

A ``QGraphicsView`` that renders the current section (outline + holes + rebars)
over a millimetre grid with the Y/Z reference axes, and lets the user *edit it
graphically*:

* Built-in parametric shapes (Rectangular / Circular / T / L / Hollow box / PSC)
  get **drag-resize handles** on their right and top edges — dragging updates the
  primary width/height dimension and the number fields stay in sync.
* **Custom** sections are fully drawable: drag any vertex, click to add a vertex,
  click to drop a rebar, right-click to delete a vertex/rebar.

The canvas is a *view over the Spec*: it never owns the model. It emits signals
(``dimChanged`` / ``outlineChanged`` / ``barsChanged`` / ``rebarAdded``) that the
window turns into Spec edits + a recompute, then calls :meth:`render_case` again.

Coordinate convention (matches the rest of the GUI): **Y = horizontal (engine z),
Z = vertical (engine y)**. Model coordinates are metres; a model point ``(z, y)``
is drawn at scene point ``(z, -y)`` so +y is up on screen.
"""
from __future__ import annotations

import math

from PySide6.QtCore import Qt, QPointF, QRectF, Signal
from PySide6.QtGui import (QBrush, QColor, QPainter, QPainterPath, QPen,
                           QPolygonF)
from PySide6.QtWidgets import (QGraphicsEllipseItem, QGraphicsItem,
                               QGraphicsLineItem, QGraphicsPathItem,
                               QGraphicsScene, QGraphicsSimpleTextItem,
                               QGraphicsView)

# primary (width, height) dimension key per parametric kind — the bounding-box
# width equals the width dim and the bbox height the height dim, so a right/top
# edge drag maps straight onto these.
_PRIMARY_DIMS = {
    "Rectangular": ("b", "h"),
    "Circular":    ("D", "D"),
    "T-shape":     ("b", "h"),
    "L-shape":     ("leg", "leg"),
    "Hollow box":  ("b", "h"),
    "PSC girder":  ("b", "h"),
}
_MIN_DIM = 0.02          # metres — don't let a drag collapse a dimension


class SectionCanvas(QGraphicsView):
    dimChanged = Signal(str, float)      # (dim_key, value_metres)
    outlineChanged = Signal(tuple)       # ((z_m, y_m), ...)
    barsChanged = Signal(tuple)          # ((z_m, y_m, dia_m), ...)
    rebarAdded = Signal(float, float)    # z_m, y_m  (non-custom kinds)
    cursorMoved = Signal(object)         # (z_mm, y_mm) or None

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setMouseTracking(True)
        self.setBackgroundBrush(QBrush(QColor("#ffffff")))
        self.setMinimumHeight(300)
        self.setTransformationAnchor(
            QGraphicsView.ViewportAnchor.AnchorUnderMouse)

        self._mode = "select"            # select | add_vertex | add_bar
        self._kind = "Rectangular"
        self._ext: list = []             # exterior ring (metres) from the case
        self._holes: list = []           # hole rings (metres)
        self._render_bars: list = []     # (z, y, r) metres — every drawn bar
        self._outline: list = []         # custom vertices (metres), editable
        self._bars: list = []            # custom bars (z, y, dia) metres
        self._default_dia = 0.020        # for click-placed custom bars
        self._grid = 0.05                # 50 mm grid
        self._drag = None                # active drag descriptor
        self._pan = None                 # last pan point (device px)
        self._fitted_kind = None         # refit when the shape kind changes

    # -------------------------------------------------- public API
    def set_mode(self, mode: str) -> None:
        self._mode = mode
        self.setCursor(Qt.CursorShape.CrossCursor if mode != "select"
                       else Qt.CursorShape.ArrowCursor)

    def set_default_bar(self, dia_m: float) -> None:
        if dia_m and dia_m > 0:
            self._default_dia = dia_m

    def render_case(self, case, spec) -> None:
        """Rebuild the scene from the current case + spec (never emits)."""
        self._kind = spec.kind
        self._outline = [(float(z), float(y)) for (z, y)
                         in (spec.custom_outline or ())]
        self._bars = [(float(z), float(y), float(d)) for (z, y, d)
                      in (spec.custom_bars or ())]
        try:
            poly = case.section.geometry.polygon
            ext = list(poly.exterior.coords)
            self._ext = [(float(z), float(y)) for (z, y) in ext[:-1]]
            self._holes = [[(float(z), float(y)) for (z, y) in list(r.coords)[:-1]]
                           for r in poly.interiors]
        except Exception:                              # noqa: BLE001
            self._ext, self._holes = [], []
        # a fresh Custom section with no stored vertices is still editable —
        # seed the editable outline from the drawn exterior.
        if spec.kind == "Custom" and not self._outline:
            self._outline = list(self._ext)
        bars = (case.section.reinforcement.bars
                if case.section.reinforcement else [])
        self._render_bars = [(float(b.z), float(b.y),
                              max(math.sqrt(float(b.area) / math.pi), 0.004))
                             for b in bars]
        self._rebuild_scene()
        if self._fitted_kind != spec.kind:
            self.fit()
            self._fitted_kind = spec.kind

    def fit(self) -> None:
        r = self._content_rect()
        if r.isValid():
            m = max(r.width(), r.height()) * 0.12 + 0.02
            self.fitInView(r.adjusted(-m, -m, m, m),
                           Qt.AspectRatioMode.KeepAspectRatio)

    # -------------------------------------------------- scene building
    def _content_rect(self) -> QRectF:
        pts = [(z, -y) for (z, y) in self._ext] or [(-0.2, -0.3), (0.2, 0.3)]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return QRectF(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))

    @staticmethod
    def _ring(path: QPainterPath, pts) -> None:
        if not pts:
            return
        path.moveTo(pts[0][0], -pts[0][1])
        for (z, y) in pts[1:]:
            path.lineTo(z, -y)
        path.closeSubpath()

    def _handle(self, z, y, color, data, r=5):
        h = QGraphicsEllipseItem(-r, -r, 2 * r, 2 * r)
        h.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        h.setPos(z, -y)
        h.setBrush(QBrush(QColor(color)))
        h.setPen(QPen(QColor("#20303c"), 1.2))
        h.setZValue(20)
        h.setData(0, data)
        self._scene.addItem(h)

    def _label(self, z, y, text, color="#333"):
        t = QGraphicsSimpleTextItem(text)
        t.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        t.setBrush(QBrush(QColor(color)))
        f = t.font()
        f.setPointSize(10)
        f.setBold(True)
        t.setFont(f)
        t.setPos(z, -y)
        t.setZValue(15)
        self._scene.addItem(t)

    def _rebuild_scene(self) -> None:
        self._scene.clear()
        # section outline (+ holes) as an even-odd path
        path = QPainterPath()
        self._ring(path, self._ext)
        for h in self._holes:
            self._ring(path, h)
        path.setFillRule(Qt.FillRule.OddEvenFill)
        body = QGraphicsPathItem(path)
        body.setBrush(QBrush(QColor("#cfe8ff")))
        body.setPen(QPen(QColor("#1f4f73"), 0, Qt.PenStyle.SolidLine))
        pen = body.pen()
        pen.setCosmetic(True)
        pen.setWidthF(1.5)
        body.setPen(pen)
        body.setZValue(1)
        self._scene.addItem(body)

        # reference axes (Y horizontal = engine z, Z vertical = engine y)
        self._draw_axes()

        # rebar dots
        for (z, y, r) in self._render_bars:
            dot = QGraphicsEllipseItem(z - r, -y - r, 2 * r, 2 * r)
            dot.setBrush(QBrush(QColor("#b03030")))
            dp = QPen(QColor("#401010"), 0)
            dp.setCosmetic(True)
            dp.setWidthF(0.6)
            dot.setPen(dp)
            dot.setZValue(5)
            self._scene.addItem(dot)

        # interactive handles
        if self._kind == "Custom":
            for i, (z, y) in enumerate(self._outline):
                self._handle(z, y, "#3d7", {"t": "vertex", "i": i})
            for i, (z, y, _d) in enumerate(self._bars):
                self._handle(z, y, "#e8b", {"t": "bar", "i": i}, r=4)
        elif self._kind in _PRIMARY_DIMS:
            self._add_dim_handles()

    def _draw_axes(self) -> None:
        r = self._content_rect()
        if not r.isValid():
            return
        mx = max(r.width(), r.height()) * 0.1 + 0.03
        # horizontal Y axis (model y = 0 -> scene y = 0)
        ax = QGraphicsLineItem(r.left() - mx, 0.0, r.right() + mx, 0.0)
        ay = QGraphicsLineItem(0.0, r.top() - mx, 0.0, r.bottom() + mx)
        for a in (ax, ay):
            p = QPen(QColor("#9aa7b2"), 0, Qt.PenStyle.DashLine)
            p.setCosmetic(True)
            a.setPen(p)
            a.setZValue(0)
            self._scene.addItem(a)
        # labels at the positive ends: +Y right, +Z up (scene top = -y)
        self._label(r.right() + mx, 0.0, "Y", "#5a6b7b")
        self._label(0.0, -(r.top() - mx) if r.top() < 0 else r.bottom() + mx,
                    "Z", "#5a6b7b")

    def _add_dim_handles(self) -> None:
        r = self._content_rect()          # scene rect (x=z, y=-y_model)
        if not r.isValid():
            return
        wkey, hkey = _PRIMARY_DIMS[self._kind]
        width, height = r.width(), r.height()
        # right-edge handle (controls width), placed at mid-height
        rz, ry = r.right(), -(r.top() + r.height() / 2.0)
        self._handle(rz, ry, "#f90",
                     {"t": "dim", "axis": "z", "key": wkey, "w0": width,
                      "h0": height, "edge": r.right()})
        # top-edge handle (controls height), placed at mid-width
        tz, ty = r.left() + r.width() / 2.0, -r.top()
        self._handle(tz, ty, "#f90",
                     {"t": "dim", "axis": "y", "key": hkey, "w0": width,
                      "h0": height, "edge": -r.top()})

    # -------------------------------------------------- interaction
    def _model_at(self, view_pos) -> tuple:
        s = self.mapToScene(view_pos)
        return (s.x(), -s.y())

    def _handle_at(self, view_pos):
        for it in self.items(view_pos):
            d = it.data(0)
            if isinstance(d, dict):
                return d
        return None

    def wheelEvent(self, e):
        f = 1.15 if e.angleDelta().y() > 0 else 1.0 / 1.15
        self.scale(f, f)

    def mousePressEvent(self, e):
        vp = e.position().toPoint()
        if e.button() == Qt.MouseButton.RightButton:
            d = self._handle_at(vp)
            if d and d.get("t") in ("vertex", "bar"):
                self._delete(d)
            return
        if e.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(e)
        d = self._handle_at(vp)
        if d:
            z, y = self._model_at(vp)
            self._drag = dict(d, sz=z, sy=y)
            return
        if self._mode == "add_vertex" and self._kind == "Custom":
            z, y = self._model_at(vp)
            self._outline.append((z, y))
            self.outlineChanged.emit(tuple(self._outline))
            return
        if self._mode == "add_bar":
            z, y = self._model_at(vp)
            if self._kind == "Custom":
                self._bars.append((z, y, self._default_dia))
                self.barsChanged.emit(tuple(self._bars))
            else:
                self.rebarAdded.emit(z, y)
            return
        self._pan = vp                    # empty space -> pan

    def mouseMoveEvent(self, e):
        vp = e.position().toPoint()
        z, y = self._model_at(vp)
        self.cursorMoved.emit((z * 1e3, y * 1e3))
        if self._drag is not None:
            self._apply_drag(z, y)
            return
        if self._pan is not None:
            delta = vp - self._pan
            self._pan = vp
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y())
            return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        self._drag = None
        self._pan = None
        super().mouseReleaseEvent(e)

    def leaveEvent(self, e):
        self.cursorMoved.emit(None)
        super().leaveEvent(e)

    def _apply_drag(self, z, y) -> None:
        d = self._drag
        t = d["t"]
        if t == "dim":
            if d["axis"] == "z":
                new = d["w0"] + (z - d["sz"])
                self.dimChanged.emit(d["key"], max(new, _MIN_DIM))
            else:
                new = d["h0"] + (y - d["sy"])
                self.dimChanged.emit(d["key"], max(new, _MIN_DIM))
        elif t == "vertex":
            i = d["i"]
            if 0 <= i < len(self._outline):
                self._outline[i] = (z, y)
                self.outlineChanged.emit(tuple(self._outline))
        elif t == "bar":
            i = d["i"]
            if 0 <= i < len(self._bars):
                dia = self._bars[i][2]
                self._bars[i] = (z, y, dia)
                self.barsChanged.emit(tuple(self._bars))

    def _delete(self, d) -> None:
        i = d.get("i")
        if d["t"] == "vertex" and 0 <= i < len(self._outline):
            del self._outline[i]
            self.outlineChanged.emit(tuple(self._outline))
        elif d["t"] == "bar" and 0 <= i < len(self._bars):
            del self._bars[i]
            self.barsChanged.emit(tuple(self._bars))

    # -------------------------------------------------- grid
    def drawBackground(self, painter, rect) -> None:
        super().drawBackground(painter, rect)
        g = self._grid
        if g <= 0:
            return
        # adapt grid so lines never get denser than ~8 px apart
        px_per_m = self.transform().m11() or 1.0
        while g * px_per_m < 8 and g < 10:
            g *= 2
        pen = QPen(QColor("#eef2f6"))
        pen.setCosmetic(True)
        painter.setPen(pen)
        left = math.floor(rect.left() / g) * g
        top = math.floor(rect.top() / g) * g
        x = left
        while x <= rect.right():
            painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
            x += g
        yv = top
        while yv <= rect.bottom():
            painter.drawLine(QPointF(rect.left(), yv), QPointF(rect.right(), yv))
            yv += g
