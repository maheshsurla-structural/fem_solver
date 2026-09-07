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
from PySide6.QtGui import QBrush, QColor, QPainter, QPainterPath, QPen
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


def _seg_dist(pz, py, z1, y1, z2, y2) -> float:
    """Distance from point (pz,py) to the segment (z1,y1)-(z2,y2)."""
    dz, dy = z2 - z1, y2 - y1
    L2 = dz * dz + dy * dy
    if L2 <= 1e-18:
        return math.hypot(pz - z1, py - y1)
    t = max(0.0, min(1.0, ((pz - z1) * dz + (py - y1) * dy) / L2))
    return math.hypot(pz - (z1 + t * dz), py - (y1 + t * dy))


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
        # dims + model bounds (metres) for placing the drag-resize handles
        self._dims = {k: float(getattr(spec, k, 0.0))
                      for k in ("b", "h", "D", "leg", "thick", "t_f", "t_w",
                                "wall_t")}
        if self._ext:
            zs = [z for z, _ in self._ext]
            ys = [y for _, y in self._ext]
            self._mb = (min(zs), min(ys), max(zs), max(ys))
        else:
            self._mb = (-0.2, -0.3, 0.2, 0.3)
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

    def _dim_handle(self, z, y, axis, key, mode, v0, vmax=None) -> None:
        """A draggable dimension handle. ``mode``: 'sym' (edge tracks cursor,
        symmetric dim = 2·|cursor|), 'grow_pos' (dim = v0 + Δ, drag outward
        grows), 'grow_neg' (dim = v0 − Δ, drag inward grows)."""
        self._handle(z, y, "#f59f00",
                     {"t": "dim", "axis": axis, "key": key, "mode": mode,
                      "v0": v0, "vmax": vmax})

    def _add_dim_handles(self) -> None:
        minz, miny, maxz, maxy = self._mb
        d = self._dims
        k = self._kind
        if k == "Rectangular":
            self._dim_handle(maxz, 0.0, "z", "b", "sym", d["b"])
            self._dim_handle(0.0, maxy, "y", "h", "sym", d["h"])
        elif k == "Circular":
            self._dim_handle(maxz, 0.0, "z", "D", "sym", d["D"])
            self._dim_handle(0.0, maxy, "y", "D", "sym", d["D"])
        elif k == "Hollow box":
            self._dim_handle(maxz, 0.0, "z", "b", "sym", d["b"])
            self._dim_handle(0.0, maxy, "y", "h", "sym", d["h"])
            self._dim_handle(maxz - d["wall_t"], 0.0, "z", "wall_t", "grow_neg",
                             d["wall_t"], vmax=min(d["b"], d["h"]) / 2 - _MIN_DIM)
        elif k == "T-shape":
            tf, tw = d["t_f"], d["t_w"]
            self._dim_handle(maxz, (maxy + (maxy - tf)) / 2, "z", "b", "sym",
                             d["b"])
            self._dim_handle(0.0, miny, "y", "h", "grow_neg", d["h"])
            self._dim_handle(0.0, maxy - tf, "y", "t_f", "grow_neg", tf,
                             vmax=d["h"] - _MIN_DIM)
            self._dim_handle(tw / 2, (miny + (maxy - tf)) / 2, "z", "t_w", "sym",
                             tw, vmax=d["b"] - _MIN_DIM)
        elif k == "L-shape":
            leg, th = d["leg"], d["thick"]
            self._dim_handle(maxz, miny + th / 2, "z", "leg", "grow_pos", leg)
            self._dim_handle(minz + th / 2, maxy, "y", "leg", "grow_pos", leg)
            self._dim_handle(minz + th, miny + th / 2, "z", "thick", "grow_pos",
                             th, vmax=leg - _MIN_DIM)
        elif k == "PSC girder":
            self._dim_handle(maxz, 0.0, "z", "b", "sym", d["b"])
            self._dim_handle(0.0, maxy, "y", "h", "grow_pos", d["h"])

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
            self._insert_vertex(z, y)
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
            c = z if d["axis"] == "z" else y
            c0 = d["sz"] if d["axis"] == "z" else d["sy"]
            mode, v0 = d["mode"], d["v0"]
            if mode == "sym":
                val = 2.0 * abs(c)
            elif mode == "grow_pos":
                val = v0 + (c - c0)
            else:                                       # grow_neg
                val = v0 + (c0 - c)
            val = max(val, _MIN_DIM)
            if d.get("vmax"):
                val = min(val, d["vmax"])
            self.dimChanged.emit(d["key"], val)
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

    def _insert_vertex(self, z, y) -> None:
        """Insert a new vertex on the outline edge nearest the click (so the
        polygon keeps its shape), or append when there's no edge yet."""
        pts = self._outline
        n = len(pts)
        if n < 2:
            pts.append((z, y))
            return
        best_i, best_d = n - 1, float("inf")
        for i in range(n):
            z1, y1 = pts[i]
            z2, y2 = pts[(i + 1) % n]
            dd = _seg_dist(z, y, z1, y1, z2, y2)
            if dd < best_d:
                best_d, best_i = dd, i
        pts.insert(best_i + 1, (z, y))

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
