"""The 3-D model viewport — a PyVista/VTK widget that renders a Model.

Subclasses ``pyvistaqt.QtInteractor`` (itself a Qt widget) so it drops
straight into the main window as the central widget. ``set_model`` is the
one entry point: clear, draw members / nodes / supports, frame the camera.
"""
from __future__ import annotations

import numpy as np
from pyvistaqt import QtInteractor

import model_geometry as mg

MEMBER_COLOR = "#3b6d11"      # green
NODE_COLOR = "#185fa5"        # blue
SUPPORT_COLOR = "#a32d2d"     # red
REFERENCE_COLOR = "#c9c9c9"   # grey (undeformed ghost)
DEFORMED_COLOR = "#d85a30"    # coral
DEFORMED_NODE = "#993c1d"     # dark coral
DIAGRAM_COLOR = {"N": "#1d4ed8", "V": "#0f766e", "M": "#b45309"}
SELECTION_COLOR = "#f59e0b"   # amber — current selection highlight


class ModelView(QtInteractor):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.set_background("white")
        self.enable_parallel_projection()   # orthographic — CAD-style elevations
        self._model = None
        self._pick_cb = None
        self._add_node_cb = None
        self._add_member_cb = None
        self._mode = "select"
        self._member_start = None
        self._snap_on = True
        self._snap_grid = 0.5
        self._highlight = None
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

    def set_snap(self, on: bool, grid: float) -> None:
        self._snap_on = bool(on)
        self._snap_grid = max(float(grid), 1e-3)

    def set_mode(self, mode: str) -> None:
        self._mode = mode
        self._member_start = None
        self.remove_actor("groundplane", render=False)
        if mode == "draw_node":
            self._add_ground_plane()
        self.render()

    def _add_ground_plane(self) -> None:
        import pyvista as pv
        if self._model is not None and len(self._model.nodes):
            _t, pts, _i = mg.node_points(self._model)
            lo, hi = pts.min(axis=0), pts.max(axis=0)
            cx, cy = (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2
            size = float(max(hi[0] - lo[0], hi[1] - lo[1])) * 2.0 + 10.0
        else:
            cx = cy = 0.0
            size = 20.0
        plane = pv.Plane(center=(cx, cy, 0.0), direction=(0.0, 0.0, 1.0),
                         i_size=size, j_size=size)
        self.add_mesh(plane, color="#9ec5ff", opacity=0.12, name="groundplane",
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
                if self._snap_on:
                    x = _snap(p[0], self._snap_grid)
                    y = _snap(p[1], self._snap_grid)
                else:
                    x, y = round(float(p[0]), 3), round(float(p[1]), 3)
                self._add_node_cb(x, y, 0.0)
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
                self.highlight("node", sel[1])
            elif sel[1] != self._member_start:
                if self._add_member_cb is not None:
                    self._add_member_cb(self._member_start, sel[1])
                self._member_start = None
            return
        if self._pick_cb is not None:
            self._pick_cb(*sel)

    def highlight(self, kind, ident) -> None:
        self._highlight = (kind, ident)
        self._draw_highlight()

    def clear_highlight(self) -> None:
        self._highlight = None
        self.remove_actor("selection", render=True)

    def _draw_highlight(self) -> None:
        self.remove_actor("selection", render=False)
        if self._highlight is None or self._model is None:
            return
        kind, ident = self._highlight
        span = mg.model_span(self._model)
        if kind == "node" and ident in self._model.nodes:
            pt = np.asarray(mg.to_xyz(self._model.nodes[ident].coords))
            self.add_points(pt.reshape(1, 3), color=SELECTION_COLOR,
                            render_points_as_spheres=True, point_size=22,
                            name="selection")
        elif kind == "member" and ident in self._model.elements:
            line = mg.element_line(self._model.element(ident))
            if line is not None:
                self.add_mesh(line.tube(radius=max(span * 0.006, 2e-3)),
                              color=SELECTION_COLOR, name="selection")
        self.render()

    def set_model(self, model) -> None:
        self._model = model
        self.clear()

        span = mg.model_span(model)
        mesh = mg.members_mesh(model)
        if mesh is not None:
            self.add_mesh(mesh.tube(radius=max(span * 0.004, 1e-3)),
                          color=MEMBER_COLOR, name="members")

        _tags, pts, _index = mg.node_points(model)
        if len(pts):
            self.add_points(pts, color=NODE_COLOR, render_points_as_spheres=True,
                            point_size=14, name="nodes")

        supports = mg.support_points(model)
        if len(supports):
            self.add_points(supports, color=SUPPORT_COLOR,
                            render_points_as_spheres=True, point_size=22,
                            name="supports")

        self.show_grid()
        self._frame(model)
        self._draw_highlight()
        if self._mode == "draw_node":
            self._add_ground_plane()

    def show_deformed(self, model, scale: float) -> None:
        """Draw the deformed shape (coral) over a grey ghost of the model."""
        self._model = model
        self.clear()
        span = mg.model_span(model)

        ref = mg.members_mesh(model)
        if ref is not None:
            self.add_mesh(ref.tube(radius=max(span * 0.0025, 1e-3)),
                          color=REFERENCE_COLOR, name="reference")

        mesh = mg.deformed_members_mesh(model, scale)
        if mesh is not None:
            self.add_mesh(mesh.tube(radius=max(span * 0.004, 1e-3)),
                          color=DEFORMED_COLOR, name="deformed")

        pts = mg.deformed_points(model, scale)
        if len(pts):
            self.add_points(pts, color=DEFORMED_NODE,
                            render_points_as_spheres=True, point_size=12,
                            name="deformed_nodes")

        supports = mg.support_points(model)
        if len(supports):
            self.add_points(supports, color=SUPPORT_COLOR,
                            render_points_as_spheres=True, point_size=20,
                            name="supports")
        self.show_grid()
        self._frame(model)

    def show_diagram(self, model, kind: str):
        """Draw the N / V / M diagram over grey members; return max |value|."""
        self.clear()
        span = mg.model_span(model)
        ref = mg.members_mesh(model)
        if ref is not None:
            self.add_mesh(ref.tube(radius=max(span * 0.003, 1e-3)),
                          color="#8a8a8a", name="members")
        vmax = mg.diagram_extreme(model, kind)
        scale = (0.16 * span / vmax) if vmax > 0 else 0.0
        fill, outline = mg.diagram_meshes(model, kind, scale)
        color = DIAGRAM_COLOR.get(kind, "#1d4ed8")
        if fill is not None:
            self.add_mesh(fill, color=color, opacity=0.35, name="diagram_fill")
        if outline is not None:
            self.add_mesh(outline, color=color, line_width=2, name="diagram_outline")
        supports = mg.support_points(model)
        if len(supports):
            self.add_points(supports, color=SUPPORT_COLOR,
                            render_points_as_spheres=True, point_size=18,
                            name="supports")
        self.show_grid()
        self._frame(model)
        return vmax

    def show_design(self, model, dcrs) -> None:
        """Colour each member by its AISC demand/capacity ratio and label it
        with the value (grey / '—' = no steel shape assigned)."""
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
            self.add_points(pts, color=NODE_COLOR, render_points_as_spheres=True,
                            point_size=10, name="nodes")
        supports = mg.support_points(model)
        if len(supports):
            self.add_points(supports, color=SUPPORT_COLOR,
                            render_points_as_spheres=True, point_size=20,
                            name="supports")
        if lab_pts:
            try:
                self.add_point_labels(np.array(lab_pts), lab_txt, font_size=12,
                                      text_color="black", shape_opacity=0.15,
                                      always_visible=True, name="dcr_labels")
            except Exception:
                pass
        try:
            self.add_legend(
                [["DCR <= 0.50", "#2f9e44"], ["0.50 - 0.90", "#f59e0b"],
                 ["0.90 - 1.00", "#ea580c"], ["> 1.00  fail", "#dc2626"],
                 ["no section", "#9aa0a6"]],
                bcolor="white", size=(0.24, 0.26), loc="upper right")
        except Exception:
            pass
        self.show_grid()
        self._frame(model)

    def _frame(self, model) -> None:
        self.view_xy() if getattr(model, "ndm", 3) == 2 else self.view_isometric()
        self.reset_camera()

    def fit(self) -> None:
        self.reset_camera()


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
