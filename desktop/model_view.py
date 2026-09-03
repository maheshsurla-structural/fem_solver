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


class ModelView(QtInteractor):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.set_background("white")
        self.enable_parallel_projection()   # orthographic — CAD-style elevations
        self._model = None

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

    def _frame(self, model) -> None:
        self.view_xy() if getattr(model, "ndm", 3) == 2 else self.view_isometric()
        self.reset_camera()

    def fit(self) -> None:
        self.reset_camera()
