"""Adapter: a femsolver ``Model`` -> PyVista geometry for the viewport.

Pure geometry, no Qt / no OpenGL — every function here can run headless
(used by the pre-flight check). Rendering (tubes, spheres, camera) lives in
``model_view``. Works for any Model: 2-D or 3-D, line or surface elements.
"""
from __future__ import annotations

import numpy as np
import pyvista as pv


def to_xyz(coords) -> tuple[float, float, float]:
    """Lift model coordinates to 3-D. A 2-D model (x, y) maps to (x, y, 0)."""
    c = np.asarray(coords, dtype=float).ravel()
    xyz = [0.0, 0.0, 0.0]
    xyz[: min(c.size, 3)] = c[:3].tolist()
    return (xyz[0], xyz[1], xyz[2])


def node_points(model):
    """Return (tags, points (N,3), tag->row index) for every node in order."""
    tags = list(model.nodes.keys())
    if tags:
        pts = np.array([to_xyz(model.nodes[t].coords) for t in tags], dtype=float)
    else:
        pts = np.zeros((0, 3), dtype=float)
    index = {t: i for i, t in enumerate(tags)}
    return tags, pts, index


def _member_segments(model):
    """Node-tag pairs to draw as lines: one per 2-node element; the closed
    boundary loop for elements with 3+ nodes (quad / shell)."""
    segs = []
    for e in model.elements.values():
        nt = e.node_tags
        if len(nt) == 2:
            segs.append((nt[0], nt[1]))
        elif len(nt) >= 3:
            n = len(nt)
            segs.extend((nt[i], nt[(i + 1) % n]) for i in range(n))
    return segs


def members_mesh(model):
    """PolyData of the element line-work, or ``None`` if there are none."""
    _tags, pts, index = node_points(model)
    segs = _member_segments(model)
    if not segs:
        return None
    cells = []
    for a, b in segs:
        cells += [2, index[a], index[b]]
    return pv.PolyData(pts, lines=np.asarray(cells, dtype=np.int64))


def support_points(model):
    """Coordinates of nodes carrying any single-point constraint (a support)."""
    pts = [to_xyz(n.coords) for n in model.nodes.values()
           if bool(np.any(n.fixity))]
    return np.asarray(pts, dtype=float) if pts else np.zeros((0, 3), dtype=float)


def model_span(model) -> float:
    """Diagonal of the model bounding box — for auto-scaling glyph sizes."""
    _tags, pts, _index = node_points(model)
    if len(pts) < 2:
        return 1.0
    return float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))) or 1.0


# --------------------------------------------------------------- deformed shape
# After an analysis, ``node.disp`` holds the DOF displacements; the first ``ndm``
# are the translations. ``scale`` exaggerates them for display.

def deformed_points(model, scale: float):
    tags = list(model.nodes.keys())
    if not tags:
        return np.zeros((0, 3), dtype=float)
    out = []
    for t in tags:
        n = model.nodes[t]
        c = np.asarray(n.coords, dtype=float).ravel()
        d = np.asarray(n.disp, dtype=float).ravel()[: c.size]   # translations
        out.append(to_xyz(c + scale * d))
    return np.asarray(out, dtype=float)


def deformed_members_mesh(model, scale: float):
    tags = list(model.nodes.keys())
    pts = deformed_points(model, scale)
    index = {t: i for i, t in enumerate(tags)}
    segs = _member_segments(model)
    if not segs:
        return None
    cells = []
    for a, b in segs:
        cells += [2, index[a], index[b]]
    return pv.PolyData(pts, lines=np.asarray(cells, dtype=np.int64))


def max_translation(model) -> float:
    """Largest nodal translation magnitude in the current results."""
    mx = 0.0
    for n in model.nodes.values():
        c = np.asarray(n.coords, dtype=float).ravel()
        d = np.asarray(n.disp, dtype=float).ravel()[: c.size]
        mx = max(mx, float(np.linalg.norm(d)))
    return mx
