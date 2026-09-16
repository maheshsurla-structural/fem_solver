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


def areas_mesh(model):
    """PolyData of the filled faces of every surface (3+ node) element — the
    slab / wall / shell fill (slab plan S6). Returns ``None`` when the model
    has no surface elements. Triangles and quads are emitted as VTK polygon
    cells; the boundary edges are still drawn by :func:`members_mesh`."""
    _tags, pts, index = node_points(model)
    faces = []
    for e in model.elements.values():
        nt = e.node_tags
        if len(nt) in (3, 4) and all(t in index for t in nt):
            faces.append(len(nt))
            faces.extend(index[t] for t in nt)
    if not faces:
        return None
    return pv.PolyData(pts, faces=np.asarray(faces, dtype=np.int64))


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


# ------------------------------------------------------------- force diagrams
# From a beam element's recovered local end forces
# ``ef = [N1, V1, M1, N2, V2, M2]`` (validated on a cantilever + tension bar):
#   N(x) = ef[3]        axial, tension-positive, constant (no axial member load)
#   V(x) = ef[1]        shear, constant (no transverse member load)
#   M(x) = -ef[2] .. ef[5]   bending, linear, sagging-positive
# Exact for nodal-only loads (the current project scope).

def member_end_values(element, kind: str):
    """(value_i, value_j) for the N / V / M diagram of a 2-node beam, or None.
    2-D end forces are [N, Vy, Mz]*2 (signed, sagging +). 3-D are
    [N, Vy, Vz, T, My, Mz]*2 and the diagram uses the *resultant* transverse
    shear / bending magnitude, since either bending plane can be active."""
    ef = getattr(element, "end_forces_local", None)
    if ef is None or len(element.node_tags) != 2:
        return None
    n = len(ef)
    if n == 6:
        if kind == "N":
            return (float(ef[3]), float(ef[3]))
        if kind == "V":
            return (float(ef[1]), float(ef[1]))
        if kind == "M":
            return (float(-ef[2]), float(ef[5]))
    elif n == 12:
        if kind == "N":
            return (float(ef[6]), float(ef[6]))
        if kind == "V":                       # resultant transverse shear
            v = float(np.hypot(ef[1], ef[2]))
            return (v, v)
        if kind == "M":                       # resultant bending moment
            return (float(np.hypot(ef[4], ef[5])),
                    float(np.hypot(ef[10], ef[11])))
    return None


def _diagram_perp(element, i, j):
    """Unit direction to offset a diagram — the member's local +y from the
    element (3-D), else the in-plane perpendicular (2-D)."""
    if hasattr(element, "length_and_axes"):
        try:
            _L, _ex, ey, _ez = element.length_and_axes()
            ey = np.asarray(ey, dtype=float)
            nrm = np.linalg.norm(ey)
            if nrm > 0:
                return ey / nrm
        except Exception:
            pass
    d = j - i
    L = np.linalg.norm(d)
    dhat = d / L if L else np.array([1.0, 0.0, 0.0])
    return np.array([-dhat[1], dhat[0], 0.0])


def diagram_extreme(model, kind: str) -> float:
    """Largest |value| of the N / V / M diagram across all members."""
    mx = 0.0
    for e in model.elements.values():
        v = member_end_values(e, kind)
        if v:
            mx = max(mx, abs(v[0]), abs(v[1]))
    return mx


def diagram_meshes(model, kind: str, scale: float):
    """(fill PolyData, outline PolyData) for the diagram, each member's
    ordinate drawn perpendicular to it at ``value * scale``. (None, None)
    if there is nothing to draw."""
    fpts, faces, lpts, lines = [], [], [], []
    for e in model.elements.values():
        vals = member_end_values(e, kind)
        if vals is None:
            continue
        c = e.node_coords()
        if c.shape[0] != 2:
            continue
        i = np.asarray(to_xyz(c[0]), dtype=float)
        j = np.asarray(to_xyz(c[1]), dtype=float)
        d = j - i
        L = float(np.linalg.norm(d))
        if L == 0.0:
            continue
        perp = _diagram_perp(e, i, j)
        ai = i + perp * (vals[0] * scale)
        aj = j + perp * (vals[1] * scale)
        b = len(fpts)
        fpts += [i.tolist(), ai.tolist(), aj.tolist(), j.tolist()]
        faces += [3, b, b + 1, b + 2, 3, b, b + 2, b + 3]
        lb = len(lpts)
        lpts += [i.tolist(), ai.tolist(), aj.tolist(), j.tolist()]
        lines += [2, lb, lb + 1, 2, lb + 1, lb + 2, 2, lb + 2, lb + 3]
    fill = pv.PolyData(np.asarray(fpts), faces=np.asarray(faces)) if fpts else None
    outline = (pv.PolyData(np.asarray(lpts), lines=np.asarray(lines))
               if lpts else None)
    return fill, outline


def element_line(element):
    """PolyData of a single 2-node element's line (for per-member colouring),
    or None for non-line elements."""
    c = element.node_coords()
    if c.shape[0] != 2:
        return None
    pts = np.array([to_xyz(c[0]), to_xyz(c[1])], dtype=float)
    return pv.PolyData(pts, lines=np.array([2, 0, 1], dtype=np.int64))


# ---------------------------------------------------------------- pick support

def point_segment_distance(p, a, b) -> float:
    """Shortest distance from point ``p`` to the segment ``a``-``b``."""
    ab = b - a
    L2 = float(ab @ ab)
    if L2 == 0.0:
        return float(np.linalg.norm(p - a))
    t = max(0.0, min(1.0, float((p - a) @ ab) / L2))
    return float(np.linalg.norm(p - (a + t * ab)))


def nearest_item(model, point, tol):
    """Nearest ('node', tag) then ('member', tag) to a 3-D ``point`` within
    ``tol``, or None. Nodes win ties (they sit on member ends)."""
    p = np.asarray(point, dtype=float).ravel()[:3]
    best = (None, float("inf"))
    for tag, n in model.nodes.items():
        d = float(np.linalg.norm(p - np.asarray(to_xyz(n.coords))))
        if d < best[1]:
            best = (("node", tag), d)
    if best[0] is not None and best[1] <= tol:
        return best[0]
    best = (None, float("inf"))
    for tag, e in model.elements.items():
        c = e.node_coords()
        if c.shape[0] != 2:
            continue
        d = point_segment_distance(p, np.asarray(to_xyz(c[0])),
                                   np.asarray(to_xyz(c[1])))
        if d < best[1]:
            best = (("member", tag), d)
    if best[0] is not None and best[1] <= tol:
        return best[0]
    return None
