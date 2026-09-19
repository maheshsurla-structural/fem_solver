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


def element_centroids(model):
    """Return (tags, centroids (M,3)) — the tag and averaged node position of
    every element, used to place element-number labels in the viewport."""
    tags, cents = [], []
    for tag, e in model.elements.items():
        pts = [to_xyz(model.nodes[t].coords)
               for t in e.node_tags if t in model.nodes]
        if not pts:
            continue
        tags.append(tag)
        cents.append(np.mean(np.asarray(pts, dtype=float), axis=0))
    if cents:
        return tags, np.asarray(cents, dtype=float)
    return tags, np.zeros((0, 3), dtype=float)


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


# quantity -> label for the area displacement contour (slab plan S7)
AREA_CONTOUR_QUANTITIES = {
    "Umag": "|U| (total)", "Uz": "Uz (vertical)",
    "Ux": "Ux", "Uy": "Uy",
}


def _node_disp_scalar(model, quantity: str):
    """Per-node scalar (aligned with :func:`node_points` order) for an area
    displacement contour, read from each node's solved ``disp`` vector."""
    tags, _pts, _index = node_points(model)
    vals = np.zeros(len(tags))
    for i, t in enumerate(tags):
        d = np.asarray(model.nodes[t].disp, dtype=float).ravel()
        tr = d[:3] if d.size >= 3 else np.pad(d, (0, 3 - d.size))
        if quantity == "Ux":
            vals[i] = tr[0]
        elif quantity == "Uy":
            vals[i] = tr[1]
        elif quantity == "Uz":
            vals[i] = tr[2]
        else:                                    # "Umag" (default)
            vals[i] = float(np.linalg.norm(tr))
    return vals


def areas_contour_mesh(model, quantity: str = "Umag"):
    """Filled-face PolyData of the surface elements carrying a per-node
    displacement scalar in ``point_data['value']`` (slab plan S7). ``None``
    when the model has no surface elements."""
    poly = areas_mesh(model)
    if poly is None:
        return None
    poly.point_data["value"] = _node_disp_scalar(model, quantity)
    return poly


# ---- shell stress-resultant contours (slab plan S7 rich) -------------------
# Recovered shell resultants are the 8-vector [N11, N22, N12, M11, M22, M12,
# V13, V23] per unit width, in the element's LOCAL axes.
_RESULTANT_INDEX = {"N11": 0, "N22": 1, "N12": 2,
                    "M11": 3, "M22": 4, "M12": 5, "V13": 6, "V23": 7}
# quantity -> (label, unit, signed?) for the area force/moment contour
AREA_RESULT_QUANTITIES = {
    "M11": ("M11 (bending)", "N·m/m", True),
    "M22": ("M22 (bending)", "N·m/m", True),
    "M12": ("M12 (twist)", "N·m/m", True),
    "N11": ("N11 (membrane)", "N/m", True),
    "N22": ("N22 (membrane)", "N/m", True),
    "N12": ("N12 (shear)", "N/m", True),
    "Vmax": ("V (max transverse shear)", "N/m", False),
    # Wood-Armer design-moment magnitudes for orthogonal reinforcement (S9);
    # the moment each bar layer must be designed for (always ≥ 0).
    "WAx_bot": ("Wood-Armer Mx* (bottom)", "N·m/m", False),
    "WAy_bot": ("Wood-Armer My* (bottom)", "N·m/m", False),
    "WAx_top": ("Wood-Armer Mx* (top)", "N·m/m", False),
    "WAy_top": ("Wood-Armer My* (top)", "N·m/m", False),
}


def _element_resultant(el):
    """Element-representative stress resultant 8-vector (mean of the Gauss-point
    resultants for a quad; the single value for a Tri3), or ``None`` if the
    element has not been recovered (no solve yet) or carries no resultants."""
    gp = getattr(el, "gp_resultants", None)
    if gp:                                        # non-empty list of GP vectors
        return np.mean(np.asarray(gp, dtype=float), axis=0)
    r = getattr(el, "resultants", None)
    return np.asarray(r, dtype=float) if r is not None else None


# Gauss-point → corner-node extrapolation for a 2×2 quad (slab plan S7 refine).
# GP order from gauss_legendre_2d_quad(2): (-g,-g),(-g,+g),(+g,-g),(+g,+g);
# node order (MITC4 _N): (-1,-1),(+1,-1),(+1,+1),(-1,+1). Bilinear extrapolation
# with a=1+√3/2 (GP at the node's corner), c=1-√3/2 (opposite), b=-1/2 (adjacent).
_A_EXT = 1.0 + np.sqrt(3.0) / 2.0
_C_EXT = 1.0 - np.sqrt(3.0) / 2.0
_GP2NODE_Q4 = np.array([
    [_A_EXT, -0.5, -0.5, _C_EXT],
    [-0.5, _C_EXT, _A_EXT, -0.5],
    [_C_EXT, -0.5, -0.5, _A_EXT],
    [-0.5, _A_EXT, _C_EXT, -0.5],
])


def _element_nodal_resultants(el):
    """Stress resultants at the element's own nodes, shape ``(n_nodes, 8)``.

    A 4-node/4-GP quad (MITC4) extrapolates its 2×2 Gauss values to the corners
    — sharper support-moment peaks than scattering a single element mean. Other
    elements (Tri3 single value, DKMQ4 3×3 GPs) repeat their mean/value to each
    node. ``None`` if the element has no recovered resultants."""
    gp = getattr(el, "gp_resultants", None)
    nt = getattr(el, "node_tags", ())
    if gp:
        arr = np.asarray(gp, dtype=float)
        if arr.shape[0] == 4 and len(nt) == 4:
            return _GP2NODE_Q4 @ arr             # (4, 8) extrapolated to corners
        n = len(nt) or arr.shape[0]
        return np.tile(arr.mean(axis=0), (n, 1))
    r = getattr(el, "resultants", None)
    if r is not None:
        n = len(nt) or 1
        return np.tile(np.asarray(r, dtype=float), (n, 1))
    return None


def _resultant_scalar(res, quantity: str):
    if res is None:
        return None
    if quantity == "Vmax":
        return float(np.hypot(res[6], res[7]))
    if quantity.startswith("WA"):                # Wood-Armer design moment (S9)
        from femsolver.design.wood_armer import wood_armer_moments
        # The shell elements use "negative M = sagging (bottom tension)"; the
        # Wood-Armer utility uses the textbook "positive = sagging", so negate
        # the moments here (the |Mxy| twist term is sign-independent).
        wa = wood_armer_moments(-res[3], -res[4], -res[5])   # M11, M22, M12
        return {"WAx_bot": abs(wa.mx_bot), "WAy_bot": abs(wa.my_bot),
                "WAx_top": abs(wa.mx_top), "WAy_top": abs(wa.my_top)}.get(quantity)
    i = _RESULTANT_INDEX.get(quantity)
    return None if i is None else float(res[i])


def areas_result_mesh(model, quantity: str = "M11"):
    """Filled-face PolyData of the surface elements carrying a **nodal-averaged**
    stress-resultant scalar in ``point_data['value']`` (slab plan S7). Each
    element's Gauss-point resultants are extrapolated to its own nodes
    (:func:`_element_nodal_resultants`) then averaged across the elements meeting
    at each node, so peaks (e.g. clamped-edge moments) stay sharp. ``None`` when
    there are no surfaces. Values are in the elements' local axes."""
    poly = areas_mesh(model)
    if poly is None:
        return None
    tags, _pts, index = node_points(model)
    acc = np.zeros(len(tags))
    cnt = np.zeros(len(tags))
    for el in model.elements.values():
        nt = getattr(el, "node_tags", ())
        if len(nt) not in (3, 4):
            continue
        nodal = _element_nodal_resultants(el)
        if nodal is None:
            continue
        for k, t in enumerate(nt):
            s = _resultant_scalar(nodal[k], quantity)
            if s is None:
                continue
            r = index.get(t)
            if r is not None:
                acc[r] += s
                cnt[r] += 1.0
    vals = np.divide(acc, cnt, out=np.zeros_like(acc), where=cnt > 0)
    poly.point_data["value"] = vals
    return poly


# quantity -> (label, Wood-Armer component) for the required-reinforcement
# contour (slab plan S9). Values are As per unit width, shown in mm²/m.
AREA_REBAR_QUANTITIES = {
    "As_x_bot": ("As x, bottom", "mx_bot"),
    "As_y_bot": ("As y, bottom", "my_bot"),
    "As_x_top": ("As x, top", "mx_top"),
    "As_y_top": ("As y, top", "my_top"),
}


def areas_reinforcement_mesh(model, quantity: str = "As_x_bot", *,
                             cover: float = 0.025, fy: float = 420e6,
                             fc: float = 30e6):
    """Filled-face PolyData carrying the **required steel area per width** (in
    **mm²/m**, in ``point_data['value']``) for a Wood-Armer design moment over
    the slab elements (slab plan S9). ``cover`` is to the bar centroid; the
    effective depth is ``element thickness − cover``. Nodal-averaged; ``None``
    when there are no surfaces. Nodes where the section is too shallow to carry
    the moment singly-reinforced get NaN (flagged blank in the contour)."""
    from femsolver.design.wood_armer import (required_reinforcement,
                                             wood_armer_moments)
    poly = areas_mesh(model)
    if poly is None:
        return None
    comp = AREA_REBAR_QUANTITIES.get(quantity, AREA_REBAR_QUANTITIES["As_x_bot"])[1]
    tags, _pts, index = node_points(model)
    acc = np.zeros(len(tags))
    cnt = np.zeros(len(tags))
    bad = np.zeros(len(tags), dtype=bool)
    for e in model.elements.values():
        nt = getattr(e, "node_tags", ())
        if len(nt) not in (3, 4):
            continue
        nodal = _element_nodal_resultants(e)     # GP→node extrapolated (S7)
        if nodal is None:
            continue
        d = float(getattr(e, "thickness", 0.0)) - cover
        if d <= 0.0:
            continue
        for k, t in enumerate(nt):
            r = index.get(t)
            if r is None:
                continue
            res = nodal[k]
            # shell convention: negative M = sagging; negate for Wood-Armer
            wa = wood_armer_moments(-res[3], -res[4], -res[5])
            m_star = abs(getattr(wa, comp))
            try:
                acc[r] += required_reinforcement(m_star, d, fy, fc) * 1.0e6
                cnt[r] += 1.0
            except ValueError:                   # section inadequate at this node
                bad[r] = True
    vals = np.divide(acc, cnt, out=np.zeros_like(acc), where=cnt > 0)
    vals[bad & (cnt == 0)] = np.nan              # flag inadequate-only nodes
    poly.point_data["value"] = vals
    return poly


def slab_results_table(model):
    """Per-node slab results for export/reporting (slab plan S9): one row per
    node that belongs to a surface element, with displacement + nodal-averaged
    stress resultants. Returns ``(headers, rows)`` or ``None`` if the model has
    no areas. Resultants are in the elements' local axes."""
    tags, pts, index = node_points(model)
    area_nodes = set()
    for e in model.elements.values():
        if len(getattr(e, "node_tags", ())) in (3, 4):
            area_nodes.update(e.node_tags)
    if not area_nodes:
        return None
    uz = _node_disp_scalar(model, "Uz")
    umag = _node_disp_scalar(model, "Umag")
    res = {q: areas_result_mesh(model, q)
           for q in ("M11", "M22", "M12", "Vmax")}
    headers = ["node", "x", "y", "z", "Uz", "|U|",
               "M11", "M22", "M12", "Vmax"]
    rows = []
    for i, t in enumerate(tags):
        if t not in area_nodes:
            continue
        x, y, z = (float(v) for v in pts[i])
        row = [t, x, y, z, float(uz[i]), float(umag[i])]
        for q in ("M11", "M22", "M12", "Vmax"):
            m = res[q]
            row.append(float(m.point_data["value"][i]) if m is not None else 0.0)
        rows.append(row)
    return headers, rows


def _area_frame(pts):
    """Orthonormal local frame (e1, e2, e3=normal) at a 3/4-node area's centroid,
    matching the shell elements' local axes. ``pts`` is (3or4, 3)."""
    pts = np.asarray(pts, dtype=float)
    if len(pts) == 4:
        r1 = 0.5 * ((pts[1] - pts[0]) + (pts[2] - pts[3]))
        r2 = 0.5 * ((pts[3] - pts[0]) + (pts[2] - pts[1]))
    else:
        r1 = pts[1] - pts[0]
        r2 = pts[2] - pts[0]
    n = np.cross(r1, r2)
    nn = np.linalg.norm(n)
    if nn < 1e-12 or np.linalg.norm(r1) < 1e-12:
        return None
    e3 = n / nn
    e1 = r1 / np.linalg.norm(r1)
    e2 = np.cross(e3, e1)
    return pts.mean(axis=0), e1, e2, e3


def area_local_axes(model, scale: float | None = None):
    """Local-axis triads for every surface element (slab plan S6): returns
    ``(e1_poly, e2_poly, e3_poly)`` line-segment PolyData (centroid → centroid +
    scale·axis) so the view can draw them red/green/blue, or ``None`` if there
    are no areas. ``scale`` defaults to ~5% of the model span."""
    _tags, _pts, index = node_points(model)
    coords = {t: to_xyz(model.nodes[t].coords) for t in model.nodes}
    if scale is None:
        scale = max(model_span(model) * 0.05, 1e-3)
    segs = ([], [], [])                           # e1, e2, e3 point pairs
    for e in model.elements.values():
        nt = getattr(e, "node_tags", ())
        if len(nt) not in (3, 4) or any(t not in coords for t in nt):
            continue
        frame = _area_frame([coords[t] for t in nt])
        if frame is None:
            continue
        c, e1, e2, e3 = frame
        for k, ax in enumerate((e1, e2, e3)):
            segs[k].append((c, c + scale * ax))
    out = []
    for pairs in segs:
        if not pairs:
            out.append(None)
            continue
        pts = np.array([p for pair in pairs for p in pair], dtype=float)
        lines = []
        for i in range(len(pairs)):
            lines += [2, 2 * i, 2 * i + 1]
        out.append(pv.PolyData(pts, lines=np.asarray(lines, dtype=np.int64)))
    return tuple(out)


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


# --------------------------------------------------------------- stories / grid
def _model_bbox(model):
    """(lo, hi) 3-vectors of the model's node bounding box, or ``None``."""
    _t, pts, _i = node_points(model)
    pts = np.asarray(pts, dtype=float)
    if not len(pts):
        return None
    return pts.min(axis=0), pts.max(axis=0)


def grid_story_mesh(project, model, *, pad=None):
    """Lines PolyData for the named grid lines (drawn on the base plane) and the
    story levels (a horizontal rectangle at each elevation), spanning the model
    plan bounds (wall plan W4b). ``None`` if there is nothing to draw.

    Grid ``axis="x"`` is a line of constant X (runs in Y); ``axis="y"`` constant
    Y (runs in X). Story rectangles let the levels read in a 3-D view."""
    stories = list(getattr(project, "stories", []))
    grids = list(getattr(project, "grid_lines", []))
    generals = list(getattr(project, "general_grids", []))
    if not stories and not grids and not generals:
        return None
    bb = _model_bbox(model)
    if bb is None:
        return None
    lo, hi = bb
    span = float(max(hi[0] - lo[0], hi[1] - lo[1], 1.0))
    if pad is None:
        pad = 0.1 * span
    x0, x1 = float(lo[0] - pad), float(hi[0] + pad)
    y0, y1 = float(lo[1] - pad), float(hi[1] + pad)
    zbase = float(lo[2])
    verts: list = []
    lines: list = []

    def _seg(a, b):
        i = len(verts)
        verts.append(a)
        verts.append(b)
        lines.extend((2, i, i + 1))

    for g in grids:
        if not getattr(g, "visible", True):
            continue
        if g.axis == "x":
            _seg((g.coord, y0, zbase), (g.coord, y1, zbase))
        else:
            _seg((x0, g.coord, zbase), (x1, g.coord, zbase))
    for g in getattr(project, "general_grids", []):     # diagonal / arbitrary
        if not getattr(g, "visible", True):
            continue
        _seg((g.x1, g.y1, zbase), (g.x2, g.y2, zbase))
    for s in stories:
        z = float(s.elev)
        c = [(x0, y0, z), (x1, y0, z), (x1, y1, z), (x0, y1, z)]
        for i in range(4):
            _seg(c[i], c[(i + 1) % 4])
    if not verts:
        return None
    poly = pv.PolyData(np.asarray(verts, dtype=float))
    poly.lines = np.asarray(lines, dtype=np.int64)
    return poly


def snap_targets(project):
    """``(xs, ys, zs)`` the named-grid X coordinates, Y coordinates and story
    elevations to snap drawing to (wall plan W4b)."""
    xs = [g.coord for g in getattr(project, "grid_lines", []) if g.axis == "x"]
    ys = [g.coord for g in getattr(project, "grid_lines", []) if g.axis == "y"]
    zs = [s.elev for s in getattr(project, "stories", [])]
    return xs, ys, zs


def grid_bubble_labels(project, model, *, pad=None):
    """``(points (N,3), labels [str])`` for each visible grid line's name bubble
    at its chosen end (wall plan W8b) — for the viewport to draw as a callout.
    Points sit just outside the model plan bounds so the bubbles clear the model.
    Returns ``([], [])`` when there is nothing to label."""
    grids = [g for g in getattr(project, "grid_lines", [])
             if getattr(g, "visible", True) and g.bubble != "none"]
    generals = [g for g in getattr(project, "general_grids", [])
                if getattr(g, "visible", True) and g.bubble != "none"]
    bb = _model_bbox(model)
    if bb is None or (not grids and not generals):
        return [], []
    lo, hi = bb
    span = float(max(hi[0] - lo[0], hi[1] - lo[1], 1.0))
    if pad is None:
        pad = 0.1 * span
    off = 0.5 * pad
    x0, x1 = float(lo[0] - pad), float(hi[0] + pad)
    y0, y1 = float(lo[1] - pad), float(hi[1] + pad)
    z = float(lo[2])
    pts, labels = [], []
    for g in grids:
        if g.axis == "x":                       # const-X line runs in Y
            y = (y0 - off) if g.bubble == "start" else (y1 + off)
            pts.append((g.coord, y, z))
        else:                                   # const-Y line runs in X
            x = (x0 - off) if g.bubble == "start" else (x1 + off)
            pts.append((x, g.coord, z))
        labels.append(g.name)
    for g in generals:
        if g.bubble == "start":
            px, py = g.x1, g.y1
        else:
            px, py = g.x2, g.y2
        pts.append((float(px), float(py), z))
        labels.append(g.name)
    return pts, labels


def snap_to_grid(x, y, z, xs, ys, zs, tol):
    """Snap each of ``x, y, z`` independently to the nearest value in ``xs, ys,
    zs`` within ``tol`` (else leave it). Grid/story snapping for the draw tools."""
    def _near(v, cands):
        if not cands:
            return v
        best = min(cands, key=lambda c: abs(c - v))
        return best if abs(best - v) <= tol else v
    return _near(x, xs), _near(y, ys), _near(z, zs)


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


def _decode_area_id(tag: int):
    """Project ``Area`` id for a mesh sub-element tag, or ``None`` if the tag is
    not in the area range (slab S10)."""
    from project import AREA_TAG_BASE, AREA_TAG_STRIDE
    if tag is None or tag < AREA_TAG_BASE:
        return None
    return (tag - AREA_TAG_BASE) // AREA_TAG_STRIDE


def _point_in_polygon_2d(pt, poly) -> bool:
    """Ray-casting point-in-polygon test in 2-D (``poly`` = list of (x, y))."""
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if ((y1 > y) != (y2 > y)) and \
                (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-30) + x1):
            inside = not inside
    return inside


def _point_in_area(p, pts, tol: float) -> bool:
    """True if 3-D point ``p`` lies within a 3/4-node area face ``pts`` (project
    to the face plane; require the out-of-plane distance ≤ ``tol``)."""
    frame = _area_frame(pts)
    if frame is None:
        return False
    c, e1, e2, e3 = frame
    rel = np.asarray(p, dtype=float) - c
    if abs(float(rel @ e3)) > tol:                # too far off the face plane
        return False
    poly = [(float((np.asarray(q) - c) @ e1), float((np.asarray(q) - c) @ e2))
            for q in pts]
    return _point_in_polygon_2d((float(rel @ e1), float(rel @ e2)), poly)


def area_faces_mesh(model, area_id: int):
    """Filled-face PolyData of just the mesh sub-elements belonging to project
    ``area_id`` (slab S10 selection highlight), or ``None``."""
    _tags, pts, index = node_points(model)
    faces = []
    for tag, e in model.elements.items():
        if _decode_area_id(tag) != area_id:
            continue
        nt = e.node_tags
        if len(nt) in (3, 4) and all(t in index for t in nt):
            faces.append(len(nt))
            faces.extend(index[t] for t in nt)
    if not faces:
        return None
    return pv.PolyData(pts, faces=np.asarray(faces, dtype=np.int64))


def _area_element_at(model, pt, tol):
    """The shell element whose face contains 3-D point ``pt`` (within ``tol`` of
    its plane), or ``None``."""
    for _tag, e in model.elements.items():
        nt = getattr(e, "node_tags", ())
        if len(nt) not in (3, 4):
            continue
        if _point_in_area(pt, [to_xyz(model.nodes[t].coords) for t in nt], tol):
            return e
    return None


def section_cut(model, p0, p1, quantity: str = "M", n: int = 64,
                tol: float = 0.05):
    """Integrate a shell stress resultant along the straight cut ``p0``→``p1``
    (a 'design strip' / SAP-style section cut, slab plan S7). Returns the total
    across the cut:

    * ``quantity="M"`` — bending moment about the cut, ``∫ nᵀ·[[M11,M12],
      [M12,M22]]·n ds`` (N·m), the design-strip moment.
    * ``quantity="V"`` — vertical shear crossing the cut, ``∫ (V13·nx + V23·ny)
      ds`` (N).

    ``n`` midpoint samples along the length. Uses element-average resultants in
    the elements' local axes — exact for an axis-aligned flat slab (local ≈
    global); rotated meshes are approximate. Points off any area contribute 0."""
    a = np.asarray(to_xyz(p0), dtype=float)
    b = np.asarray(to_xyz(p1), dtype=float)
    d = b - a
    L = float(np.linalg.norm(d[:2]))
    if L < 1e-12:
        return 0.0
    tdir = d[:2] / L
    nrm = np.array([-tdir[1], tdir[0]])          # in-plane unit normal to the cut
    ds = L / n
    total = 0.0
    for i in range(n):
        s = (i + 0.5) * ds
        pt = a + d * (s / L)
        el = _area_element_at(model, pt, tol)
        if el is None:
            continue
        res = _element_resultant(el)
        if res is None:
            continue
        if quantity == "V":
            val = res[6] * nrm[0] + res[7] * nrm[1]
        else:                                     # "M" — moment about the cut
            M = np.array([[res[3], res[5]], [res[5], res[4]]])
            val = float(nrm @ M @ nrm)
        total += val * ds
    return total


def nearest_item(model, point, tol):
    """Nearest ('node', tag) then ('member', tag), then the ('area', id) whose
    face contains ``point`` — within ``tol`` — or None. Nodes win ties (they sit
    on member ends); areas are picked by clicking their interior."""
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
        # a beam split at a slab edge (BE2) is many 2-node sub-elements; report
        # the project member it belongs to, not the sub-element tag (BE3).
        from project import decode_member_id
        return ("member", decode_member_id(best[0][1]))
    # areas last — click inside a face selects the (project) area object
    for tag, e in model.elements.items():
        nt = getattr(e, "node_tags", ())
        if len(nt) not in (3, 4):
            continue
        aid = _decode_area_id(tag)
        if aid is None:
            continue
        if _point_in_area(p, [to_xyz(model.nodes[t].coords) for t in nt], tol):
            return ("area", aid)
    return None
