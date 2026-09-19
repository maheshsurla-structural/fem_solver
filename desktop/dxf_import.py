"""Minimal DXF grid import (wall plan W8d).

Reads ``LINE`` and ``LWPOLYLINE`` entities from an ASCII DXF and turns them into
grid lines — a real ETABS pain point we make one-click. No third-party
dependency: DXF is group-code / value pairs, so a tiny scanner suffices (mirrors
the native writer in ``femsolver.results.dxf``).

``grids_from_dxf`` classifies each imported segment: an axis-aligned line becomes
a :class:`project.GridLine` (const-X or const-Y at its ordinate); anything
diagonal becomes a :class:`project.GeneralGrid`. Coincident parallel lines are
de-duplicated so a doubled CAD grid doesn't import twice.
"""
from __future__ import annotations

from project import GeneralGrid, GridLine


def _pairs(text: str):
    """Yield (code:int, value:str) group pairs from DXF ``text``."""
    lines = text.splitlines()
    i = 0
    n = len(lines)
    while i + 1 < n:
        code = lines[i].strip()
        val = lines[i + 1].strip()
        i += 2
        try:
            yield int(code), val
        except ValueError:
            continue


def read_segments(text: str):
    """``[((x1,y1),(x2,y2), layer), …]`` for every LINE and each LWPOLYLINE
    segment in the DXF ``text``."""
    segs = []
    ent = None            # current entity type
    cur: dict = {}
    poly_pts: list = []
    poly_layer = "0"

    def _flush_line():
        if {10, 20, 11, 21} <= cur.keys():
            segs.append(((cur[10], cur[20]), (cur[11], cur[21]),
                         cur.get(8, "0")))

    def _flush_poly():
        for a, b in zip(poly_pts, poly_pts[1:]):
            segs.append((a, b, poly_layer))

    for code, val in _pairs(text):
        if code == 0:                       # entity boundary
            if ent == "LINE":
                _flush_line()
            elif ent == "LWPOLYLINE":
                _flush_poly()
            ent = val.upper()
            cur = {}
            poly_pts = []
            poly_layer = "0"
            continue
        if ent == "LINE":
            if code in (10, 20, 11, 21):
                try:
                    cur[code] = float(val)
                except ValueError:
                    pass
            elif code == 8:
                cur[8] = val
        elif ent == "LWPOLYLINE":
            if code == 8:
                poly_layer = val
            elif code == 10:
                poly_pts.append([float(val), None])
            elif code == 20 and poly_pts and poly_pts[-1][1] is None:
                poly_pts[-1][1] = float(val)
    # trailing entity (files sometimes omit a final 0/ENDSEC before EOF pair)
    if ent == "LINE":
        _flush_line()
    elif ent == "LWPOLYLINE":
        _flush_poly()
    # normalize polyline vertices to tuples
    return [(tuple(a), tuple(b), lay) for a, b, lay in segs
            if None not in a and None not in b]


def grids_from_dxf(text: str, *, tol: float = 1e-6, scale: float = 1.0):
    """``(grid_lines, general_grids)`` from a DXF's line entities (wall plan
    W8d). ``scale`` converts DXF units to metres. Axis-aligned segments →
    ``GridLine``; diagonal → ``GeneralGrid``. Coincident parallels are merged."""
    segs = read_segments(text)
    x_coords: list = []                     # const-X ordinates (vertical lines)
    y_coords: list = []                     # const-Y ordinates (horizontal)
    generals: list = []

    def _near(v, seen):
        return any(abs(v - s) <= tol for s in seen)

    for (x1, y1), (x2, y2), _lay in segs:
        x1, y1, x2, y2 = (x1 * scale, y1 * scale, x2 * scale, y2 * scale)
        if abs(x1 - x2) <= tol and abs(y1 - y2) > tol:      # vertical → const-X
            if not _near(x1, x_coords):
                x_coords.append(x1)
        elif abs(y1 - y2) <= tol and abs(x1 - x2) > tol:    # horizontal → const-Y
            if not _near(y1, y_coords):
                y_coords.append(y1)
        elif abs(x1 - x2) > tol or abs(y1 - y2) > tol:      # diagonal → general
            generals.append((x1, y1, x2, y2))

    grids = []
    gid = 1
    for i, x in enumerate(sorted(x_coords)):
        grids.append(GridLine(id=gid, name=_alpha(i), axis="x", coord=x))
        gid += 1
    for j, y in enumerate(sorted(y_coords)):
        grids.append(GridLine(id=gid, name=str(j + 1), axis="y", coord=y))
        gid += 1
    gens = [GeneralGrid(id=k + 1, name=f"D{k + 1}", x1=a, y1=b, x2=c, y2=d)
            for k, (a, b, c, d) in enumerate(generals)]
    return grids, gens


def _alpha(k: int) -> str:
    s = ""
    k += 1
    while k:
        k, r = divmod(k - 1, 26)
        s = chr(ord("A") + r) + s
    return s
