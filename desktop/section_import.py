"""Import a Custom section outline (+ holes + rebars) from a DXF or CSV file.

Both parsers return a dict ``{"outline": [(z, y), ...], "holes": [[...], ...],
"bars": [(z, y, dia), ...]}`` in the file's own units (the caller scales to
metres). No third-party dependency: the DXF reader is a small ASCII-DXF parser
covering the entities section designers export — LWPOLYLINE, POLYLINE/VERTEX,
and CIRCLE (circles become rebars). The largest closed polyline is the outline;
any smaller closed polylines are treated as voids.

**CSV** — one entity per row, tagged in the first column:
    outline, z, y
    hole,    z, y          (consecutive hole rows form one void ring)
    bar,     z, y, dia
A header-less file whose rows are just ``z, y`` is read as the outline.
"""
from __future__ import annotations

import csv
import io


def _area(ring) -> float:
    """Absolute shoelace area of a ring of (z, y) points."""
    n = len(ring)
    if n < 3:
        return 0.0
    a = 0.0
    for i in range(n):
        z1, y1 = ring[i]
        z2, y2 = ring[(i + 1) % n]
        a += z1 * y2 - z2 * y1
    return abs(a) / 2.0


def _dedupe_closed(ring):
    """Drop a trailing point equal to the first (closed rings)."""
    if len(ring) > 1 and abs(ring[0][0] - ring[-1][0]) < 1e-9 \
            and abs(ring[0][1] - ring[-1][1]) < 1e-9:
        return ring[:-1]
    return ring


def _assemble(rings, circles):
    """Pick the largest ring as the outline, the rest as holes; circles→bars."""
    rings = [_dedupe_closed(r) for r in rings if len(r) >= 3]
    rings.sort(key=_area, reverse=True)
    outline = rings[0] if rings else []
    holes = rings[1:]
    bars = [(cz, cy, 2.0 * r) for (cz, cy, r) in circles if r > 0]
    return {"outline": outline, "holes": holes, "bars": bars}


# --------------------------------------------------------------- CSV
def parse_csv(text: str) -> dict:
    rows = list(csv.reader(io.StringIO(text)))
    outline, bars = [], []
    holes: list = []
    cur_hole: list = []

    def _nums(cells):
        out = []
        for c in cells:
            c = c.strip()
            if c == "":
                continue
            out.append(float(c))
        return out

    tagged = any(r and r[0].strip().lower() in ("outline", "hole", "bar")
                 for r in rows)
    for r in rows:
        if not r or not any(c.strip() for c in r):
            continue
        if tagged:
            tag = r[0].strip().lower()
            nums = _nums(r[1:])
            if tag == "outline" and len(nums) >= 2:
                if cur_hole:
                    holes.append(cur_hole)
                    cur_hole = []
                outline.append((nums[0], nums[1]))
            elif tag == "hole" and len(nums) >= 2:
                cur_hole.append((nums[0], nums[1]))
            elif tag == "bar" and len(nums) >= 3:
                if cur_hole:
                    holes.append(cur_hole)
                    cur_hole = []
                bars.append((nums[0], nums[1], nums[2]))
        else:
            nums = _nums(r)
            if len(nums) >= 2:
                outline.append((nums[0], nums[1]))
    if cur_hole:
        holes.append(cur_hole)
    rings = [outline] + holes
    circles = [(z, y, d / 2.0) for (z, y, d) in bars]
    res = _assemble([rr for rr in rings if rr], circles)
    # a plain outline is authoritative — don't let a hole outrank it
    if not tagged:
        res = {"outline": _dedupe_closed(outline), "holes": [], "bars": []}
    return res


# --------------------------------------------------------------- DXF
def _dxf_pairs(text: str):
    lines = text.splitlines()
    i, out = 0, []
    while i + 1 < len(lines):
        code = lines[i].strip()
        val = lines[i + 1]
        i += 2
        try:
            out.append((int(code), val.strip()))
        except ValueError:
            continue
    return out


def parse_dxf(path: str) -> dict:
    with open(path, "r", errors="replace") as f:
        text = f.read()
    pairs = _dxf_pairs(text)
    # split into entities at each code-0 marker
    ents, cur_type, cur = [], None, []
    for code, val in pairs:
        if code == 0:
            if cur_type is not None:
                ents.append((cur_type, cur))
            cur_type, cur = val.upper(), []
        elif cur_type is not None:
            cur.append((code, val))
    if cur_type is not None:
        ents.append((cur_type, cur))

    rings, circles = [], []
    pending_polyline = None
    for etype, body in ents:
        if etype == "LWPOLYLINE":
            ring, x = [], None
            for code, val in body:
                if code == 10:
                    x = float(val)
                elif code == 20 and x is not None:
                    ring.append((x, float(val)))
                    x = None
            if len(ring) >= 3:
                rings.append(ring)
        elif etype == "POLYLINE":
            pending_polyline = []
        elif etype == "VERTEX" and pending_polyline is not None:
            x = y = None
            for code, val in body:
                if code == 10:
                    x = float(val)
                elif code == 20:
                    y = float(val)
            if x is not None and y is not None:
                pending_polyline.append((x, y))
        elif etype == "SEQEND" and pending_polyline is not None:
            if len(pending_polyline) >= 3:
                rings.append(pending_polyline)
            pending_polyline = None
        elif etype == "CIRCLE":
            cx = cy = r = None
            for code, val in body:
                if code == 10:
                    cx = float(val)
                elif code == 20:
                    cy = float(val)
                elif code == 40:
                    r = float(val)
            if cx is not None and cy is not None and r:
                circles.append((cx, cy, r))
    if pending_polyline and len(pending_polyline) >= 3:
        rings.append(pending_polyline)
    return _assemble(rings, circles)
