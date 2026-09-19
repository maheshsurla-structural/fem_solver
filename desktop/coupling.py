"""Coupling beams between shear walls (wall plan W6).

A coupling beam spans the gap between two wall piers at a floor level — the
coupled-wall system where a lintel over a doorway ties two walls so they act
partly as one. In the desktop's *meshed shell* wall model (unlike the engine's
2-D macro `coupling_beam`), a coupling beam is simply a beam ``Member`` between a
node on one wall's facing (inner) edge and the matching node on the other's, at
the coupling elevation. Because the build merges coincident nodes
(``Project._node_registry``), a project node placed on a wall edge at a mesh-row
elevation is unified with the shell mesh node there, so the beam ties into the
wall.

The elevation is snapped to each wall's nearest mesh row so the beam ends land
on real shell nodes.
"""
from __future__ import annotations

import math

from project import Member, Node

_TOL = 1e-6


def _corners(project, area):
    c = {n.id: (n.x, n.y, n.z) for n in project.nodes}
    return [c[nid] for nid in area.nodes]


def _bilin(corners, s, t):
    p0, p1, p2, p3 = corners
    a, b = (1 - s) * (1 - t), s * (1 - t)
    cc, d = s * t, (1 - s) * t
    return tuple(a * p0[k] + b * p1[k] + cc * p2[k] + d * p3[k]
                 for k in range(3))


def _plan_centroid(corners):
    n = len(corners)
    return (sum(c[0] for c in corners) / n, sum(c[1] for c in corners) / n)


def _snap_level(corners, n2, elev):
    """Snap absolute elevation ``elev`` to the wall's nearest mesh-row elevation
    (so an end lands on a shell node). Returns the height fraction t in 0..1."""
    z0 = corners[0][2]                              # base corner z
    H = corners[3][2] - z0                          # top-left corner z − base
    if abs(H) < _TOL:
        return 0.0
    n2 = max(1, int(n2))
    j = min(max(round((elev - z0) / H * n2), 0), n2)
    return j / n2


def add_coupling_beam(project, area_a_id, area_b_id, elev, section, material):
    """Add a coupling beam between wall panels ``area_a_id`` and ``area_b_id`` at
    elevation ``elev``. Each end sits on the wall's *inner* vertical edge (the
    one facing the other wall), snapped to that wall's nearest mesh row. Creates
    the two end nodes (merged with existing coincident nodes) and a beam
    ``Member``. Returns ``(node_a, node_b, member_id)``.

    Raises ``ValueError`` for a missing/non-quad area or coincident ends."""
    a = project.area(area_a_id)
    b = project.area(area_b_id)
    if a is None or b is None or len(a.nodes) != 4 or len(b.nodes) != 4:
        raise ValueError("Coupling beams need two 4-node wall panels.")
    ca, cb = _corners(project, a), _corners(project, b)
    cen_a, cen_b = _plan_centroid(ca), _plan_centroid(cb)
    ta = _snap_level(ca, a.mesh[1], elev)
    tb = _snap_level(cb, b.mesh[1], elev)

    def _inner_point(corners, t, other_centroid):
        # candidate edge points: s=0 (P0-P3 edge) and s=1 (P1-P2 edge); pick the
        # vertical edge whose plan position is nearer the other wall
        p_lo = _bilin(corners, 0.0, t)
        p_hi = _bilin(corners, 1.0, t)
        d_lo = math.dist(p_lo[:2], other_centroid)
        d_hi = math.dist(p_hi[:2], other_centroid)
        return p_lo if d_lo <= d_hi else p_hi

    pa = _inner_point(ca, ta, cen_b)
    pb = _inner_point(cb, tb, cen_a)

    if math.dist(pa, pb) < _TOL:
        raise ValueError("Coupling-beam ends coincide — pick two distinct piers.")

    node_a = _node_at_coord(project, pa)
    node_b = _node_at_coord(project, pb)
    mid = max((m.id for m in project.members), default=0) + 1
    project.members.append(Member(id=mid, n1=node_a, n2=node_b,
                                  section=int(section), material=int(material)))
    return node_a, node_b, mid


def _node_at_coord(project, p, tol=1e-6):
    """Id of a project node at ``p`` (within ``tol``), creating one if absent."""
    for n in project.nodes:
        if (abs(n.x - p[0]) < tol and abs(n.y - p[1]) < tol
                and abs(n.z - p[2]) < tol):
            return n.id
    nid = max((n.id for n in project.nodes), default=0) + 1
    project.nodes.append(Node(id=nid, x=float(p[0]), y=float(p[1]),
                              z=float(p[2])))
    return nid
