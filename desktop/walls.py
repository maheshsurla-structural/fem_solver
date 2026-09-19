"""Wall creation helpers (wall plan W1).

A wall is an ``Area`` with ``role="wall"`` (see wall plan W0). The natural way
to draw one — the ETABS "draw wall by line + height" idiom — is to take a
**base line** (an ordered run of existing nodes, the wall's bottom edge) and
**extrude it upward** by a height into one vertical quad wall panel per base
segment. This module holds that geometry as a pure function so it is unit-
testable headless, with the GUI action (``MainWindow.draw_wall``) a thin shell
over it.

The panel for a base segment ``A -> B`` is the quad ``[A, B, B', A']`` where
``A'``/``B'`` are the new top nodes directly above ``A``/``B`` (same x, y; z +
``height``). Top nodes are shared between consecutive segments so a multi-
segment wall keeps a conforming mesh. ``mesh = (n1, n2)`` divides each panel
``n1`` along the base (horizontal) and ``n2`` up the height (vertical),
resolved to elements at solve time exactly like a slab area.
"""
from __future__ import annotations

from project import Area, Node


def _next_node_id(project) -> int:
    return max((n.id for n in project.nodes), default=0) + 1


def wall_baseline_from_nodes(project, node_ids) -> list:
    """Order a set/list of base node ids into a connected polyline for
    extrusion. For the common two-node case this just returns the pair. For
    more nodes it is a nearest-neighbour chain from an endpoint, which is
    adequate for the straight / gently-kinked base lines walls are drawn on;
    callers that already have an order should pass it through unchanged.

    Raises ``ValueError`` if fewer than two distinct nodes are given or a node
    id is unknown."""
    ids = [int(n) for n in node_ids]
    # de-dup preserving order
    seen, ordered = set(), []
    for n in ids:
        if n not in seen:
            seen.add(n)
            ordered.append(n)
    if len(ordered) < 2:
        raise ValueError("A wall base line needs at least two distinct nodes.")
    coords = {n.id: n for n in project.nodes}
    missing = [n for n in ordered if n not in coords]
    if missing:
        raise ValueError(f"Unknown base node id(s): {missing}")
    if len(ordered) == 2:
        return ordered
    # nearest-neighbour chain from the node with the most extreme (x, y)
    def xy(nid):
        p = coords[nid]
        return (p.x, p.y)
    start = min(ordered, key=xy)
    chain, remaining = [start], set(ordered) - {start}
    while remaining:
        cx, cy = xy(chain[-1])
        nxt = min(remaining,
                  key=lambda n: (xy(n)[0] - cx) ** 2 + (xy(n)[1] - cy) ** 2)
        chain.append(nxt)
        remaining.discard(nxt)
    return chain


def build_wall_line(project, base_nodes, height, shell_section, material,
                    mesh=(2, 2), pier=None, spandrel=None) -> list:
    """Extrude an ordered base line upward by ``height`` into vertical wall
    area(s) — one quad panel per base segment.

    Parameters
    ----------
    project : Project
        Mutated in place: new top ``Node``s and wall ``Area``s are appended.
    base_nodes : list[int]
        Ordered existing node ids forming the bottom edge (>= 2). Order matters;
        use :func:`wall_baseline_from_nodes` first if you only have a set.
    height : float
        Extrusion height (project length units, along +Z). Must be non-zero.
    shell_section, material : int
        Ids assigned to every created wall area.
    mesh : tuple[int, int]
        ``(n1, n2)`` = (divisions along the base, divisions up the height).
    pier, spandrel : str | None
        Wall labels applied to every created area.

    Returns
    -------
    list[int]
        The ids of the newly created wall areas (one per base segment).
    """
    if height == 0:
        raise ValueError("Wall height must be non-zero.")
    ordered = [int(n) for n in base_nodes]
    if len(ordered) < 2 or len(set(ordered)) != len(ordered):
        raise ValueError("A wall base line needs at least two distinct nodes.")
    coords = {n.id: n for n in project.nodes}
    missing = [n for n in ordered if n not in coords]
    if missing:
        raise ValueError(f"Unknown base node id(s): {missing}")

    # one new top node per distinct base node, directly above it
    nid = _next_node_id(project)
    top_of: dict[int, int] = {}
    for b in ordered:
        base = coords[b]
        top = Node(id=nid, x=base.x, y=base.y, z=base.z + float(height))
        project.nodes.append(top)
        top_of[b] = nid
        nid += 1

    aid = project.next_area_id()
    new_ids: list[int] = []
    for a, b in zip(ordered, ordered[1:]):
        # quad CCW loop bottom A->B, top B'->A'
        panel = Area(id=aid, nodes=[a, b, top_of[b], top_of[a]],
                     shell_section=int(shell_section), material=int(material),
                     mesh=tuple(mesh), role="wall", pier=pier,
                     spandrel=spandrel)
        project.areas.append(panel)
        new_ids.append(aid)
        aid += 1
    return new_ids


# --------------------------------------------------------- plan-based drawing (W7)
def _coord_index(project, tol=1e-6):
    """A coordinate → node-id map + a factory that reuses a coincident node or
    creates one, so stacked/adjacent walls share their joints."""
    q = max(tol, 1e-9)

    def key(x, y, z):
        return (round(x / q), round(y / q), round(z / q))

    index = {key(n.x, n.y, n.z): n.id for n in project.nodes}
    counter = [max((n.id for n in project.nodes), default=0) + 1]

    def node_at(x, y, z):
        k = key(x, y, z)
        nid = index.get(k)
        if nid is None:
            nid = counter[0]
            counter[0] += 1
            index[k] = nid
            project.nodes.append(Node(id=nid, x=float(x), y=float(y),
                                      z=float(z)))
        return nid

    return node_at


def build_wall_between(project, p1, p2, top_elev, bottom_elev, shell_section,
                       material, mesh=(2, 2), pier=None, spandrel=None,
                       node_at=None):
    """Build one wall panel spanning ``bottom_elev``→``top_elev`` between plan
    points ``p1``/``p2`` (each an ``(x, y)``) — the ETABS plan-draw idiom (wall
    plan W7). The panel is the quad ``[A, B, B', A']`` (A,B at the bottom, A',B'
    at the top), same convention as :func:`build_wall_line`. Nodes are reused
    via ``node_at`` (a coincident-node factory) so a stack shares its floor
    joints. Returns the new area id, or ``None`` if the points coincide or the
    span is zero."""
    if abs(float(top_elev) - float(bottom_elev)) < 1e-9:
        return None
    if (abs(p1[0] - p2[0]) < 1e-9) and (abs(p1[1] - p2[1]) < 1e-9):
        return None
    na = node_at or _coord_index(project)
    a = na(p1[0], p1[1], bottom_elev)
    b = na(p2[0], p2[1], bottom_elev)
    bt = na(p2[0], p2[1], top_elev)
    at = na(p1[0], p1[1], top_elev)
    aid = project.next_area_id()
    project.areas.append(Area(id=aid, nodes=[a, b, bt, at],
                              shell_section=int(shell_section),
                              material=int(material), mesh=tuple(mesh),
                              role="wall", pier=pier, spandrel=spandrel))
    return aid


def build_wall_stack(project, p1, p2, levels, shell_section, material,
                     mesh=(2, 2), pier=None, spandrel=None):
    """Build a wall on every level in ``levels`` (each a ``(top_elev,
    bottom_elev)``) between the same plan points ``p1``/``p2`` — the story-scope
    draw (One / Similar / All, wall plan W7). All panels share one coincident-
    node factory so the stack is continuous. Returns the list of new area ids."""
    na = _coord_index(project)
    ids = []
    for top, bottom in levels:
        aid = build_wall_between(project, p1, p2, top, bottom, shell_section,
                                 material, mesh=mesh, pier=pier,
                                 spandrel=spandrel, node_at=na)
        if aid is not None:
            ids.append(aid)
    return ids
