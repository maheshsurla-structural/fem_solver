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
