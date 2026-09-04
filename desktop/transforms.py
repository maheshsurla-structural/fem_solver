"""Geometry transforms on a selection — the manual-editing counterpart to the
parametric generators. Each mutates the Project in place (wrapped by the
window's undoable ``_apply_edit``); copy/array create new nodes + members.
"""
from __future__ import annotations

from project import Member, Node


def move_nodes(project, node_ids, dx, dy, dz) -> None:
    """Translate the given nodes by (dx, dy, dz) — members follow their ends."""
    ids = set(node_ids)
    for n in project.nodes:
        if n.id in ids:
            n.x += dx
            n.y += dy
            if project.ndm == 3:
                n.z += dz


def copy_selection(project, node_ids, member_ids, dx, dy, dz, count) -> None:
    """Array ``count`` copies of the selected nodes and members, each offset by
    k*(dx, dy, dz). Selected members drag their end nodes along; new nodes and
    members get fresh ids and copied section/material/supports."""
    nodes = {n.id: n for n in project.nodes}
    members = {m.id: m for m in project.members}
    node_set = set(node_ids)
    for mid in member_ids:
        m = members.get(mid)
        if m:
            node_set.update((m.n1, m.n2))

    next_nid = max((n.id for n in project.nodes), default=0) + 1
    next_mid = max((m.id for m in project.members), default=0) + 1
    for k in range(1, int(count) + 1):
        ox, oy, oz = k * dx, k * dy, k * dz
        remap = {}
        for nid in sorted(node_set):
            s = nodes[nid]
            project.nodes.append(Node(
                id=next_nid, x=s.x + ox, y=s.y + oy,
                z=(s.z + oz) if project.ndm == 3 else 0.0,
                supports=tuple(s.supports)))
            remap[nid] = next_nid
            next_nid += 1
        for mid in member_ids:
            m = members.get(mid)
            if m and m.n1 in remap and m.n2 in remap:
                project.members.append(Member(
                    id=next_mid, n1=remap[m.n1], n2=remap[m.n2],
                    section=m.section, material=m.material, kind=m.kind))
                next_mid += 1
