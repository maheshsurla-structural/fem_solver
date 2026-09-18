"""Similar-story replication (wall plan W4c).

Copies the model objects (nodes, members, areas/walls) of one *source* story up
to one or more *target* stories — the ETABS "similar stories" idiom, where you
draw a plan once and it repeats up the building. Objects are selected by
elevation (a node belongs to the source band; a member/area belongs when all of
its nodes do) and copied translated in Z by ``target.elev − source.elev``.

New nodes that land on an existing node (a shared column line, the story below's
top) are **merged** so the stories stay connected rather than accumulating
duplicate joints. Wall pier/spandrel labels copy unchanged, so a labelled pier
runs up the building as one pier. Supports are not replicated (they belong at
the base).

Pure — mutates the ``Project`` in place, wrapped by the window's undoable edit.
"""
from __future__ import annotations

from project import Area, Member, Node

_TOL = 1e-6


def _story_band(story):
    """(z_lo, z_hi] elevation band a story owns: floor-to-floor."""
    return story.elev - story.height, story.elev


def objects_in_band(project, z_lo, z_hi, tol=1e-6):
    """``(node_ids, member_ids, area_ids)`` whose geometry lies in the elevation
    band ``[z_lo − tol, z_hi + tol]`` — a node by its z, a member/area when all
    of its nodes qualify."""
    zof = {n.id: n.z for n in project.nodes}

    def _in(nid):
        z = zof.get(nid)
        return z is not None and (z_lo - tol) <= z <= (z_hi + tol)

    node_ids = {n.id for n in project.nodes if _in(n.id)}
    member_ids = {m.id for m in project.members
                  if m.n1 in node_ids and m.n2 in node_ids}
    area_ids = {a.id for a in project.areas
                if a.nodes and all(n in node_ids for n in a.nodes)}
    return node_ids, member_ids, area_ids


def _node_key(x, y, z, tol):
    """A rounded coordinate key for coincident-node merging."""
    q = max(tol, 1e-9)
    return (round(x / q), round(y / q), round(z / q))


def replicate_story(project, source_id, target_ids, tol=1e-6) -> dict:
    """Replicate the source story's nodes/members/areas to each target story.

    Returns a summary dict ``{"nodes", "members", "areas"}`` of how many new
    objects were created (merged nodes are not counted). No-op (zeros) if the
    source has no objects or ids are unknown."""
    by_id = {s.id: s for s in project.stories}
    source = by_id.get(source_id)
    if source is None:
        return {"nodes": 0, "members": 0, "areas": 0}
    z_lo, z_hi = _story_band(source)
    node_ids, member_ids, area_ids = objects_in_band(project, z_lo, z_hi, tol)
    if not node_ids:
        return {"nodes": 0, "members": 0, "areas": 0}

    nodes = {n.id: n for n in project.nodes}
    members = {m.id: m for m in project.members}
    areas = {a.id: a for a in project.areas}

    # index existing nodes by coordinate so copies merge onto them
    coord_index = {_node_key(n.x, n.y, n.z, tol): n.id for n in project.nodes}
    next_nid = max((n.id for n in project.nodes), default=0) + 1
    next_mid = max((m.id for m in project.members), default=0) + 1
    next_aid = project.next_area_id()

    made = {"nodes": 0, "members": 0, "areas": 0}
    for tid in target_ids:
        target = by_id.get(tid)
        if target is None or tid == source_id:
            continue
        dz = target.elev - source.elev
        if abs(dz) < tol:
            continue
        remap = {}
        for nid in sorted(node_ids):
            s = nodes[nid]
            x, y, z = s.x, s.y, s.z + dz
            key = _node_key(x, y, z, tol)
            existing = coord_index.get(key)
            if existing is not None:                 # merge onto the shared node
                remap[nid] = existing
                continue
            project.nodes.append(Node(id=next_nid, x=x, y=y, z=z))
            coord_index[key] = next_nid
            remap[nid] = next_nid
            next_nid += 1
            made["nodes"] += 1
        for mid in sorted(member_ids):
            m = members[mid]
            project.members.append(Member(
                id=next_mid, n1=remap[m.n1], n2=remap[m.n2],
                section=m.section, material=m.material, kind=m.kind))
            next_mid += 1
            made["members"] += 1
        for aid in sorted(area_ids):
            a = areas[aid]
            project.areas.append(Area(
                id=next_aid, nodes=[remap[n] for n in a.nodes],
                shell_section=a.shell_section, material=a.material,
                mesh=a.mesh, local_axis=a.local_axis, role=a.role,
                pier=a.pier, spandrel=a.spandrel))
            next_aid += 1
            made["areas"] += 1
    return made
