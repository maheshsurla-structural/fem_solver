"""Parametric structure generators — build a whole Project from a few
parameters, so large models are not drawn node-by-node.

Pure functions returning a ``Project`` (no Qt). ``frame`` covers regular
building frames in 2-D and 3-D; more templates (grillage decks, girder
lines for bridges) plug in the same way.
"""
from __future__ import annotations

from project import Load, Material, Member, Node, Project, Section


def _section_from_shape(shape, sid=1):
    """A Section carrying full properties for the given AISC shape."""
    try:
        from femsolver.design.steel.sections import get_section
        ss = get_section(shape.replace("X", "x"))
        return Section(id=sid, name=shape, A=ss.A, Iz=ss.Ix, shape=shape,
                       Iy=ss.Iy, J=ss.J)
    except Exception:
        return Section(id=sid, name=shape or "Default", A=0.012, Iz=2.0e-4,
                       shape=shape or "")


def frame(bays_x=3, bay_x=6.0, storeys=3, storey_h=3.5,
          bays_y=0, bay_y=6.0, shape="W12x65", fixed_base=True) -> Project:
    """A regular moment frame. 2-D when ``bays_y < 1``, else a 3-D space frame
    (columns at every grid point, beams along X — and Y in 3-D — at each floor;
    the base level is fixed)."""
    three_d = bays_y >= 1
    ndm, ndf = (3, 6) if three_d else (2, 3)
    name = (f"Frame {bays_x}x{bays_y}x{storeys}" if three_d
            else f"Frame {bays_x}x{storeys}")
    p = Project(name=name, ndm=ndm, ndf=ndf)
    p.materials.append(Material(id=1, name="A992", E=200e9, nu=0.3, rho=7850.0))
    p.sections.append(_section_from_shape(shape))

    nx = bays_x + 1
    ny = (bays_y + 1) if three_d else 1
    fix = (1, 1, 1, 1, 1, 1) if three_d else (1, 1, 1)
    ids, tag = {}, 1
    for k in range(storeys + 1):
        z = k * storey_h
        for j in range(ny):
            for i in range(nx):
                sup = fix if (k == 0 and fixed_base) else ()
                if three_d:
                    p.nodes.append(Node(id=tag, x=i * bay_x, y=j * bay_y, z=z,
                                        supports=sup))
                else:
                    p.nodes.append(Node(id=tag, x=i * bay_x, y=z, supports=sup))
                ids[(i, j, k)] = tag
                tag += 1

    etag = 1
    for k in range(storeys):                          # columns
        for j in range(ny):
            for i in range(nx):
                p.members.append(Member(id=etag, n1=ids[(i, j, k)],
                                        n2=ids[(i, j, k + 1)],
                                        section=1, material=1))
                etag += 1
    for k in range(1, storeys + 1):                   # beams at each floor
        for j in range(ny):
            for i in range(bays_x):                    # X-direction
                p.members.append(Member(id=etag, n1=ids[(i, j, k)],
                                        n2=ids[(i + 1, j, k)],
                                        section=1, material=1))
                etag += 1
        if three_d:
            for j in range(bays_y):                    # Y-direction
                for i in range(nx):
                    p.members.append(Member(id=etag, n1=ids[(i, j, k)],
                                            n2=ids[(i, j + 1, k)],
                                            section=1, material=1))
                    etag += 1
    return p


# ---------------------------------------------------------------- load patterns

def _vertical_index(ndm: int) -> int:
    return 1 if ndm == 2 else 2            # Fy in 2-D, Fz in 3-D


def _elev(project, node) -> float:
    return node.y if project.ndm == 2 else node.z


def gravity_loads(project, p_node: float):
    """A downward point load of magnitude ``p_node`` (N) at every node above
    the base level."""
    vidx = _vertical_index(project.ndm)
    base = min((_elev(project, n) for n in project.nodes), default=0.0)
    loads = []
    for n in project.nodes:
        if _elev(project, n) <= base + 1e-9:
            continue
        vals = [0.0] * project.ndf
        vals[vidx] = -abs(p_node)
        loads.append(Load(node=n.id, values=tuple(vals)))
    return loads


def lateral_loads(project, base_shear: float, direction: str = "X"):
    """Storey lateral forces summing to ``base_shear`` (N), distributed
    linearly with height (seismic-style) and split among each floor's nodes."""
    hidx = 0 if direction == "X" else 1
    base = min((_elev(project, n) for n in project.nodes), default=0.0)
    levels: dict = {}
    for n in project.nodes:
        e = round(_elev(project, n), 6)
        if e <= base + 1e-9:
            continue
        levels.setdefault(e, []).append(n.id)
    total_h = sum(e - base for e in levels)
    loads = []
    for e, node_ids in levels.items():
        f_floor = (base_shear * (e - base) / total_h if total_h > 0
                   else base_shear / len(levels))
        per_node = f_floor / len(node_ids)
        for nid in node_ids:
            vals = [0.0] * project.ndf
            vals[hidx] = per_node
            loads.append(Load(node=nid, values=tuple(vals)))
    return loads
