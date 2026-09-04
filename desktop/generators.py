"""Parametric structure generators — build a whole Project from a few
parameters, so large models are not drawn node-by-node.

Pure functions returning a ``Project`` (no Qt). ``frame`` covers regular
building frames in 2-D and 3-D; more templates (grillage decks, girder
lines for bridges) plug in the same way.
"""
from __future__ import annotations

from project import Material, Member, Node, Project, Section


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
    p.materials.append(Material(id=1, name="A992", E=200e9, nu=0.3))
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
