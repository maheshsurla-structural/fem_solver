"""A real project to show on first launch (until you open one from disk).

Returns a ``Project`` (the information-model document), not a raw solver
model — so the app's open / save / compile path is exercised from the very
first frame.
"""
from __future__ import annotations

from project import Load, Material, Member, Node, Project, Section


def demo_project(bays: int = 2, storeys: int = 2,
                 bay: float = 4.0, storey: float = 3.0) -> Project:
    """A 2-D moment frame (fixed base) with an inverted-triangle lateral load."""
    p = Project(name="Demo portal frame", ndm=2, ndf=3)
    p.materials.append(Material(id=1, name="A992", E=200e9, nu=0.3, rho=7850.0))
    p.sections.append(Section(id=1, name="W12x65", A=0.012323, Iz=2.2185e-4,
                              shape="W12x65"))

    ids: dict[tuple[int, int], int] = {}
    tag = 1
    for j in range(storeys + 1):
        for i in range(bays + 1):
            supports = (1, 1, 1) if j == 0 else ()
            p.nodes.append(Node(id=tag, x=i * bay, y=j * storey, supports=supports))
            ids[(i, j)] = tag
            tag += 1

    etag = 1
    for i in range(bays + 1):                    # columns
        for j in range(storeys):
            p.members.append(Member(id=etag, n1=ids[(i, j)], n2=ids[(i, j + 1)],
                                    section=1, material=1))
            etag += 1
    for j in range(1, storeys + 1):              # beams
        for i in range(bays):
            p.members.append(Member(id=etag, n1=ids[(i, j)], n2=ids[(i + 1, j)],
                                    section=1, material=1))
            etag += 1

    Fx = 100e3                                   # inverted-triangle lateral load
    for j in range(1, storeys + 1):
        p.loads.append(Load(node=ids[(0, j)], values=(Fx * j / storeys, 0.0, 0.0)))
    return p


def demo_project_3d(L: float = 5.0, H: float = 4.0) -> Project:
    """A single-storey 3-D moment frame (fixed base, box roof) with a lateral
    load in +x — exercises 3-D build / solve / design."""
    p = Project(name="Demo 3-D frame", ndm=3, ndf=6)
    p.materials.append(Material(id=1, name="A992", E=200e9, nu=0.3, rho=7850.0))
    p.sections.append(Section(id=1, name="W12x65", A=0.012323, Iz=2.2185e-4,
                              shape="W12x65"))
    corners = [(0.0, 0.0), (L, 0.0), (L, L), (0.0, L)]
    tag, base_ids, top_ids = 1, [], []
    for x, y in corners:
        p.nodes.append(Node(id=tag, x=x, y=y, z=0.0, supports=(1, 1, 1, 1, 1, 1)))
        base_ids.append(tag)
        tag += 1
    for x, y in corners:
        p.nodes.append(Node(id=tag, x=x, y=y, z=H))
        top_ids.append(tag)
        tag += 1
    etag = 1
    for b, t in zip(base_ids, top_ids):          # columns
        p.members.append(Member(id=etag, n1=b, n2=t, section=1, material=1))
        etag += 1
    for k in range(4):                           # roof beams (closed loop)
        p.members.append(Member(id=etag, n1=top_ids[k],
                                n2=top_ids[(k + 1) % 4], section=1, material=1))
        etag += 1
    for t in top_ids:                            # lateral load in +x at the roof
        p.loads.append(Load(node=t, values=(25e3, 0.0, 0.0, 0.0, 0.0, 0.0)))
    return p
