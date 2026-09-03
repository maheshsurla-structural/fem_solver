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
    p.materials.append(Material(id=1, name="Steel", E=200e9, nu=0.3))
    p.sections.append(Section(id=1, name="Default", A=6.0e-3, Iz=2.0e-4))

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
