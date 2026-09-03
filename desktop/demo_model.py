"""A real femsolver model to show on first launch (until file open exists).

Built through the public API (``Model`` + ``BeamColumn2D`` + material) so the
viewport is proven against genuine engine objects, not mock geometry.
"""
from __future__ import annotations

from femsolver import BeamColumn2D, ElasticIsotropic, Model


def portal_frame(bays: int = 2, storeys: int = 2,
                 bay: float = 4.0, storey: float = 3.0) -> Model:
    """A 2-D moment frame (fixed base), ``bays`` wide and ``storeys`` tall."""
    E, A, Iz = 200e9, 6.0e-3, 2.0e-4
    m = Model(ndm=2, ndf=3)
    mat = ElasticIsotropic(1, E=E, nu=0.3)
    m.add_material(mat)

    ids: dict[tuple[int, int], int] = {}
    tag = 1
    for j in range(storeys + 1):                 # row 0 = base
        for i in range(bays + 1):
            m.add_node(tag, i * bay, j * storey)
            ids[(i, j)] = tag
            tag += 1

    etag = 1
    for i in range(bays + 1):                    # columns
        for j in range(storeys):
            m.add_element(BeamColumn2D(etag, (ids[(i, j)], ids[(i, j + 1)]),
                                       mat, A, Iz))
            etag += 1
    for j in range(1, storeys + 1):              # beams
        for i in range(bays):
            m.add_element(BeamColumn2D(etag, (ids[(i, j)], ids[(i + 1, j)]),
                                       mat, A, Iz))
            etag += 1

    for i in range(bays + 1):                    # fix the base row
        m.fix(ids[(i, 0)], [1, 1, 1])

    Fx = 100e3                                   # lateral (pushover) load pattern:
    for j in range(1, storeys + 1):              # inverted triangle up the height
        m.add_nodal_load(ids[(0, j)], [Fx * j / storeys, 0.0, 0.0])
    return m
