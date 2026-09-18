"""Pier force integration (wall plan W2).

The headline ETABS wall output: the internal **P, V, M** carried by a wall
pier at each level, recovered from the meshed shell membrane stresses.

A *pier* is the set of wall areas (``Area.role == "wall"``) sharing a
``pier`` label (wall plan W0). This module cuts that pier horizontally at a
series of elevations and integrates the shell membrane force across the cut:

* **P** — vertical axial force (∫ of the vertical membrane stress ``σ_vv``
  across the cut width), positive in **tension**.
* **V** — in-plane horizontal shear (∫ of the in-plane shear stress across the
  cut).
* **M** — in-plane bending moment about the pier's centroidal vertical axis
  (∫ σ_vv · lever-arm).

The recovered element resultants (``model_geometry._element_resultant``) are the
membrane force/length ``[N11, N22, N12, ...]`` in each element's **local** axes,
so we rebuild the membrane tensor in **global** axes via the element's local
frame (``model_geometry._area_frame``) and take the traction on the horizontal
cut (normal = global +Z). This is exact for a planar vertical wall meshed with
structured quads and satisfies the free-body equilibrium check (a base cut
returns the total applied load above it).

The integration is **element-wise**: at cut elevation ``z0`` every wall element
whose vertical span straddles ``z0`` contributes its (constant, over a vertical-
sided quad) membrane stress times the element's horizontal width. The default
cut elevations are the mid-heights of each mesh row, giving a P/V/M profile up
the pier without needing a Story object (that is wall plan W4).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import model_geometry as mg
from model_geometry import _area_frame, _decode_area_id, _element_resultant, to_xyz


@dataclass
class PierForce:
    """Integrated pier force at one cut elevation (project/SI units)."""
    z: float                # cut elevation (global Z)
    axial: float            # P, vertical, + tension  (N)
    shear: float            # V, in-plane horizontal   (N)
    moment: float           # M about pier centroid    (N·m)
    width: float            # horizontal length of wall crossed by the cut (m)


def pier_area_ids(project, pier: str) -> set:
    """Ids of the wall areas grouped under ``pier`` (wall plan W0)."""
    return {a.id for a in project.areas
            if a.role == "wall" and a.pier == pier}


def _pier_elements(project, model, pier: str) -> list:
    """``(tag, element, node_coords)`` for every meshed shell element belonging
    to the pier's wall areas in the solved ``model``."""
    ids = pier_area_ids(project, pier)
    out = []
    for tag, e in model.elements.items():
        nt = getattr(e, "node_tags", ())
        if len(nt) not in (3, 4):
            continue
        if _decode_area_id(tag) not in ids:
            continue
        try:
            pts = np.array([to_xyz(model.nodes[t].coords) for t in nt], float)
        except KeyError:
            continue
        out.append((tag, e, pts))
    return out


def _horizontal_axis(all_pts: np.ndarray) -> np.ndarray:
    """Unit horizontal in-plane axis ``ĥ`` of the wall — the dominant direction
    of the pier nodes' plan (x, y) spread (a straight wall runs along it).
    Returned as a 3-vector with zero Z."""
    xy = all_pts[:, :2] - all_pts[:, :2].mean(axis=0)
    # principal direction via the 2×2 covariance's top eigenvector
    cov = xy.T @ xy
    w, v = np.linalg.eigh(cov)
    h2 = v[:, int(np.argmax(w))]
    n = np.linalg.norm(h2)
    if n < 1e-12:
        h2 = np.array([1.0, 0.0])
    else:
        h2 = h2 / n
    return np.array([h2[0], h2[1], 0.0])


def _membrane_traction_on_horizontal_cut(res, pts):
    """(σ_vv, σ_hv-parts) for one element: return the global membrane traction
    vector ``t = N_glob · ẑ`` (N/m) on a horizontal cut, or ``None``.

    ``res`` is the element's local 8-vector; ``pts`` its node coords."""
    if res is None:
        return None
    frame = _area_frame(pts)
    if frame is None:
        return None
    _c, e1, e2, _e3 = frame
    n11, n22, n12 = float(res[0]), float(res[1]), float(res[2])
    # membrane resultant as a global (3×3) symmetric tensor
    N = (n11 * np.outer(e1, e1) + n22 * np.outer(e2, e2)
         + n12 * (np.outer(e1, e2) + np.outer(e2, e1)))
    zhat = np.array([0.0, 0.0, 1.0])
    return N @ zhat                       # traction on the horizontal cut (N/m)


def pier_cut_elevations(project, model, pier: str) -> list:
    """Default cut elevations: the mid-height of each distinct mesh row of the
    pier (so each cut cleanly crosses one row of elements)."""
    els = _pier_elements(project, model, pier)
    if not els:
        return []
    zs = sorted({round(float(z), 9)
                 for _t, _e, pts in els for z in pts[:, 2]})
    return [0.5 * (a + b) for a, b in zip(zs, zs[1:])]


def pier_forces(project, model, pier: str, elevations=None) -> list:
    """Integrated :class:`PierForce` at each cut elevation up the pier.

    ``model`` must be solved (element resultants recovered) — otherwise every
    force is zero. ``elevations`` defaults to :func:`pier_cut_elevations`."""
    els = _pier_elements(project, model, pier)
    if not els:
        return []
    all_pts = np.vstack([pts for _t, _e, pts in els])
    hhat = _horizontal_axis(all_pts)
    # pier centroid along ĥ (its vertical centroidal axis) — moment reference
    h_all = all_pts @ hhat
    h_c = float(0.5 * (h_all.min() + h_all.max()))
    zhat = np.array([0.0, 0.0, 1.0])

    if elevations is None:
        elevations = pier_cut_elevations(project, model, pier)

    out = []
    for z0 in elevations:
        P = V = M = width = 0.0
        for _t, e, pts in els:
            z_lo, z_hi = float(pts[:, 2].min()), float(pts[:, 2].max())
            if not (z_lo < z0 < z_hi):        # element not crossed by this cut
                continue
            t = _membrane_traction_on_horizontal_cut(_element_resultant(e), pts)
            if t is None:
                continue
            h = pts @ hhat
            w = float(h.max() - h.min())      # element width along the wall
            if w <= 0:
                continue
            sigma_vv = float(zhat @ t)         # vertical membrane stress (N/m)
            tau = float(hhat @ t)              # in-plane horizontal shear (N/m)
            h_mid = float(0.5 * (h.min() + h.max()))
            P += sigma_vv * w
            V += tau * w
            M += sigma_vv * w * (h_mid - h_c)
            width += w
        out.append(PierForce(z=float(z0), axial=P, shear=V, moment=M,
                             width=width))
    return out
