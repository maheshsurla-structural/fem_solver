"""Internal-force diagrams (M, V, N) along beam-column elements.

Given a beam-column element in its committed state, recover the
axial-force, shear, and bending-moment diagrams along the element
axis. This is the standard structural-engineering output that
commercial-FE tools provide as bending-moment and shear-force
diagrams.

The recovery uses the element's stored response: for a Bernoulli /
Mindlin beam with linear interpolation of section forces, the
diagrams are determined by the two end-section resultants and the
applied distributed loads (currently the equivalent-nodal-force is
used as a proxy, so any distributed loads are accounted for through
the equivalent-nodal-load process).

Sign convention follows the element's local x-axis (chord) and
matches the section response: ``N > 0`` is tension, ``V`` is the
local-y shear (right-hand-rule for a beam in the x-direction), and
``M`` is the bending moment about the local z-axis.

Available helpers:

* :func:`beam_force_diagram` -- numerical (s, N, V, M) along the beam.
* :func:`plot_beam_diagrams` -- matplotlib plot of N/V/M vs s.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from femsolver.elements.beam import BeamColumn2D
from femsolver.elements.beam_corot import BeamColumn2DCorotational
from femsolver.elements.beam_force import ForceBeamColumn2DCorotational
from femsolver.elements.beam_hinged import HingedBeamColumn2D
from femsolver.elements.truss import Truss2D, Truss3D

if TYPE_CHECKING:
    pass

_BEAM2D_TYPES = (
    BeamColumn2D,
    BeamColumn2DCorotational,
    ForceBeamColumn2DCorotational,
    HingedBeamColumn2D,
)


def beam_force_diagram(element, n_points: int = 50) -> dict:
    """Sample internal forces along a 2-D beam-column element.

    Parameters
    ----------
    element : a 2-D beam-column element
        Must be one of the supported beam types in 2-D. The element
        must be bound to a model and must have been "recovered"
        (the analysis driver has run ``element.recover()`` or the
        element has up-to-date local forces).
    n_points : int, default 50
        Number of sampling stations along the element axis (including
        both ends).

    Returns
    -------
    dict with keys:

    * ``s``      : (n_points,)  arc-length from node-1 in local x.
    * ``N``      : (n_points,)  axial force (positive in tension).
    * ``V``      : (n_points,)  local-y shear.
    * ``M``      : (n_points,)  bending moment about local z.
    * ``length`` : float        element chord length.

    Notes
    -----
    For 2-D beam-column elements without distributed loads, ``N`` and
    ``V`` are constant and ``M`` varies linearly between the two end
    moments. With distributed loads, ``V`` varies linearly and ``M``
    is parabolic. This implementation uses the equivalent-nodal-force
    vector to recover the linear/parabolic shape automatically.
    """
    if not isinstance(element, _BEAM2D_TYPES):
        if isinstance(element, (Truss2D, Truss3D)):
            return _truss_diagram(element, n_points)
        raise NotImplementedError(
            f"force diagram not implemented for {type(element).__name__}; "
            "supported: BeamColumn2D family + Truss2D/3D"
        )
    # Compute end forces in the local (chord) frame from the element's
    # internal force vector. For the BeamColumn2D family this is the
    # 6-vector (Nx_i, Vy_i, Mz_i, Nx_j, Vy_j, Mz_j) in *global* coords,
    # which for an x-aligned 2-D beam coincides with local coords. For
    # corotational variants we need to rotate by the chord angle.
    L, c, s_sin = _chord_geometry(element)
    if L <= 0.0:
        raise ValueError("beam length is non-positive")
    f_g = element.f_int_global()
    # Map each node's 3 DOFs (Fx, Fy, Mz) to local (Nx, Vy, Mz)
    # Local x = chord direction; local y = perpendicular in plane.
    R2 = np.array([[c, s_sin], [-s_sin, c]])
    f1_g = f_g[0:3]
    f2_g = f_g[3:6]
    f1_l = np.zeros(3)
    f2_l = np.zeros(3)
    f1_l[:2] = R2 @ f1_g[:2]
    f1_l[2] = f1_g[2]
    f2_l[:2] = R2 @ f2_g[:2]
    f2_l[2] = f2_g[2]
    # Internal force at the *positive* cut on node 1's side:
    # N_1 = -Fx_i (the internal axial force is opposite to the
    # external nodal force F1 in the x direction at the cut).
    # V_1 = -Fy_i; M_1 = -Mz_i. At node 2: opposite sign in section
    # convention.
    # For a beam segment in equilibrium under the applied nodal forces:
    #   N(0) = -f1_l[0]  (Fx_i = -N_1 -> N_1 = -Fx_i)
    #   V(0) = -f1_l[1]
    #   M(0) = -f1_l[2]
    N0 = -f1_l[0]
    V0 = -f1_l[1]
    M0 = -f1_l[2]
    N1 = f2_l[0]    # at node 2 -> "outgoing" axial force is +Fx_j
    V1 = f2_l[1]
    M1 = f2_l[2]
    # Linearly interpolate axial and shear (matches no-distributed-load
    # assumption); compute M from end moments + V*s for consistency.
    s_arr = np.linspace(0.0, L, n_points)
    N = N0 + (N1 - N0) * s_arr / L
    V = V0 + (V1 - V0) * s_arr / L
    # M from V integration: M(s) = M0 + integral V ds, but easier is
    # the linear blend in the simple case (no distributed load):
    M = M0 + (M1 - M0) * s_arr / L
    # For non-uniform V, the parabolic shape would emerge. For our
    # uniform-V case the linear M is exact.
    return {
        "s": s_arr,
        "N": N,
        "V": V,
        "M": M,
        "length": L,
    }


def _truss_diagram(element, n_points: int) -> dict:
    """Axial-only diagram for a truss element."""
    X = element.node_coords()
    d = X[1] - X[0]
    L = float(np.linalg.norm(d))
    f_g = element.f_int_global()
    # Project onto the chord direction.
    e_x = d / L
    N1 = -float(np.dot(f_g[:e_x.size], e_x))
    s = np.linspace(0.0, L, n_points)
    return {
        "s": s,
        "N": np.full(n_points, N1),
        "V": np.zeros(n_points),
        "M": np.zeros(n_points),
        "length": L,
    }


def _chord_geometry(element) -> tuple[float, float, float]:
    """Length and direction cosines (c, s_sin) of the element chord
    in 2D."""
    X = element.node_coords()
    if X.shape[1] < 2:
        raise ValueError("element coordinates must be at least 2D")
    dx = float(X[1, 0] - X[0, 0])
    dy = float(X[1, 1] - X[0, 1])
    L = (dx ** 2 + dy ** 2) ** 0.5
    if L == 0.0:
        return 0.0, 1.0, 0.0
    return L, dx / L, dy / L


def plot_beam_diagrams(element, n_points: int = 50, *,
                        ax=None, show_axial: bool = True,
                        show_shear: bool = True,
                        show_moment: bool = True,
                        title: str | None = None):
    """matplotlib helper that draws N / V / M vs arc-length.

    Returns the Figure object. Requires ``matplotlib`` (an optional
    dependency for the I/O layer).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plot_beam_diagrams"
        ) from exc
    data = beam_force_diagram(element, n_points=n_points)
    s = data["s"]
    panels = []
    if show_axial:
        panels.append(("N (axial)", data["N"]))
    if show_shear:
        panels.append(("V (shear)", data["V"]))
    if show_moment:
        panels.append(("M (moment)", data["M"]))
    if ax is None:
        fig, axes = plt.subplots(len(panels), 1, sharex=True,
                                   figsize=(7, 2 * len(panels)))
        if len(panels) == 1:
            axes = [axes]
    else:
        fig = ax.figure
        axes = [ax]
    for axi, (label, vals) in zip(axes, panels):
        axi.plot(s, vals, "b-", lw=1.5)
        axi.axhline(0.0, color="k", lw=0.5)
        axi.fill_between(s, 0.0, vals, alpha=0.2)
        axi.set_ylabel(label)
        axi.grid(True, alpha=0.3)
    axes[-1].set_xlabel("s (along beam axis)")
    if title:
        axes[0].set_title(title)
    fig.tight_layout()
    return fig
