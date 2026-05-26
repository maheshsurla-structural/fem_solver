"""Base-isolation device macros.

Helper factories that build pre-configured :class:`ZeroLengthElement`
instances for common seismic isolators. Each macro encapsulates the
bilinear / EPP backbone with parameters matched to product literature
or design code:

* :func:`lead_rubber_bearing` -- LRB / LDR. Bilinear horizontal
  spring (initial K1, post-yield K2, characteristic strength Q) with
  a high (rigid) vertical spring. The standard "Bouc-Wen-like"
  bilinear simplification used for design.
* :func:`friction_pendulum` -- FPS isolator. Bilinear horizontal
  spring with friction force ``mu * W`` and post-yield restoring
  stiffness ``W / R`` from pendulum geometry, with rigid vertical.

These return a ``ZeroLengthElement`` ready to add to a model with
``model.add_element(elem)``.
"""
from __future__ import annotations

from femsolver.elements.zero_length import ZeroLengthElement
from femsolver.materials.uniaxial.bilinear import UniaxialBilinear
from femsolver.materials.uniaxial.elastic import UniaxialElastic


def lead_rubber_bearing(
    tag: int,
    nodes,
    *,
    K1: float,
    K2: float,
    Q: float,
    K_vertical: float | None = None,
    dofs_per_node: int = 6,
    direction: int = 0,
) -> ZeroLengthElement:
    """Lead-rubber bearing (LRB) base isolator.

    The horizontal force-displacement relation is bilinear with
    initial stiffness ``K1``, characteristic strength ``Q`` (the
    intercept of the post-yield line at zero displacement), and
    post-yield slope ``K2``. The yield displacement and force are:

        d_y = Q / (K1 - K2)
        F_y = K1 * d_y

    With the standard hardening ratio ``b = K2 / K1``.

    Parameters
    ----------
    tag : int
    nodes : (i, j) two coincident node tags
    K1 : float
        Initial (elastic) horizontal stiffness.
    K2 : float
        Post-yield horizontal stiffness (``< K1``).
    Q : float
        Characteristic strength (yield-line intercept at zero
        displacement). ``Q > 0``.
    K_vertical : float, optional
        Vertical stiffness in the gravity direction (typically very
        high). Defaults to ``1000 * K1``. Vertical DOF is taken as
        DOF index 2 (z) for 3D models or 1 (y) for 2D.
    dofs_per_node : int, default 6
    direction : int, default 0
        DOF index in which the bilinear horizontal spring acts
        (0=ux, 1=uy, 2=uz).
    """
    if K1 <= 0.0:
        raise ValueError(f"K1 must be positive, got {K1}")
    if not (0.0 <= K2 < K1):
        raise ValueError(
            f"K2 must satisfy 0 <= K2 < K1, got K2={K2}, K1={K1}"
        )
    if Q <= 0.0:
        raise ValueError(f"Q must be positive, got {Q}")
    if K_vertical is None:
        K_vertical = 1000.0 * K1
    # Bilinear with kinematic hardening
    d_y = Q / (K1 - K2)
    F_y = K1 * d_y
    b = K2 / K1
    h_mat = UniaxialBilinear(E=K1, sigma_y=F_y, b=b)
    v_mat = UniaxialElastic(E=K_vertical)
    # Vertical direction: DOF 2 in 3D models with ndf=6; for 2D
    # frames (ndf=3) the vertical is DOF 1 (uy).
    vert_dof = 2 if dofs_per_node >= 4 else 1
    materials = {direction: h_mat}
    if vert_dof != direction:
        materials[vert_dof] = v_mat
    return ZeroLengthElement(
        tag, nodes, materials=materials, dofs_per_node=dofs_per_node,
    )


def friction_pendulum(
    tag: int,
    nodes,
    *,
    mu: float,
    R: float,
    W: float,
    K_vertical: float | None = None,
    K_initial_factor: float = 100.0,
    dofs_per_node: int = 6,
    direction: int = 0,
) -> ZeroLengthElement:
    """Friction-pendulum (FPS) isolator.

    Bilinear horizontal behaviour with:

    * Friction force at zero displacement: ``F_f = mu * W``
    * Post-yield restoring slope: ``K2 = W / R``  (pendulum geometry)
    * Initial (pre-slip) stiffness: ``K1 = K_initial_factor * K2``

    The "yield" displacement is the small slip distance at which the
    friction force is reached; with ``K_initial_factor = 100`` the
    behaviour is essentially rigid-perfectly-plastic up to the
    friction force.

    Parameters
    ----------
    mu : float
        Friction coefficient (e.g. 0.05 for typical FPS bearings).
    R : float
        Pendulum radius (e.g. 1-4 m for buildings).
    W : float
        Vertical load (axial force on the isolator), used to compute
        F_f and K2.
    K_vertical : float, optional
        Vertical stiffness. Defaults to ``1000 * K2``.
    K_initial_factor : float, default 100
        Ratio of pre-slip to post-slip stiffness.
    """
    if mu <= 0.0:
        raise ValueError(f"mu must be positive, got {mu}")
    if R <= 0.0:
        raise ValueError(f"R must be positive, got {R}")
    if W <= 0.0:
        raise ValueError(f"W must be positive (vertical load), got {W}")
    K2 = W / R
    F_f = mu * W
    K1 = K_initial_factor * K2
    if K_vertical is None:
        K_vertical = 1000.0 * K2
    # Bilinear backbone with kinematic hardening; yield = friction force.
    # b = K2 / K1
    b = K2 / K1
    h_mat = UniaxialBilinear(E=K1, sigma_y=F_f, b=b)
    v_mat = UniaxialElastic(E=K_vertical)
    vert_dof = 2 if dofs_per_node >= 4 else 1
    materials = {direction: h_mat}
    if vert_dof != direction:
        materials[vert_dof] = v_mat
    return ZeroLengthElement(
        tag, nodes, materials=materials, dofs_per_node=dofs_per_node,
    )
