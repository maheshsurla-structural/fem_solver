"""Krawinkler panel-zone model for steel moment-frame joints.

The column-beam joint panel zone is the rectangular region bounded by
the column flanges (left/right) and beam flanges (top/bottom). When
the frame deforms in sway, this region shears in pure shear, and its
deformation contributes significantly to overall storey drift in
unbraced steel moment frames.

Krawinkler (1978, 1996) proposed a three-parameter bilinear shear
backbone for the panel zone:

* **Elastic shear stiffness** ``K_e = G d_c t_p d_b`` (force / shear
  strain) -- the panel acts as a thin plate in shear.
* **Yield shear force** ``V_y = 0.55 R_y f_y d_c t_p ·
  [1 + 3 b_cf t_cf^2 / (d_b d_c t_p)]`` -- AISC J10 form. The bracket
  term captures the boundary-element stiffening from the column flanges.
* **Post-yield stiffness** ``K_p`` ≈ 0.06 K_e (Krawinkler 1996).

The model is implemented at two levels:

1. :func:`krawinkler_panel_zone` -- bare capacity / stiffness numbers
   useful for hand calc or as inputs to a custom UniaxialMaterial.
2. :func:`build_panel_zone_spring` -- builds a
   :class:`UniaxialBilinear` material configured to be installed in a
   :class:`ZeroLengthElement` between the column-face and beam-end
   nodes at a joint, acting on the rotational DOF.

References
----------
* Krawinkler, H. (1978). "Shear in beam-column joints in seismic
  design of steel frames." *Engineering Journal AISC*, 15(3), 82-91.
* Krawinkler, H. (1996). "Cyclic loading histories for seismic
  experimentation on structural components." *Earthquake Spectra*,
  12(1), 1-12.
* AISC 360-22, Cl. J10.6 (panel-zone shear strength).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from femsolver.materials.uniaxial.bilinear import UniaxialBilinear


# ============================================================ result type

@dataclass
class PanelZoneProperties:
    """Krawinkler panel-zone shear backbone.

    Attributes
    ----------
    V_y : float
        Yield shear force in the panel zone (N).
    K_e : float
        Elastic shear stiffness (N per unit shear strain, i.e. N/rad
        when shear strain is taken as angle).
    K_p : float
        Post-yield shear stiffness (N/rad).
    gamma_y : float
        Yield shear strain (rad) = V_y / K_e.
    M_y_joint : float
        Equivalent yield moment of the panel zone about its centroid,
        useful when modelling as a rotational spring on a joint
        master node:  M_y = V_y * d_b.
    K_e_rot : float
        Equivalent rotational stiffness (N·m/rad) = K_e * d_b^2 /
        d_b = K_e * d_b. (We treat panel-zone shear strain as the
        rotation of one corner relative to the opposite.)
    b_over_a : float
        Ratio of the strength contribution from the column flanges
        (3 b_cf t_cf^2 / (d_b d_c t_p)). Useful diagnostic.
    """

    V_y: float
    K_e: float
    K_p: float
    gamma_y: float
    M_y_joint: float
    K_e_rot: float
    b_over_a: float


# ============================================================ model

def krawinkler_panel_zone(
    *,
    f_y: float,
    d_c: float, t_p: float,
    d_b: float,
    b_cf: float, t_cf: float,
    R_y: float = 1.1,
    G: float = 77.0e9,
    K_p_ratio: float = 0.06,
) -> PanelZoneProperties:
    """Krawinkler panel-zone shear backbone (AISC J10 strength).

    Parameters
    ----------
    f_y : float
        Column-web (panel) yield stress (Pa).
    d_c : float
        Column depth (m).
    t_p : float
        Panel-zone web thickness = column web thickness +
        any doubler plate (m).
    d_b : float
        Beam depth (m).
    b_cf : float
        Column flange width (m).
    t_cf : float
        Column flange thickness (m).
    R_y : float, default 1.1
        AISC expected-yield ratio for the column steel.
    G : float, default 77 GPa
        Steel shear modulus.
    K_p_ratio : float, default 0.06
        Post-yield to elastic stiffness ratio (Krawinkler 1996).
    """
    for name, val in [("f_y", f_y), ("d_c", d_c), ("t_p", t_p),
                        ("d_b", d_b), ("b_cf", b_cf),
                        ("t_cf", t_cf)]:
        if val <= 0.0:
            raise ValueError(f"{name} must be > 0, got {val}")
    if R_y <= 0.0 or G <= 0.0:
        raise ValueError("R_y and G must be > 0")
    if not (0.0 < K_p_ratio <= 1.0):
        raise ValueError("K_p_ratio must be in (0, 1]")

    boundary_term = 3.0 * b_cf * t_cf ** 2 / (d_b * d_c * t_p)
    V_y = 0.55 * R_y * f_y * d_c * t_p * (1.0 + boundary_term)
    K_e = G * d_c * t_p * d_b              # N per unit shear strain
    K_p = K_p_ratio * K_e
    gamma_y = V_y / K_e
    M_y_joint = V_y * d_b
    K_e_rot = K_e * d_b                    # M = V·d_b, theta = gamma -> K_rot = K_e·d_b
    return PanelZoneProperties(
        V_y=float(V_y), K_e=float(K_e), K_p=float(K_p),
        gamma_y=float(gamma_y),
        M_y_joint=float(M_y_joint),
        K_e_rot=float(K_e_rot),
        b_over_a=float(boundary_term),
    )


# ============================================================ spring helper

def build_panel_zone_material(props: PanelZoneProperties) -> UniaxialBilinear:
    """Build a :class:`UniaxialBilinear` moment-rotation spring sized
    to the panel-zone backbone.

    The material is intended for use in a :class:`ZeroLengthElement`
    in the rotational DOF (index 2 in 2D, index 5 in 3D), connecting
    the beam-end node to the column-face node. After installation,
    the rotational spring captures panel-zone shear flexibility.

    The mapping is::

        E_eq (stiffness) = K_e_rot          (N·m / rad)
        sigma_y_eq (yield)= M_y_joint        (N·m)
        b (post-yield ratio) = K_p / K_e     (dimensionless)
    """
    return UniaxialBilinear(
        E=props.K_e_rot,
        sigma_y=props.M_y_joint,
        b=props.K_p / props.K_e,
    )
