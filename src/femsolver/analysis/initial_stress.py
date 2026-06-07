"""General initial-stress / eigenstrain equivalent loads.

Prestress, residual stress, and thermal strain are all the *same*
finite-element operation: an **initial stress** ``σ₀`` present in an
element at zero nodal displacement produces a work-equivalent nodal
load

    f = -∫ Bᵀ σ₀ dV

(the sign moves the ``σ₀`` term to the right-hand side of
``K u = F_ext - ∫ Bᵀ σ₀ dV``). Applying this load and solving gives the
structure's response to the imposed initial-stress field.

This module exposes that operation directly for the continuum elements
(:class:`~femsolver.elements.plane.Quad4`,
:class:`~femsolver.elements.solid.Hex8`,
:class:`~femsolver.elements.solid.Tet4`), which is exactly what is
needed to **prestress a solid model**: a tendon force ``P`` smeared
over an area ``A`` along a unit direction ``d`` is the uniaxial initial
stress

    σ₀ = -(P / A) · (d ⊗ d)        (compression along the tendon)

See :func:`prestress_initial_stress`.

The thermal-strain module (:mod:`femsolver.analysis.thermal_strain`)
is the special isotropic case ``σ₀ = D · (α ΔT) I``; this module is the
general directional version.

Sign convention
---------------
``σ₀`` follows the usual tension-positive convention. A restrained body
with an initial tension ``σ₀`` develops support reactions that balance
it; an unrestrained body with uniform ``σ₀`` strains to
``ε = -D⁻¹ σ₀`` (the initial stress relaxes to zero net stress as the
body deforms freely).
"""
from __future__ import annotations

import numpy as np

from femsolver.elements.plane import Quad4
from femsolver.elements.shell import ShellMITC4, _jacobian2d
from femsolver.elements.solid import Hex8, Tet4, _hex8_dN_dxi
from femsolver.numerics.quadrature import gauss_legendre_2d_quad


# ============================================================ per-element f = -∫Bᵀσ₀

def _f_quad4(elem: Quad4, sigma0: np.ndarray) -> np.ndarray:
    """Equivalent nodal load (8,) for a plane Quad4 under initial stress
    ``sigma0 = [sxx, syy, sxy]`` (Pa)."""
    X = elem.node_coords()
    t = elem.thickness
    f = np.zeros(8)
    xi, eta, w = gauss_legendre_2d_quad(elem.quadrature)
    for q in range(xi.size):
        _, detJ, dN_dx = elem.jacobian(float(xi[q]), float(eta[q]), X)
        B = elem.B_matrix(dN_dx)
        f += (B.T @ sigma0) * (t * detJ * float(w[q]))
    return -f


def _f_hex8(elem: Hex8, sigma0: np.ndarray) -> np.ndarray:
    """Equivalent nodal load (24,) for a Hex8 under initial stress
    ``sigma0 = [sxx, syy, szz, sxy, syz, szx]`` (Pa)."""
    X = elem.node_coords()
    f = np.zeros(24)
    gp = 1.0 / np.sqrt(3.0)
    pts = [(-gp, -gp, -gp), (gp, -gp, -gp), (gp, gp, -gp), (-gp, gp, -gp),
           (-gp, -gp, gp), (gp, -gp, gp), (gp, gp, gp), (-gp, gp, gp)]
    for (xi, eta, zeta) in pts:
        dN = _hex8_dN_dxi(xi, eta, zeta)
        J = dN @ X
        detJ = float(np.linalg.det(J))
        dN_dx = np.linalg.solve(J, dN)
        B = np.zeros((6, 24))
        for i in range(8):
            dNx, dNy, dNz = dN_dx[0, i], dN_dx[1, i], dN_dx[2, i]
            B[0, 3 * i + 0] = dNx
            B[1, 3 * i + 1] = dNy
            B[2, 3 * i + 2] = dNz
            B[3, 3 * i + 0] = dNy
            B[3, 3 * i + 1] = dNx
            B[4, 3 * i + 1] = dNz
            B[4, 3 * i + 2] = dNy
            B[5, 3 * i + 0] = dNz
            B[5, 3 * i + 2] = dNx
        f += (B.T @ sigma0) * detJ        # weight = 1 for 2x2x2 Gauss
    return -f


def _f_tet4(elem: Tet4, sigma0: np.ndarray) -> np.ndarray:
    """Equivalent nodal load (12,) for a Tet4 under initial stress
    ``sigma0 = [sxx, syy, szz, sxy, syz, szx]`` (Pa)."""
    B, V = elem._B_and_volume()
    return -(B.T @ sigma0) * V


def _f_shell_membrane(elem: ShellMITC4, sigma0: np.ndarray) -> np.ndarray:
    """Global equivalent nodal load (24,) for a ShellMITC4 under an
    in-plane membrane initial stress ``sigma0 = [sxx, syy, sxy]`` (Pa,
    in the element's LOCAL in-plane axes).

    The membrane stress resultant is ``N₀ = σ₀ · t``; the work-equivalent
    load is ``f = -∫ Bmᵀ N₀ dA``, expanded from the 20-DOF (5/node)
    membrane ordering to 24 DOF and rotated to global axes.
    """
    R, Xl = elem._local_geom()
    t = elem.thickness
    N0 = sigma0 * t                          # membrane resultant (N/m)
    f20 = np.zeros(20)
    xi, eta, w = gauss_legendre_2d_quad(2)
    for q in range(xi.size):
        _, detJ, dN_dx = _jacobian2d(float(xi[q]), float(eta[q]), Xl)
        Bm = elem._Bm_local(dN_dx)
        f20 += (Bm.T @ N0) * (detJ * float(w[q]))
    f20 = -f20
    # 5-DOF/node (u,v,w,θx,θy) -> 6-DOF/node (insert θz = 0)
    idx5 = [6 * i + k for i in range(4) for k in range(5)]
    f24_local = np.zeros(24)
    f24_local[idx5] = f20
    # local -> global: f_global = Tᵀ f_local  (T: u_local = T u_global)
    T = elem._T_global_to_local(R)
    return T.T @ f24_local


# ============================================================ public API

def apply_initial_stress(model, stresses: dict, *, factor: float = 1.0) -> int:
    """Add initial-stress (eigenstress) equivalent nodal loads to a model.

    Parameters
    ----------
    model : Model
        A 2-D (``ndf=2``) plane model or 3-D (``ndf=3``) solid model.
    stresses : dict
        ``{element_tag: sigma0}`` mapping. ``sigma0`` is a Voigt vector:

        * plane (Quad4): ``[σxx, σyy, σxy]`` (3,)
        * solid (Hex8 / Tet4): ``[σxx, σyy, σzz, σxy, σyz, σzx]`` (6,)
        * shell (ShellMITC4): in-plane **membrane** stress
          ``[σxx, σyy, σxy]`` (3,) in the element's local axes; the
          resultant ``N₀ = σ₀·t`` is integrated over the membrane.

        Tension-positive (Pa).
    factor : float, default 1.0
        Scale applied to every initial stress (for load combinations).

    Returns
    -------
    int
        Number of elements processed.

    Raises
    ------
    NotImplementedError
        For element types without an initial-stress lowering (currently
        only the continuum elements Quad4 / Hex8 / Tet4 are supported;
        shells are a planned follow-up).
    """
    n_done = 0
    for tag, s0 in stresses.items():
        elem = model.element(tag)
        s0 = np.asarray(s0, dtype=float).ravel() * factor
        if isinstance(elem, Quad4):
            if s0.size != 3:
                raise ValueError("Quad4 sigma0 must be [sxx, syy, sxy]")
            f = _f_quad4(elem, s0)
            for k, ntag in enumerate(elem.node_tags):
                model.add_nodal_load(ntag, [f[2 * k], f[2 * k + 1]])
        elif isinstance(elem, Hex8):
            if s0.size != 6:
                raise ValueError(
                    "Hex8 sigma0 must be [sxx,syy,szz,sxy,syz,szx]"
                )
            f = _f_hex8(elem, s0)
            for k, ntag in enumerate(elem.node_tags):
                model.add_nodal_load(
                    ntag, [f[3 * k], f[3 * k + 1], f[3 * k + 2]]
                )
        elif isinstance(elem, Tet4):
            if s0.size != 6:
                raise ValueError(
                    "Tet4 sigma0 must be [sxx,syy,szz,sxy,syz,szx]"
                )
            f = _f_tet4(elem, s0)
            for k, ntag in enumerate(elem.node_tags):
                model.add_nodal_load(
                    ntag, [f[3 * k], f[3 * k + 1], f[3 * k + 2]]
                )
        elif isinstance(elem, ShellMITC4):
            if s0.size != 3:
                raise ValueError(
                    "ShellMITC4 sigma0 must be the in-plane membrane "
                    "stress [sxx, syy, sxy] (local axes, Pa)"
                )
            f = _f_shell_membrane(elem, s0)
            for k, ntag in enumerate(elem.node_tags):
                model.add_nodal_load(ntag, f[6 * k:6 * k + 6].tolist())
        else:
            raise NotImplementedError(
                f"initial-stress load not implemented for "
                f"{type(elem).__name__}; supported: Quad4, Hex8, Tet4"
            )
        n_done += 1
    return n_done


# ============================================================ prestress helper

def prestress_initial_stress(
    *,
    P: float,
    A: float,
    direction,
    ndim: int = 3,
) -> np.ndarray:
    """Uniaxial prestress as an initial-stress Voigt vector.

    A tendon force ``P`` smeared over a host area ``A`` along the unit
    vector ``direction`` puts the host in **compression** along that
    direction:

        σ₀ = -(P / A) · (d ⊗ d)

    Parameters
    ----------
    P : float
        Effective tendon force (N, > 0).
    A : float
        Host cross-section area the force is smeared over (m²).
    direction : sequence
        Tendon direction (need not be normalised).
    ndim : {2, 3}, default 3
        Return a 3-vector ``[σxx, σyy, σxy]`` (plane) or a 6-vector
        ``[σxx, σyy, σzz, σxy, σyz, σzx]`` (solid).

    Returns
    -------
    np.ndarray
        The Voigt initial-stress vector to pass to
        :func:`apply_initial_stress`.
    """
    if P <= 0 or A <= 0:
        raise ValueError("P and A must be > 0")
    d = np.asarray(direction, dtype=float).ravel()
    nd = np.linalg.norm(d)
    if nd == 0:
        raise ValueError("direction must be non-zero")
    d = d / nd
    s = -P / A          # compression
    if ndim == 2:
        if d.size < 2:
            raise ValueError("2-D direction needs 2 components")
        dx, dy = d[0], d[1]
        return np.array([s * dx * dx, s * dy * dy, s * dx * dy])
    if d.size != 3:
        raise ValueError("3-D direction needs 3 components")
    dx, dy, dz = d
    return np.array([
        s * dx * dx, s * dy * dy, s * dz * dz,
        s * dx * dy, s * dy * dz, s * dz * dx,
    ])
