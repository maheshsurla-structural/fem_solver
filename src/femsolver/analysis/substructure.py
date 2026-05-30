"""Substructuring and DOF reduction.

Two classical techniques for reducing the size of a large stiffness /
mass system:

* **Guyan static condensation** (Guyan 1965). Partition the DOFs
  into "master" (retained) and "slave" (condensed) sets, then
  eliminate the slaves under the assumption that they carry no
  inertia of their own::

       K = [ K_mm   K_ms ]      f = [ f_m ]
           [ K_sm   K_ss ]          [ f_s ]

       K_red = K_mm - K_ms K_ss^{-1} K_sm
       f_red = f_m  - K_ms K_ss^{-1} f_s

  Used in classical mechanical analyses for first-pass model
  reduction. Loses high-frequency accuracy because it neglects slave
  inertia.

* **Craig-Bampton fixed-interface modes** (Craig & Bampton 1968).
  Combine static condensation with the lowest few **fixed-interface**
  eigenmodes (the modes of the substructure with all masters
  clamped). Gives a much more accurate reduced model for dynamic
  analyses; the workhorse method for component-mode synthesis in
  commercial multi-body / FE codes.

The reduced system is built directly from full ``K`` and ``M`` matrices
plus a list of master DOF indices. After solving the reduced system
for the master responses, slave responses are recovered by back-
substitution (also provided).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# ============================================================ partition helper

def _partition(K, master_dofs: np.ndarray):
    """Return ``(K_mm, K_ms, K_sm, K_ss, slave_dofs)`` given a
    boolean / integer master selection."""
    K = K.tocsc() if sp.issparse(K) else np.asarray(K)
    neq = K.shape[0]
    master = np.asarray(master_dofs, dtype=int).ravel()
    mask = np.zeros(neq, dtype=bool)
    mask[master] = True
    slave = np.flatnonzero(~mask)
    if sp.issparse(K):
        K_mm = K[master, :][:, master]
        K_ms = K[master, :][:, slave]
        K_sm = K[slave, :][:, master]
        K_ss = K[slave, :][:, slave]
    else:
        K_mm = K[np.ix_(master, master)]
        K_ms = K[np.ix_(master, slave)]
        K_sm = K[np.ix_(slave, master)]
        K_ss = K[np.ix_(slave, slave)]
    return K_mm, K_ms, K_sm, K_ss, slave


# ============================================================ Guyan

@dataclass
class GuyanResult:
    """Statically-condensed system.

    Attributes
    ----------
    K_red : np.ndarray (n_master, n_master)
        Reduced stiffness matrix.
    f_red : np.ndarray (n_master,)
        Reduced load vector.
    master_dofs : np.ndarray
        DOF indices in the original numbering retained as masters.
    slave_dofs : np.ndarray
        DOF indices condensed out.
    T : np.ndarray (n_full, n_master)
        Reduction matrix:
        ``u_full = T @ u_master``  (slaves recovered exactly under the
        Guyan assumption that no external load acts on them).
    """

    K_red: np.ndarray
    f_red: np.ndarray
    master_dofs: np.ndarray
    slave_dofs: np.ndarray
    T: np.ndarray


def guyan_condensation(
    K, f, master_dofs,
) -> GuyanResult:
    """Static Guyan condensation.

    Parameters
    ----------
    K : sparse or dense (n, n)
        Stiffness matrix of the full system.
    f : array (n,)
        Load vector. Slaves CAN carry load; the reduction adds the
        appropriate back-substitution.
    master_dofs : array-like of int
        DOF indices (in the full numbering) to retain.

    Returns
    -------
    GuyanResult
    """
    f = np.asarray(f, dtype=float).ravel()
    K_mm, K_ms, K_sm, K_ss, slave = _partition(K, master_dofs)
    master = np.asarray(master_dofs, dtype=int).ravel()
    # Convert to dense for the reduction step (sparse partitions are
    # already small if the user picks few masters)
    K_mm = np.asarray(K_mm.todense() if sp.issparse(K_mm) else K_mm)
    K_ms = np.asarray(K_ms.todense() if sp.issparse(K_ms) else K_ms)
    K_sm = np.asarray(K_sm.todense() if sp.issparse(K_sm) else K_sm)
    K_ss_csc = sp.csc_matrix(K_ss) if not sp.issparse(K_ss) else K_ss.tocsc()
    # Solve K_ss^{-1} K_sm columns at a time (sparse solve)
    Phi = spla.spsolve(K_ss_csc, sp.csc_matrix(K_sm))
    if sp.issparse(Phi):
        Phi = np.asarray(Phi.todense())
    Phi = np.atleast_2d(Phi).reshape(K_sm.shape)
    K_red = K_mm - K_ms @ Phi
    # Load reduction
    f_s = f[slave]
    f_m = f[master]
    f_red = f_m - K_ms @ spla.spsolve(K_ss_csc, f_s)
    # Reduction matrix T
    n = K.shape[0]
    n_m = master.size
    T = np.zeros((n, n_m))
    # Master rows: identity
    for i, idx in enumerate(master):
        T[idx, i] = 1.0
    # Slave rows: -Phi (since u_s = -K_ss^{-1} K_sm u_m + K_ss^{-1} f_s)
    for j in range(n_m):
        T[slave, j] = -Phi[:, j]
    return GuyanResult(
        K_red=K_red, f_red=f_red,
        master_dofs=master, slave_dofs=slave,
        T=T,
    )


def guyan_recover_full(
    res: GuyanResult, u_master: np.ndarray,
    *, K=None, f=None,
) -> np.ndarray:
    """Recover the full displacement vector from the master solution.

    Slaves are computed by static back-substitution::

        u_slave = K_ss^{-1} (f_s - K_sm @ u_master)

    If ``K`` and ``f`` are omitted, only the homogeneous part is
    returned (``u_full = T @ u_master``, which is exact when ``f_s = 0``).
    """
    u_master = np.asarray(u_master, dtype=float).ravel()
    u_full = res.T @ u_master
    if K is not None and f is not None:
        f = np.asarray(f, dtype=float).ravel()
        K_sm = (K[res.slave_dofs, :][:, res.master_dofs]
                if sp.issparse(K)
                else K[np.ix_(res.slave_dofs, res.master_dofs)])
        K_ss = (K[res.slave_dofs, :][:, res.slave_dofs]
                if sp.issparse(K)
                else K[np.ix_(res.slave_dofs, res.slave_dofs)])
        K_sm = np.asarray(K_sm.todense() if sp.issparse(K_sm) else K_sm)
        K_ss = sp.csc_matrix(K_ss) if not sp.issparse(K_ss) else K_ss.tocsc()
        f_s = f[res.slave_dofs]
        u_s = spla.spsolve(K_ss, f_s - K_sm @ u_master)
        u_full = u_full.copy()
        u_full[res.slave_dofs] = u_s
    return u_full


# ============================================================ Craig-Bampton

@dataclass
class CraigBamptonResult:
    """Reduced system from Craig-Bampton component-mode synthesis.

    Attributes
    ----------
    K_red, M_red : np.ndarray
        Reduced stiffness and mass matrices, shape
        ``(n_master + n_keep, n_master + n_keep)``.
    master_dofs : np.ndarray
        DOF indices in the original numbering (boundary / interface).
    slave_dofs : np.ndarray
        DOF indices that have been condensed via constraint modes +
        replaced with kept modes.
    n_keep : int
        Number of fixed-interface modes retained.
    omega_fixed : np.ndarray
        Fixed-interface natural frequencies (rad/s) of the kept modes.
    Phi : np.ndarray
        Constraint modes (slave-rows of T_static).
    Psi : np.ndarray (n_slave, n_keep)
        Kept fixed-interface eigenmodes (slave-DOF only).
    """

    K_red: np.ndarray
    M_red: np.ndarray
    master_dofs: np.ndarray
    slave_dofs: np.ndarray
    n_keep: int
    omega_fixed: np.ndarray
    Phi: np.ndarray
    Psi: np.ndarray


def craig_bampton(
    K, M, master_dofs, *, n_keep: int,
) -> CraigBamptonResult:
    """Craig-Bampton fixed-interface component-mode synthesis.

    Constructs the transformation
    ``u = [Phi  Psi] @ q``, where:

    * ``Phi`` are the **constraint modes** (one per master DOF) =
      static deflections of slaves when each master moves by 1 with
      the other masters held at 0.  Same as the Guyan reduction
      matrix; provides the static response.
    * ``Psi`` are the **fixed-interface modes** — the lowest
      ``n_keep`` eigenmodes of the slave-only system
      ``K_ss Psi = M_ss Psi diag(omega_fixed^2)``.

    The reduced system has order ``n_master + n_keep``.

    Parameters
    ----------
    K, M : sparse or dense (n, n)
    master_dofs : array-like of int
        Boundary / interface DOF indices.
    n_keep : int
        Number of fixed-interface modes to retain.

    Returns
    -------
    CraigBamptonResult
    """
    if n_keep < 0:
        raise ValueError("n_keep must be >= 0")
    K_mm, K_ms, K_sm, K_ss, slave = _partition(K, master_dofs)
    M_mm, M_ms, M_sm, M_ss, _ = _partition(M, master_dofs)
    master = np.asarray(master_dofs, dtype=int).ravel()
    # Constraint modes Phi = -K_ss^{-1} K_sm  (shape: n_slave x n_master)
    K_ss_csc = (K_ss.tocsc() if sp.issparse(K_ss)
                else sp.csc_matrix(K_ss))
    K_sm_dense = np.asarray(K_sm.todense() if sp.issparse(K_sm)
                              else K_sm)
    Phi = -spla.spsolve(K_ss_csc, sp.csc_matrix(K_sm_dense))
    if sp.issparse(Phi):
        Phi = np.asarray(Phi.todense())
    Phi = np.atleast_2d(Phi).reshape(K_sm_dense.shape)
    # Fixed-interface modes
    if n_keep > 0:
        if sp.issparse(K_ss):
            K_ss_dense = np.asarray(K_ss.todense())
        else:
            K_ss_dense = np.asarray(K_ss)
        if sp.issparse(M_ss):
            M_ss_dense = np.asarray(M_ss.todense())
        else:
            M_ss_dense = np.asarray(M_ss)
        # Generalized eigenvalue: K_ss Psi = lambda M_ss Psi
        from scipy.linalg import eigh as scipy_eigh
        w, V = scipy_eigh(K_ss_dense, M_ss_dense)
        # Sort ascending (eigh already does), take the smallest n_keep
        Psi = V[:, :n_keep]
        omega_fixed = np.sqrt(np.maximum(w[:n_keep], 0.0))
    else:
        Psi = np.zeros((slave.size, 0))
        omega_fixed = np.zeros(0)
    # Build T = [I  0;  Phi  Psi]  (n_master rows then n_slave rows)
    n = K.shape[0]
    n_m = master.size
    T = np.zeros((n, n_m + n_keep))
    for i, idx in enumerate(master):
        T[idx, i] = 1.0
    for j in range(n_m):
        T[slave, j] = Phi[:, j]
    for k in range(n_keep):
        T[slave, n_m + k] = Psi[:, k]
    # Convert K, M to dense if sparse so T^T K T works directly
    Kd = np.asarray(K.todense() if sp.issparse(K) else K)
    Md = np.asarray(M.todense() if sp.issparse(M) else M)
    K_red = T.T @ Kd @ T
    M_red = T.T @ Md @ T
    return CraigBamptonResult(
        K_red=K_red, M_red=M_red,
        master_dofs=master, slave_dofs=slave,
        n_keep=int(n_keep),
        omega_fixed=omega_fixed,
        Phi=Phi, Psi=Psi,
    )
