"""Linear (eigenvalue) buckling analysis.

Solves the generalized eigenvalue problem

    (K + lambda K_g) v = 0

where ``K`` is the initial-elastic stiffness, ``K_g`` is the geometric
stiffness computed at a reference loaded state (a small linear-static
displacement under a user-supplied reference load), and ``lambda`` is
the multiplier of the reference load at which the structure buckles.

The smallest positive ``lambda`` is the critical load factor. The
critical buckling load is then ``lambda_cr * F_ref``.

Algorithm
---------
1. Apply the reference load (already attached to the model via
   ``add_nodal_load`` / element distributed loads) and run a linear
   static analysis. ``Node.disp`` then holds the linear-static
   displacement; element internal forces follow from these
   displacements.
2. Assemble two global matrices:

   * ``K = sum_e K_e_global()``         — initial elastic stiffness
   * ``K_T = sum_e K_e_tangent_global()`` — tangent at the loaded state

   The difference ``K_g = K_T - K`` isolates the *geometric* part of
   the tangent contributed by any element that overrides
   ``K_tangent_global`` (corotational truss and beam, both in this
   project). Elements without geometric content contribute zero ``K_g``.
3. Solve the generalized eigenvalue problem ``K_g v = nu K v``. For
   each negative ``nu`` the buckling load factor is ``lambda = -1/nu``
   (we want compression that softens the structure into instability,
   hence ``nu < 0``).
4. Sort the positive ``lambda`` values ascending; the first is the
   critical buckling load factor.

This is the standard linearised-pre-buckling formulation used in
commercial structural-FE codes for stability-design checks. It is
exact for problems where the pre-buckling displacement is small
(slender columns, frames near onset of instability) and approximate
for larger pre-buckling deformation. For nonlinear post-buckling, use
arc-length continuation (Phase 10) on a corotational model.

Caveats
-------
* The model needs at least one element with a non-trivial
  ``K_tangent_global`` for buckling modes to exist. Use
  :class:`~femsolver.elements.truss_corot.Truss2DCorotational` or
  :class:`~femsolver.elements.beam_corot.BeamColumn2DCorotational`.
* The eigenvalue solver here uses dense :func:`scipy.linalg.eigh` for
  simplicity and robustness — fine up to a few thousand DOFs. Larger
  problems would benefit from a sparse iterative solver
  (:func:`scipy.sparse.linalg.eigsh`) with shift-invert.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh

from femsolver.analysis.assembler import assemble_stiffness
from femsolver.analysis.static_integrator import _assemble_tangent
from femsolver.analysis.linear_static import LinearStaticAnalysis
from femsolver.numerics.dof_numbering import rcm_renumber


class LinearBucklingAnalysis:
    """Eigenvalue-based buckling analysis.

    Parameters
    ----------
    model : Model
        Model with reference loads attached. The load pattern is the
        unit pattern; the analysis returns the multiplier ``lambda``
        such that ``lambda * F_ref`` causes buckling.
    num_modes : int, default 4
        Number of (positive) buckling modes to retain in the result.
    numberer : {"default", "rcm"}, default "default"

    Notes
    -----
    Run via :meth:`run`. After running:

    * ``self.load_factors`` is an ``ndarray`` of positive load factors
      in ascending order — the first entry is the critical factor.
    * ``self.mode_shapes`` is an ``(neq, num_modes)`` array of mode
      shape vectors.
    * ``Node.mode_disp[i, k]`` is the i-th DOF amplitude in the k-th
      buckling mode (parallel to the convention used by
      :class:`~femsolver.analysis.eigen.EigenAnalysis`).
    """

    def __init__(self, model, num_modes: int = 4, *,
                 numberer: str = "default", mode: str = "sparse"):
        if num_modes < 1:
            raise ValueError("num_modes must be >= 1")
        if numberer not in ("default", "rcm"):
            raise ValueError(f"unknown numberer {numberer!r}")
        if mode not in ("sparse", "dense"):
            raise ValueError(
                f"mode must be 'sparse' or 'dense', got {mode!r}"
            )
        self.model = model
        self.num_modes = int(num_modes)
        self.numberer = numberer
        self.mode = mode

        # Results, populated by run()
        self.load_factors: np.ndarray | None = None
        self.mode_shapes: np.ndarray | None = None

    # ------------------------------------------------------------------ run
    def run(self) -> dict:
        m = self.model
        m.reset_results()
        if self.numberer == "rcm":
            rcm_renumber(m)
        else:
            m.number_dofs()
        if m.neq == 0:
            raise RuntimeError(
                "no free DOFs — model is fully constrained or empty"
            )

        # Step 1+2: linear static analysis at the reference load.
        # After this, Node.disp holds the linear-static displacement
        # and element internal-state is consistent with it (for
        # corotational elements, K_tangent_global will pick up the
        # geometric softening from the induced internal forces).
        LinearStaticAnalysis(m).run()

        # Step 3: assemble K (initial) and K_T (tangent at loaded state).
        K_sparse = assemble_stiffness(m)
        K_T_sparse = _assemble_tangent(m)
        K_g_sparse = (K_T_sparse - K_sparse).tocsc()
        if abs(K_g_sparse).max() < 1.0e-12 * max(1.0, abs(K_sparse).max()):
            raise RuntimeError(
                "geometric stiffness K_g is essentially zero — the model "
                "has no elements with a state-dependent K_tangent_global, "
                "or the loads do not create internal forces. For linear "
                "buckling, use corotational elements (Truss2DCorotational, "
                "BeamColumn2DCorotational) and an axial / compressive "
                "load pattern."
            )

        # Step 4: solve K_g v = nu K v. Sparse (eigsh + shift-invert)
        # is the default and scales to large problems; dense is a
        # fallback for very small models or when the sparse solver
        # cannot find enough modes (e.g., extremely small neq).
        use_sparse = self.mode == "sparse" and m.neq > self.num_modes + 2
        if use_sparse:
            try:
                lambdas, vecs = self._solve_sparse(K_sparse, K_g_sparse)
            except Exception:
                # Fall back to dense on sparse-solver failure (typical
                # for small / poorly-conditioned models).
                lambdas, vecs = self._solve_dense(K_sparse, K_g_sparse)
        else:
            lambdas, vecs = self._solve_dense(K_sparse, K_g_sparse)
        n_keep = lambdas.size

        self.load_factors = lambdas
        self.mode_shapes = vecs

        # Scatter mode shapes to Node.mode_disp for output, matching
        # EigenAnalysis's convention.
        for node in m.nodes.values():
            node.mode_disp = np.zeros((node.ndf, n_keep))
            for i in range(node.ndf):
                eq = int(node.eqn[i])
                if eq >= 0:
                    node.mode_disp[i, :] = vecs[eq, :]

        return {
            "neq": int(m.neq),
            "num_modes": int(n_keep),
            "load_factors": lambdas.tolist(),
            "critical_load_factor": float(lambdas[0]),
        }

    # --------------------------------------------------------------- solvers
    def _solve_dense(self, K_sparse: sp.spmatrix,
                       K_g_sparse: sp.spmatrix) -> tuple[np.ndarray, np.ndarray]:
        """Dense fallback: scipy.linalg.eigh on the full generalized
        eigenproblem K_g v = nu K v. Use this for very small models
        (neq < num_modes + 3) or when the sparse path fails."""
        K = np.asarray(K_sparse.toarray())
        K_g = np.asarray(K_g_sparse.toarray())
        try:
            nu_all, vecs_all = eigh(K_g, K)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(
                f"buckling eigenproblem failed: {exc}. Likely cause: K is "
                "near-singular (rigid-body mode not constrained) or K_g is "
                "ill-conditioned."
            ) from exc
        return self._filter_modes(nu_all, vecs_all)

    def _solve_sparse(self, K_sparse: sp.spmatrix,
                        K_g_sparse: sp.spmatrix) -> tuple[np.ndarray, np.ndarray]:
        """Sparse path: scipy.sparse.linalg.eigsh with shift-invert at
        sigma=0. Finds eigenvalues closest to zero, which for buckling
        correspond to the smallest |nu| (= largest |lambda| = critical
        and near-critical load factors). We request 2 * num_modes
        modes and filter the negative-nu (compression-buckling) branch.
        """
        # Request more modes than needed to account for the +/- split
        # of eigenvalues near zero; we'll filter to the compression
        # branch (nu < 0) afterward.
        k = min(2 * self.num_modes + 2, K_sparse.shape[0] - 1)
        nu_all, vecs_all = eigsh(
            K_g_sparse, M=K_sparse, k=k, sigma=0.0, which="LM",
        )
        return self._filter_modes(nu_all, vecs_all)

    def _filter_modes(self, nu_all: np.ndarray,
                        vecs_all: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Common post-processing: pick the negative-nu branch,
        convert to load factors, sort ascending, keep ``num_modes``."""
        scale = float(np.max(np.abs(nu_all))) if nu_all.size else 0.0
        threshold = max(1.0e-12, 1.0e-6 * scale)
        valid = nu_all < -threshold
        if not valid.any():
            raise RuntimeError(
                "no buckling modes found — every generalized eigenvalue "
                "of K_g v = nu K v is non-negative (or below the noise "
                "floor). Apply a compressive reference load (or flip the "
                "sign)."
            )
        nu_buck = nu_all[valid]
        vecs_buck = vecs_all[:, valid]
        lambdas = -1.0 / nu_buck
        order = np.argsort(lambdas)
        n_keep = min(self.num_modes, lambdas.size)
        return lambdas[order][:n_keep], vecs_buck[:, order][:, :n_keep]
