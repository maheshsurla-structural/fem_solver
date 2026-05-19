"""Generalized eigen-analysis ``K phi = omega^2 M phi`` for free vibration."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

from femsolver.analysis.assembler import assemble_mass, assemble_stiffness
from femsolver.analysis.constraint_handler import TransformationHandler
from femsolver.numerics.dof_numbering import rcm_renumber


class EigenAnalysis:
    """Free-vibration eigen-analysis.

    Solves the generalized eigenproblem ``K phi = omega^2 M phi`` for the
    ``num_modes`` lowest eigenvalues using a shift-invert Lanczos
    iteration (``scipy.sparse.linalg.eigsh`` with ``sigma=0``).

    Parameters
    ----------
    model : Model
    num_modes : int, default 6
        Number of lowest modes to return.
    lumped : bool, default False
        Use lumped (diagonal) mass instead of consistent.
    constraints : str, default "transformation"
        MP-constraint handler (only ``"transformation"`` is supported for
        eigen analysis — penalty enforcement perturbs eigenvalues).
    numberer : {"default", "rcm"}, default "default"

    Results (populated after ``run()``)
    ----------------------------------
    eigenvalues : ndarray of shape (num_modes,)
        Sorted ascending. Each entry is :math:`\\omega^2`.
    frequencies : ndarray of shape (num_modes,)
        Cyclic frequencies in Hz, ``omega / (2 pi)``.
    periods : ndarray of shape (num_modes,)
        Natural periods in seconds, ``2 pi / omega``.
    mode_shapes : ndarray of shape (neq, num_modes)
        Mass-orthonormalized eigenvectors in the *full* (free-DOF) space.
        Each column also written into ``Node.mode_disp[mode]`` for nodes.
    """

    def __init__(
        self,
        model,
        num_modes: int = 6,
        *,
        lumped: bool = False,
        constraints: str = "transformation",
        numberer: str = "default",
    ):
        self.model = model
        if num_modes < 1:
            raise ValueError("num_modes must be >= 1")
        self.num_modes = int(num_modes)
        self.lumped = bool(lumped)
        if constraints != "transformation":
            raise ValueError(
                "EigenAnalysis only supports the transformation handler "
                "(penalty introduces stiff fictitious modes)"
            )
        self.handler = TransformationHandler()
        if numberer not in ("default", "rcm"):
            raise ValueError(f"unknown numberer {numberer!r}")
        self.numberer = numberer

        self.K: sp.csc_matrix | None = None
        self.M: sp.csc_matrix | None = None
        self.eigenvalues: np.ndarray | None = None
        self.frequencies: np.ndarray | None = None
        self.periods: np.ndarray | None = None
        self.mode_shapes: np.ndarray | None = None

    def run(self) -> dict:
        m = self.model
        if self.numberer == "rcm":
            rcm_renumber(m)
        else:
            m.number_dofs()
        if m.neq == 0:
            raise RuntimeError(
                "no free DOFs — model is fully constrained or empty"
            )

        K = assemble_stiffness(m)
        M = assemble_mass(m, lumped=self.lumped)
        self.K, self.M = K, M

        self._validate(K, M)

        if m.mp_constraints:
            build = self.handler.build(m)
            T = build.T
            K_solve = (T.T @ K @ T).tocsc()
            M_solve = (T.T @ M @ T).tocsc()
        else:
            build = None
            K_solve, M_solve = K, M

        if K_solve.shape[0] < self.num_modes + 1:
            raise RuntimeError(
                f"requested {self.num_modes} modes but the constrained "
                f"system only has {K_solve.shape[0]} DOFs"
            )

        # shift-invert near sigma=0 finds the smallest eigenvalues
        try:
            vals, vecs = eigsh(
                K_solve, k=self.num_modes, M=M_solve, sigma=0.0, which="LM"
            )
        except Exception as exc:
            raise RuntimeError(
                f"eigen solve failed: {exc}. Common causes: zero density on "
                "all elements (M is singular), insufficient supports, or "
                "rigid-body modes that need explicit fixity."
            ) from exc

        order = np.argsort(vals)
        vals = vals[order]
        vecs = vecs[:, order]
        # eigsh can return tiny negative values for true zeros — clamp before sqrt
        vals_clamped = np.where(vals < 0.0, 0.0, vals)
        omegas = np.sqrt(vals_clamped)

        self.eigenvalues = vals
        self.frequencies = omegas / (2.0 * np.pi)
        # avoid division by zero for rigid-body modes
        with np.errstate(divide="ignore", invalid="ignore"):
            self.periods = np.where(omegas > 0.0, 2.0 * np.pi / omegas, np.inf)

        # recover full-size mode shapes
        if build is not None:
            full = build.T @ vecs  # (neq, num_modes)
        else:
            full = vecs
        self.mode_shapes = np.asarray(full)

        # scatter mode shapes onto nodes for convenient access
        for node in m.nodes.values():
            node.mode_disp = np.zeros((node.ndf, self.num_modes), dtype=float)
            for i in range(node.ndf):
                eq = int(node.eqn[i])
                if eq >= 0:
                    node.mode_disp[i, :] = self.mode_shapes[eq, :]

        return {
            "neq": int(m.neq),
            "num_modes": self.num_modes,
            "frequencies_hz": self.frequencies.tolist(),
            "periods_s": self.periods.tolist(),
        }

    # ------------------------------------------------------------------ checks
    def _validate(self, K: sp.spmatrix, M: sp.spmatrix) -> None:
        if K.shape != M.shape:
            raise RuntimeError(
                f"K and M shape mismatch: {K.shape} vs {M.shape}"
            )
        m_diag = np.asarray(M.diagonal()).ravel()
        if not np.any(m_diag > 0.0):
            raise RuntimeError(
                "mass matrix has no positive diagonal — at least one element's "
                "material must have rho > 0"
            )
        # warn-equivalent: zero-mass DOFs are common (e.g., beam rotational
        # DOFs with the lumped scheme used here). We don't error since the
        # eigen solver handles the singular-M case via the shift-invert.
