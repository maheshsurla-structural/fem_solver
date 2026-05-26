"""Linear solver abstractions.

Provides a pluggable ``LinearSolver`` interface for the linear-system
solves at the heart of every analysis. The default
:class:`DirectSparseSolver` wraps SciPy's :func:`spsolve` (UMFPACK /
SuperLU) and reproduces the existing behaviour of
:class:`~femsolver.analysis.linear_static.LinearStaticAnalysis`. The
:class:`IterativeSolver` wraps :func:`scipy.sparse.linalg.cg` (for
symmetric positive-definite systems) or :func:`gmres` (for general
non-symmetric systems) with an optional ILU preconditioner.

For models in the 10k+ DOF range, the iterative path with a good
preconditioner can dramatically outperform the direct path's memory
footprint, while preserving correctness to the requested tolerance.
For most engineering structural problems (< 10k DOF), the direct
solver is faster and simpler -- stick with the default.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import cg, factorized, gmres, spilu, spsolve


class LinearSolver(ABC):
    """Abstract solver for ``A x = b`` with a sparse ``A``."""

    @abstractmethod
    def solve(self, A: sp.spmatrix, b: np.ndarray) -> np.ndarray:
        """Return the solution vector ``x`` to ``A x = b``."""


class DirectSparseSolver(LinearSolver):
    """Default solver: :func:`scipy.sparse.linalg.spsolve` (SuperLU /
    UMFPACK).

    Best for small-to-medium problems (< 10k--50k DOF depending on
    sparsity). Reproduces the existing behaviour of femsolver
    analyses.
    """

    def solve(self, A: sp.spmatrix, b: np.ndarray) -> np.ndarray:
        try:
            x = np.asarray(spsolve(A, b)).ravel()
        except Exception as exc:
            raise RuntimeError(
                f"direct sparse solve failed: {exc}. Likely cause: singular "
                "stiffness matrix (insufficient supports, mechanism, or "
                "duplicate elements)."
            ) from exc
        if not np.all(np.isfinite(x)):
            raise RuntimeError(
                "direct sparse solve produced non-finite values. Likely "
                "cause: singular stiffness matrix (insufficient supports, "
                "mechanism, or duplicate elements)."
            )
        return x


class IterativeSolver(LinearSolver):
    """Iterative solver -- CG (SPD) or GMRES (general) with an
    optional ILU preconditioner.

    Parameters
    ----------
    method : ``"cg"`` (default) or ``"gmres"``
        ``"cg"``: conjugate-gradient, requires SPD matrices. Faster
        per iteration for structural-FE stiffness matrices when the
        model has no rigid-body modes / negative-definite branches.

        ``"gmres"``: generalized minimal residuals, works for
        non-symmetric / indefinite systems. Use when the tangent
        matrix loses symmetry (e.g., non-associated plasticity).
    tol : float, default 1e-10
        Convergence tolerance for the iterative method (relative).
    max_iter : int, default 5000
    preconditioner : ``"ilu"`` (default) or ``"none"``
        ``"ilu"``: incomplete LU factorization as preconditioner;
        substantially accelerates convergence for most structural-FE
        problems at modest memory cost.

        ``"none"``: no preconditioning. Convergence may be slow.
    drop_tol : float, default 1e-4
        Drop tolerance for the ILU preconditioner. Smaller values give
        a tighter preconditioner (faster convergence per iteration)
        at higher memory cost. ``1e-4`` is a sensible default.
    """

    def __init__(
        self,
        *,
        method: str = "cg",
        tol: float = 1.0e-10,
        max_iter: int = 5000,
        preconditioner: str = "ilu",
        drop_tol: float = 1.0e-4,
    ):
        if method not in ("cg", "gmres"):
            raise ValueError(
                f"method must be 'cg' or 'gmres', got {method!r}"
            )
        if preconditioner not in ("ilu", "none"):
            raise ValueError(
                f"preconditioner must be 'ilu' or 'none', got "
                f"{preconditioner!r}"
            )
        if tol <= 0.0:
            raise ValueError(f"tol must be positive, got {tol}")
        if max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {max_iter}")
        self.method = method
        self.tol = float(tol)
        self.max_iter = int(max_iter)
        self.preconditioner = preconditioner
        self.drop_tol = float(drop_tol)
        # Diagnostic: number of iterations from the last solve.
        self.last_iterations: int = 0
        self.last_residual: float = 0.0

    def solve(self, A: sp.spmatrix, b: np.ndarray) -> np.ndarray:
        b = np.asarray(b).ravel()
        # Build preconditioner
        M = None
        if self.preconditioner == "ilu":
            try:
                ilu = spilu(A.tocsc(), drop_tol=self.drop_tol)
            except Exception as exc:
                raise RuntimeError(
                    f"ILU preconditioner construction failed: {exc}. "
                    "Try preconditioner='none' or fall back to "
                    "DirectSparseSolver."
                ) from exc
            M = sp.linalg.LinearOperator(A.shape, matvec=ilu.solve)
        iter_counter = {"n": 0}

        def _cb(*args, **kwargs) -> None:
            iter_counter["n"] += 1

        if self.method == "cg":
            x, info = cg(A, b, rtol=self.tol, maxiter=self.max_iter,
                          M=M, callback=_cb)
        else:
            x, info = gmres(A, b, rtol=self.tol, maxiter=self.max_iter,
                              M=M, callback=_cb)
        self.last_iterations = iter_counter["n"]
        # Residual norm for diagnostic
        r = b - A @ x
        self.last_residual = float(np.linalg.norm(r))
        if info > 0:
            raise RuntimeError(
                f"iterative solver ({self.method}) did not converge "
                f"after {info} iterations (residual = "
                f"{self.last_residual:.3e}). Increase max_iter, tighten "
                f"drop_tol, or fall back to DirectSparseSolver."
            )
        if info < 0:
            raise RuntimeError(
                f"iterative solver ({self.method}) reported "
                f"illegal-input error code {info}."
            )
        return np.asarray(x).ravel()
