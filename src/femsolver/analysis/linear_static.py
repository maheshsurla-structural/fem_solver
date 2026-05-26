"""Linear static analysis."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from femsolver.analysis.assembler import (
    assemble_force,
    assemble_reactions,
    assemble_stiffness,
)
from femsolver.analysis.constraint_handler import (
    PenaltyHandler,
    TransformationHandler,
)
from femsolver.analysis.solvers import DirectSparseSolver, LinearSolver
from femsolver.numerics.dof_numbering import rcm_renumber


class LinearStaticAnalysis:
    """Solve K u = F once, then recover element responses and reactions.

    Parameters
    ----------
    model : Model
    constraints : str or handler instance, optional
        Constraint enforcement strategy. ``"transformation"`` (default) or
        ``"penalty"``, or a handler instance. Has no effect when the model
        has no MP constraints.
    numberer : {"default", "rcm"}, optional
        DOF numbering scheme. ``"default"`` (the model's sequential
        numbering) or ``"rcm"`` (reverse Cuthill-McKee on the node
        adjacency graph — typically reduces bandwidth and direct-solve
        fill-in on structured meshes).
    solver : LinearSolver, optional
        Linear solver used for ``K u = F``. Defaults to a
        :class:`DirectSparseSolver`. Pass an
        :class:`~femsolver.analysis.solvers.IterativeSolver` instance to
        use CG / GMRES with preconditioning instead.
    """

    def __init__(self, model, constraints="transformation",
                 numberer: str = "default",
                 solver: LinearSolver | None = None):
        self.model = model
        if isinstance(constraints, str):
            key = constraints.lower()
            if key in ("transformation", "transform"):
                self.handler = TransformationHandler()
            elif key == "penalty":
                self.handler = PenaltyHandler()
            else:
                raise ValueError(
                    f"unknown constraint handler {constraints!r}; "
                    "expected 'transformation' or 'penalty'"
                )
        else:
            self.handler = constraints
        if numberer not in ("default", "rcm"):
            raise ValueError(
                f"unknown numberer {numberer!r}; expected 'default' or 'rcm'"
            )
        self.numberer = numberer
        self.solver: LinearSolver = solver or DirectSparseSolver()
        self.K: sp.csc_matrix | None = None
        self.F: np.ndarray | None = None
        self.u: np.ndarray | None = None

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

        K, elem_K_list = assemble_stiffness(m, return_element_K=True)
        F = assemble_force(m, elem_K_list=elem_K_list)
        self.K, self.F = K, F

        has_mp = bool(m.mp_constraints)
        if has_mp and isinstance(self.handler, TransformationHandler):
            build = self.handler.build(m)
            K_solve, F_solve = self.handler.apply(K, F, build)
            self._check_diagonal(K_solve, build=build)
            u_eff = self.solver.solve(K_solve, F_solve)
            u = self.handler.recover_full(np.atleast_1d(u_eff), build)
        elif has_mp and isinstance(self.handler, PenaltyHandler):
            K_solve, F_solve = self.handler.apply(m, K, F)
            self._check_diagonal(K_solve)
            u = self.solver.solve(K_solve, F_solve)
        else:
            self._check_diagonal(K)
            u = self.solver.solve(K, F)

        u = np.atleast_1d(np.asarray(u).ravel())

        # scatter back to nodes (constrained DOFs get their recovered value)
        for node in m.nodes.values():
            for i in range(node.ndf):
                eq = node.eqn[i]
                node.disp[i] = u[eq] if eq >= 0 else 0.0
        self.u = u

        # element response and reactions (operate on the recovered full u)
        for e in m.elements.values():
            e.recover()
        assemble_reactions(m, elem_K_list=elem_K_list)

        return {
            "neq": int(m.neq),
            "u_norm": float(np.linalg.norm(u)),
            "F_norm": float(np.linalg.norm(F)),
            "n_constraints": len(m.mp_constraints),
        }

    # ---------------------------------------------------------------- diagnostics
    def _check_diagonal(self, K, *, build=None) -> None:
        """Translate any zero diagonals of the matrix to be solved into a
        helpful error message."""
        diag = np.asarray(K.diagonal()).ravel()
        zero_diag = np.where(diag == 0.0)[0]
        if not zero_diag.size:
            return
        m = self.model
        if build is not None:
            # zero_diag is in the reduced index — translate via eff_eqns
            zero_eqns = build.eff_eqns[zero_diag]
        else:
            zero_eqns = zero_diag
        zset = set(int(z) for z in zero_eqns)
        offenders: list[tuple[int, int]] = []
        for n in m.nodes.values():
            for i in range(n.ndf):
                if int(n.eqn[i]) in zset:
                    offenders.append((n.tag, i))
        msg = (
            "stiffness matrix has zero diagonal entries — these DOFs have no "
            "stiffness contribution from any element. Likely you have nodes "
            "with rotational DOFs unconstrained (e.g., truss-only nodes in a "
            "frame model). Offending (node_tag, dof_index): "
            f"{offenders[:20]}"
        )
        raise RuntimeError(msg)
