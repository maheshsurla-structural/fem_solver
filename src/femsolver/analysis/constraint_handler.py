"""Constraint handlers — translate MP constraints into modifications of the
assembled (K, F) system.

Two strategies are provided:

- :class:`TransformationHandler` (default) eliminates constrained DOFs by
  building a transformation matrix ``T`` such that ``u_full = T u_eff + g``.
  The reduced system ``(T^T K T) u_eff = T^T (F - K g)`` is positive-definite
  (assuming the original system was) and smaller. This matches OpenSees'
  ``constraints Transformation`` handler.

- :class:`PenaltyHandler` keeps the original DOF set and adds a stiff
  penalty term ``alpha * C^T C`` to ``K`` and ``alpha * C^T g`` to ``F``.
  Easier to implement but introduces ill-conditioning that scales with
  ``alpha``. Useful as a sanity check or fallback.

Both handlers operate on the assembled CSC stiffness matrix and dense force
vector restricted to free DOFs (i.e., after SP constraints have been
applied via the standard ``eqn`` numbering).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from femsolver.constraints.base import BasicConstraint


# ---------------------------------------------------------------------------
# helpers


def _collect_basics(model) -> list[BasicConstraint]:
    """Gather all elementary constraint rows from the model's MP constraints."""
    out: list[BasicConstraint] = []
    for c in model.mp_constraints:
        out.extend(c.basic_constraints(model))
    return out


def _eqn_of(model, node_tag: int, dof: int) -> int:
    return int(model.node(node_tag).eqn[dof])


# ---------------------------------------------------------------------------
# Transformation handler


@dataclass
class _TransformBuild:
    T: sp.csc_matrix
    g_full: np.ndarray
    eff_eqns: np.ndarray  # mapping: column j of T -> equation number of the retained/free DOF
    constrained_eqns: set[int]  # set of free-DOF equation numbers that are slaves


class TransformationHandler:
    """Constraint elimination via a transformation matrix.

    The free-DOF displacement vector ``u`` (size ``neq``) is split into
    *effective* (kept) and *constrained* (eliminated) entries. We build a
    sparse ``T`` of shape ``(neq, n_eff)`` and a constant vector ``g_full``
    of size ``neq`` such that

    .. math::

        u = T u_{\\text{eff}} + g_{\\text{full}}.

    The reduced system is

    .. math::

        T^T K T \\, u_{\\text{eff}} = T^T (F - K g_{\\text{full}}).
    """

    def build(self, model) -> _TransformBuild:
        neq = model.neq
        basics = _collect_basics(model)

        # Resolve which equation numbers are constrained (slaves).
        # A constrained DOF whose own equation is -1 (SP-fixed) is silently
        # dropped — the SP constraint already pins it to zero (or its
        # imposed value, if any).
        constrained_eqns: set[int] = set()
        for b in basics:
            c_eqn = _eqn_of(model, b.c_node, b.c_dof)
            if c_eqn < 0:
                continue
            if c_eqn in constrained_eqns:
                raise RuntimeError(
                    f"DOF (node={b.c_node}, dof={b.c_dof}) is the slave of "
                    "more than one MP constraint"
                )
            constrained_eqns.add(c_eqn)

        # Validate: a retained DOF cannot itself be constrained (no chained
        # constraints in the simple transformation handler).
        for b in basics:
            for r_node, r_dof, _ in b.r_terms:
                r_eqn = _eqn_of(model, r_node, r_dof)
                if r_eqn in constrained_eqns:
                    raise RuntimeError(
                        f"chained MP constraint: retained DOF (node={r_node}, "
                        f"dof={r_dof}) is itself constrained — not supported"
                    )

        # Effective equation set = all free DOFs minus constrained ones.
        eff_mask = np.ones(neq, dtype=bool)
        for e in constrained_eqns:
            eff_mask[e] = False
        eff_eqns = np.flatnonzero(eff_mask).astype(np.int64)
        n_eff = eff_eqns.size

        # Inverse map: equation number -> column index in T (or -1 if not effective)
        col_of = np.full(neq, -1, dtype=np.int64)
        col_of[eff_eqns] = np.arange(n_eff, dtype=np.int64)

        # Build T — start with identity rows for effective DOFs, then add
        # constraint rows.
        rows = list(eff_eqns.tolist())
        cols = list(range(n_eff))
        vals = [1.0] * n_eff

        g_full = np.zeros(neq, dtype=float)

        for b in basics:
            c_eqn = _eqn_of(model, b.c_node, b.c_dof)
            if c_eqn < 0:
                # Slave is SP-fixed; the SP constraint dominates. Note: we
                # do not propagate `g` here because SP already fixes the
                # value (typically to 0).
                continue
            for r_node, r_dof, coeff in b.r_terms:
                r_eqn = _eqn_of(model, r_node, r_dof)
                if r_eqn < 0:
                    # retained DOF is SP-fixed -> contributes 0 (its
                    # prescribed value is zero in the current model)
                    continue
                col = int(col_of[r_eqn])
                if col < 0:
                    # should not happen given validation above
                    raise RuntimeError("internal: retained DOF is not effective")
                rows.append(int(c_eqn))
                cols.append(col)
                vals.append(float(coeff))
            if b.g != 0.0:
                g_full[c_eqn] = b.g

        T = sp.csc_matrix(
            (vals, (rows, cols)), shape=(neq, n_eff)
        )

        return _TransformBuild(
            T=T, g_full=g_full, eff_eqns=eff_eqns, constrained_eqns=constrained_eqns
        )

    @staticmethod
    def apply(K: sp.spmatrix, F: np.ndarray, build: _TransformBuild):
        """Return the reduced ``(K_eff, F_eff)`` system."""
        T = build.T
        g = build.g_full
        K_eff = (T.T @ K @ T).tocsc()
        F_eff = T.T @ (F - K @ g)
        return K_eff, np.asarray(F_eff).ravel()

    @staticmethod
    def recover_full(u_eff: np.ndarray, build: _TransformBuild) -> np.ndarray:
        return build.T @ u_eff + build.g_full


# ---------------------------------------------------------------------------
# Penalty handler


class PenaltyHandler:
    """Penalty enforcement of MP constraints.

    For each scalar constraint ``u_c - sum_j coeff_j * u_r_j = g`` we add
    ``alpha * C^T C`` to the stiffness and ``alpha * C^T g`` to the force.
    The penalty parameter is auto-selected as ``alpha = factor *
    max(|diag(K)|)`` if not supplied; ``factor=1e8`` is a typical compromise
    between accuracy and conditioning for double-precision solves.
    """

    def __init__(self, alpha: float | None = None, factor: float = 1e8):
        self.alpha = alpha
        self.factor = float(factor)

    def apply(self, model, K: sp.spmatrix, F: np.ndarray):
        basics = _collect_basics(model)
        if not basics:
            return K, F

        neq = model.neq
        # Build C as a sparse matrix of shape (n_basic, neq).
        rows: list[int] = []
        cols: list[int] = []
        vals: list[float] = []
        gs: list[float] = []
        n_eff_rows = 0
        for b in basics:
            c_eqn = _eqn_of(model, b.c_node, b.c_dof)
            if c_eqn < 0:
                continue
            row = n_eff_rows
            n_eff_rows += 1
            rows.append(row); cols.append(c_eqn); vals.append(1.0)
            for r_node, r_dof, coeff in b.r_terms:
                r_eqn = _eqn_of(model, r_node, r_dof)
                if r_eqn < 0:
                    continue
                rows.append(row); cols.append(r_eqn); vals.append(-float(coeff))
            gs.append(float(b.g))

        if n_eff_rows == 0:
            return K, F

        C = sp.csr_matrix(
            (vals, (rows, cols)), shape=(n_eff_rows, neq)
        )
        g_vec = np.asarray(gs, dtype=float)

        alpha = self.alpha
        if alpha is None:
            diag_max = float(np.abs(np.asarray(K.diagonal()).ravel()).max())
            if diag_max == 0.0:
                diag_max = 1.0
            alpha = self.factor * diag_max

        K_aug = (K + alpha * (C.T @ C)).tocsc()
        F_aug = F + alpha * (C.T @ g_vec)
        return K_aug, np.asarray(F_aug).ravel()
