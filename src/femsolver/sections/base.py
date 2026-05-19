"""Section abstract base class.

A section abstracts the cross-section response from the beam element.
Concrete subclasses implement :meth:`get_response`, which evaluates the
section forces and tangent stiffness for a given generalized strain.

Strain / force conventions
--------------------------
2-D beam-column section (``n_resultants = 2``):

    e = [eps_axial, kappa_z]^T              (axial strain, curvature about local z)
    s = [N, Mz]^T                           (axial force, bending moment about z)

3-D beam-column section (``n_resultants = 4``):

    e = [eps_axial, kappa_z, kappa_y, gamma_torsion]^T
    s = [N, Mz, My, T]^T

Implementations may carry internal state (plastic strains, damage, fiber
states, etc.). The :meth:`commit_state` and :meth:`revert_state` hooks
mirror the element-level lifecycle and let the section roll its history
forward at the end of a converged Newton step or back up when a step fails
to converge.

The class-level flag :attr:`is_stateful` tells the element whether each
integration point needs its own copy of the section. Stateless sections
(e.g. :class:`ElasticSection2D`) can be safely shared across all IPs;
stateful sections (e.g. fiber sections with per-fiber plasticity) must
be cloned. The element relies on this flag to decide whether to call
:meth:`clone` per integration point.
"""
from __future__ import annotations

import copy
from abc import ABC, abstractmethod

import numpy as np


class SectionBase(ABC):
    """Generalized stress-strain law for a beam-column cross-section."""

    n_resultants: int = 0

    # ``True`` if the section carries history that evolves under loading
    # (plastic strains, damage, fiber states, ...). Stateful sections
    # must be cloned per integration point so that each IP has its own
    # independent state. Default: ``True``, the safe choice for
    # subclasses; stateless sections override to ``False``.
    is_stateful: bool = True

    @abstractmethod
    def get_response(self, e: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(s, ks)`` — section forces and tangent stiffness.

        Parameters
        ----------
        e : ndarray, shape ``(n_resultants,)``
            Generalized section strains.

        Returns
        -------
        s : ndarray, shape ``(n_resultants,)``
            Generalized section forces.
        ks : ndarray, shape ``(n_resultants, n_resultants)``
            Tangent stiffness :math:`\\partial s / \\partial e`.
        """

    # ----------------------------------------------------------------- state
    def commit_state(self) -> None:
        """Persist any iteratively-updated internal state at the end of a
        converged Newton step. Default: no-op.
        """
        return None

    def revert_state(self) -> None:
        """Undo any state changes since the last :meth:`commit_state`.
        Default: no-op.
        """
        return None

    # ----------------------------------------------------------------- clone
    def clone(self) -> "SectionBase":
        """Return an independent copy with its own state.

        Default: deep copy. Stateless sections (``is_stateful = False``)
        may override to return ``self`` for zero-cost sharing across
        integration points, but the default is safe in all cases.
        """
        return copy.deepcopy(self)
