"""Element abstract base class."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Element(ABC):
    """Base element. Subclasses must define `dofs_per_node` and implement
    `K_global` (and optionally `f_eq_global`, `recover`)."""

    dofs_per_node: int = 0
    n_nodes: int = 0

    def __init__(self, tag: int, nodes, material):
        self.tag = int(tag)
        self.node_tags = tuple(int(n) for n in nodes)
        if len(self.node_tags) != self.n_nodes:
            raise ValueError(
                f"{self.__class__.__name__} requires {self.n_nodes} nodes, "
                f"got {len(self.node_tags)}"
            )
        self.material = material
        self._model = None  # set by Model.add_element via bind()

    @property
    def n_dof(self) -> int:
        return self.n_nodes * self.dofs_per_node

    def bind(self, model) -> None:
        self._model = model

    @property
    def model(self):
        if self._model is None:
            raise RuntimeError(
                f"element {self.tag} is not bound to a model — add it via Model.add_element()"
            )
        return self._model

    def node_coords(self) -> np.ndarray:
        """Coordinates of the element's nodes as an (n_nodes, ndm) array."""
        return np.array([self.model.node(t).coords for t in self.node_tags])

    def gather_u(self) -> np.ndarray:
        """Element nodal displacement vector (size = n_dof)."""
        m = self.model
        out = np.empty(self.n_dof)
        k = 0
        for nt in self.node_tags:
            disp = m.node(nt).disp[: self.dofs_per_node]
            out[k : k + self.dofs_per_node] = disp
            k += self.dofs_per_node
        return out

    @abstractmethod
    def K_global(self) -> np.ndarray:
        """Element initial stiffness in global coordinates, shape (n_dof, n_dof).

        For linear elements this is the only stiffness — the tangent equals
        this matrix at every state. Nonlinear elements override
        :meth:`K_tangent_global` to return a state-dependent tangent.
        """

    def K_tangent_global(self) -> np.ndarray:
        """Tangent stiffness at the *current* deformation state.

        Default: returns ``K_global()`` — correct for linear elements where
        the stiffness is independent of the deformation. Nonlinear elements
        (geometric or material) override this.
        """
        return self.K_global()

    def f_int_global(self) -> np.ndarray:
        """Internal resisting force at the *current* deformation state.

        Default: ``K_global() @ u_e`` — correct for linear elements.
        Nonlinear elements override this with their constitutive
        evaluation. Returns a vector of length ``n_dof`` in global coords.
        """
        return self.K_global() @ self.gather_u()

    def M_global(self, *, lumped: bool = False) -> np.ndarray:
        """Element mass matrix in global coordinates, shape (n_dof, n_dof).

        Default: returns zeros, meaning the element contributes no mass.
        Concrete subclasses override to provide consistent mass (default)
        and a row-summed lumped form when ``lumped=True``.
        """
        return np.zeros((self.n_dof, self.n_dof))

    def f_eq_global(self) -> np.ndarray:
        """Equivalent nodal force vector from distributed/body loads."""
        return np.zeros(self.n_dof)

    def recover(self):
        """Compute and store internal element response. Default: no-op.
        Subclasses should override to populate `self.forces`, `self.stresses`,
        etc."""
        return None

    # --------------------------------------------------------------- state
    def commit_state(self) -> None:
        """Persist any iteratively-updated internal state at the end of a
        converged Newton step. Default: no-op. Plasticity / damage models
        will override to roll back-history variables forward.
        """
        return None

    def revert_state(self) -> None:
        """Undo any state changes since the last :meth:`commit_state`.
        Called when a Newton step fails to converge and the analysis backs
        up. Default: no-op.
        """
        return None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(tag={self.tag}, nodes={self.node_tags})"
