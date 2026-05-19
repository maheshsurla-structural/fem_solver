"""Model — the top-level container, equivalent to OpenSees' Domain."""
from __future__ import annotations

from collections import OrderedDict
from typing import Iterable

import numpy as np

from femsolver.core.node import Node


class Model:
    """Container for nodes, elements, materials, constraints, and loads.

    Parameters
    ----------
    ndm : int
        Number of model dimensions (2 or 3).
    ndf : int
        Default number of DOFs per node. Common combinations:
        (2,2)=2D truss, (2,3)=2D frame, (3,3)=3D truss, (3,6)=3D frame.
    """

    def __init__(self, ndm: int, ndf: int):
        if ndm not in (2, 3):
            raise ValueError(f"ndm must be 2 or 3, got {ndm}")
        if ndf < 1:
            raise ValueError(f"ndf must be >= 1, got {ndf}")
        self.ndm = ndm
        self.ndf = ndf
        self._nodes: dict[int, Node] = OrderedDict()
        self._elements: dict[int, "Element"] = OrderedDict()
        self._materials: dict[int, "Material"] = OrderedDict()
        self._mp_constraints: list = []
        self._neq: int = 0  # number of free equations (set by numberer)
        self._numbered: bool = False

    # ------------------------------------------------------------------ nodes
    def add_node(self, tag: int, *coords) -> Node:
        if tag in self._nodes:
            raise ValueError(f"duplicate node tag {tag}")
        if len(coords) == 1 and hasattr(coords[0], "__len__"):
            xs = np.asarray(coords[0], dtype=float).ravel()
        else:
            xs = np.asarray(coords, dtype=float).ravel()
        if xs.size != self.ndm:
            raise ValueError(
                f"node {tag} expected {self.ndm} coords, got {xs.size}"
            )
        node = Node(tag, xs, self.ndf)
        self._nodes[tag] = node
        self._numbered = False
        return node

    def node(self, tag: int) -> Node:
        return self._nodes[tag]

    @property
    def nodes(self):
        return self._nodes

    # --------------------------------------------------------------- elements
    def add_element(self, element) -> None:
        if element.tag in self._elements:
            raise ValueError(f"duplicate element tag {element.tag}")
        for nt in element.node_tags:
            if nt not in self._nodes:
                raise ValueError(f"element {element.tag} references unknown node {nt}")
        element.bind(self)
        self._elements[element.tag] = element
        self._numbered = False

    def element(self, tag: int):
        return self._elements[tag]

    @property
    def elements(self):
        return self._elements

    # -------------------------------------------------------------- materials
    def add_material(self, material) -> None:
        if material.tag in self._materials:
            raise ValueError(f"duplicate material tag {material.tag}")
        self._materials[material.tag] = material

    def material(self, tag: int):
        return self._materials[tag]

    # ------------------------------------------------------------ constraints
    def fix(self, node_tag: int, mask: Iterable[int]) -> None:
        """Apply single-point constraints (boundary conditions) at a node."""
        self._nodes[node_tag].fix(mask)
        self._numbered = False

    def add_mp_constraint(self, constraint) -> None:
        self._mp_constraints.append(constraint)
        self._numbered = False

    @property
    def mp_constraints(self):
        return self._mp_constraints

    # convenience constructors that mirror common OpenSees commands ------
    def equal_dof(self, retained: int, constrained: int, dofs: Iterable[int]):
        from femsolver.constraints import EqualDOF
        c = EqualDOF(retained=retained, constrained=constrained, dofs=dofs)
        self.add_mp_constraint(c)
        return c

    def rigid_link(self, retained: int, constrained: int, kind: str = "beam"):
        from femsolver.constraints import RigidLink
        c = RigidLink(retained=retained, constrained=constrained, kind=kind)
        self.add_mp_constraint(c)
        return c

    def rigid_diaphragm(self, master: int, slaves: Iterable[int], perp_dir: int = 2):
        from femsolver.constraints import RigidDiaphragm
        c = RigidDiaphragm(master=master, slaves=slaves, perp_dir=perp_dir)
        self.add_mp_constraint(c)
        return c

    # ---------------------------------------------------------------- loading
    def add_nodal_load(self, node_tag: int, load) -> None:
        self._nodes[node_tag].add_load(load)

    def clear_loads(self) -> None:
        for n in self._nodes.values():
            n._load[:] = 0.0
        for e in self._elements.values():
            if hasattr(e, "clear_distributed_loads"):
                e.clear_distributed_loads()

    # ------------------------------------------------------------ DOF numbering
    def number_dofs(self) -> int:
        """Assign equation numbers to free DOFs. Fixed DOFs get -1.

        Returns the number of free equations.
        """
        eq = 0
        for node in self._nodes.values():
            for i in range(node.ndf):
                if node.fixity[i]:
                    node.eqn[i] = -1
                else:
                    node.eqn[i] = eq
                    eq += 1
        self._neq = eq
        self._numbered = True
        return eq

    @property
    def neq(self) -> int:
        if not self._numbered:
            self.number_dofs()
        return self._neq

    def element_dof_map(self, element) -> np.ndarray:
        """Return the global equation numbers for the DOFs of an element."""
        if not self._numbered:
            self.number_dofs()
        dofs = []
        for nt in element.node_tags:
            n = self._nodes[nt]
            dofs.extend(n.eqn[: element.dofs_per_node].tolist())
        return np.asarray(dofs, dtype=np.int64)

    # ---------------------------------------------------------------- summary
    def __repr__(self) -> str:
        return (
            f"Model(ndm={self.ndm}, ndf={self.ndf}, "
            f"nodes={len(self._nodes)}, elements={len(self._elements)}, "
            f"materials={len(self._materials)})"
        )

    def reset_results(self) -> None:
        for n in self._nodes.values():
            n.reset_results()
