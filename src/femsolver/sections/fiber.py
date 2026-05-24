"""2-D fiber cross-section: distributed plasticity from individual fibers.

A fiber section discretizes the cross-section into a list of small
strips of area, each holding a uniaxial constitutive law. The section
response is the area-weighted aggregate of fiber responses — axial
force from sigma * area, bending moment from -y * sigma * area, plus
their tangent counterparts.

Strain / force ordering (matches :class:`ElasticSection2D`):

    e = [eps_axial, kappa_z]^T
    s = [N, Mz]^T

The tangent stiffness picks up an off-diagonal term once different
fibers carry different tangent moduli (e.g., when the compression side
yields before the tension side):

    ks = [[ EA,  -ES ],
          [-ES,  EI  ]]

with ``EA = sum(Et_f A_f)``, ``ES = sum(Et_f A_f y_f)``,
``EI = sum(Et_f A_f y_f^2)``. This off-diagonal coupling is what gives
fiber sections their P-M interaction post-yield — pure bending develops
an axial-force component and vice versa.

State
-----
Each fiber holds an independent :class:`UniaxialMaterial`. The section
forwards lifecycle calls (:meth:`commit_state` / :meth:`revert_state`)
to every fiber's material. :meth:`clone` deep-copies the fibers so each
integration point of a beam element gets its own independent state.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from femsolver.materials.uniaxial.base import UniaxialMaterial
from femsolver.sections.base import SectionBase


@dataclass
class Fiber:
    """A single fiber: position, area, and uniaxial material.

    ``y`` is measured from the section centroid, perpendicular to the
    bending axis (positive on the +y side of the local frame).
    ``z`` is the out-of-plane coordinate; it is unused by
    :class:`FiberSection2D` but kept on the dataclass so the same
    Fiber type can later feed a 3-D fiber section.
    """
    y: float
    z: float
    area: float
    material: UniaxialMaterial


class FiberSection2D(SectionBase):
    """Fiber-discretized 2-D cross-section."""

    n_resultants = 2
    is_stateful = True   # per-fiber materials carry history

    def __init__(self, fibers: list[Fiber]):
        if not fibers:
            raise ValueError("FiberSection2D needs at least one fiber")
        for f in fibers:
            if f.area <= 0.0:
                raise ValueError(f"fiber at y={f.y} has non-positive area")
        self.fibers = list(fibers)

    # --------------------------------------------------------- gross props
    @property
    def gross_area(self) -> float:
        return float(sum(f.area for f in self.fibers))

    @property
    def centroid_y(self) -> float:
        A = self.gross_area
        if A == 0.0:
            return 0.0
        return float(sum(f.area * f.y for f in self.fibers) / A)

    @property
    def gross_Iz(self) -> float:
        """Second moment of area about the centroidal z-axis (parallel-
        axis-theorem-corrected)."""
        yc = self.centroid_y
        return float(sum(f.area * (f.y - yc) ** 2 for f in self.fibers))

    # -------------------------------------------------------- response
    def get_response(self, e: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Aggregate fiber responses into section forces and tangent.

        Fiber strain at offset ``y`` from the centroid:

            eps_f(y) = eps_axial - y * kappa_z

        (the minus sign is the standard plane-sections beam kinematics:
        positive curvature compresses the +y fibers).
        """
        eps_a = float(e[0])
        kappa = float(e[1])
        N = 0.0
        M = 0.0
        EA = 0.0
        ES = 0.0
        EI = 0.0
        for f in self.fibers:
            eps_f = eps_a - f.y * kappa
            sigma, Et = f.material.get_response(eps_f)
            dF = sigma * f.area
            dEt_A = Et * f.area
            N += dF
            M -= f.y * dF
            EA += dEt_A
            ES += dEt_A * f.y
            EI += dEt_A * f.y * f.y
        s = np.array([N, M])
        ks = np.array([[EA, -ES], [-ES, EI]])
        return s, ks

    # -------------------------------------------------------- lifecycle
    def commit_state(self) -> None:
        for f in self.fibers:
            f.material.commit_state()

    def revert_state(self) -> None:
        for f in self.fibers:
            f.material.revert_state()

    # ------------------------------------------------------------- clone
    def clone(self) -> "FiberSection2D":
        """Deep copy of fibers with independent material state."""
        return FiberSection2D([
            Fiber(y=f.y, z=f.z, area=f.area, material=f.material.clone())
            for f in self.fibers
        ])

    # --------------------------------------------------------- factories
    @classmethod
    def rectangular(
        cls,
        width: float,
        height: float,
        n_fibers: int,
        material: UniaxialMaterial,
        *,
        centroid_y: float = 0.0,
    ) -> "FiberSection2D":
        """Build a rectangular section discretized into ``n_fibers``
        equal-area strips along the height direction (``y``).

        Parameters
        ----------
        width : float
            Cross-section width (in z-direction).
        height : float
            Cross-section height (in y-direction). Strips are stacked
            along this direction.
        n_fibers : int
            Number of strips, each of area ``width * height / n_fibers``.
        material : UniaxialMaterial
            Material assigned to every fiber. Each fiber gets its own
            cloned copy so state is independent.
        centroid_y : float, default 0.0
            Y-coordinate of the section centroid; useful when the
            section is offset from the beam neutral axis.
        """
        if width <= 0.0 or height <= 0.0:
            raise ValueError("width and height must be positive")
        if n_fibers < 2:
            raise ValueError("need at least 2 fibers")
        strip_h = height / n_fibers
        fiber_area = width * strip_h
        # strips centred at y_i, evenly spaced from -h/2 to +h/2 about centroid
        y_top = centroid_y + 0.5 * height - 0.5 * strip_h
        fibers = [
            Fiber(
                y=y_top - i * strip_h,
                z=0.0,
                area=fiber_area,
                material=material.clone(),
            )
            for i in range(n_fibers)
        ]
        return cls(fibers)

    def __repr__(self) -> str:
        return (
            f"FiberSection2D(n_fibers={len(self.fibers)}, "
            f"A={self.gross_area:g}, Iz={self.gross_Iz:g})"
        )


# ============================================================ 3-D ==

class FiberSection3D(SectionBase):
    """Fiber-discretized 3-D cross-section.

    Strain ordering matches :class:`ElasticSection3D` and the
    :class:`~femsolver.elements.beam.BeamColumn3D` B-matrix:

        e = [eps_axial, kappa_z, kappa_y, gamma_torsion]
        s = [N, Mz, My, T]

    Fiber kinematics (Bernoulli plane sections):

        eps_f(y, z) = eps_axial - y * kappa_z + z * kappa_y

    Section forces from area integration over fibers:

        N  = sum sigma_f * A_f
        Mz = -sum y_f * sigma_f * A_f          (positive Mz compresses +y)
        My = +sum z_f * sigma_f * A_f          (positive My stretches +z)

    Torsion is **uncoupled** at the section level: the user supplies a
    constant ``GJ``, and torsional response is ``T = GJ * gamma``.
    Coupling axial / bending with torsion via warping is a substantial
    extension (warping degree of freedom, sectorial coordinate, etc.)
    deferred to a future phase.

    Tangent stiffness picks up cross-coupling terms once any fiber's
    tangent ``Et_f`` differs from the others (e.g. one side yields
    while the other stays elastic). For a symmetric elastic section
    the matrix is diagonal; after asymmetric yielding the off-diagonal
    blocks ``ES_z, ES_y, EI_yz`` become non-zero, producing the
    classical P-Mz-My interaction.
    """

    n_resultants = 4
    is_stateful = True

    def __init__(self, fibers: list[Fiber], *, GJ: float):
        if not fibers:
            raise ValueError("FiberSection3D needs at least one fiber")
        for f in fibers:
            if f.area <= 0.0:
                raise ValueError(
                    f"fiber at (y={f.y}, z={f.z}) has non-positive area"
                )
        if GJ <= 0.0:
            raise ValueError("GJ must be positive")
        self.fibers = list(fibers)
        self.GJ = float(GJ)

    # --------------------------------------------------------- gross props
    @property
    def gross_area(self) -> float:
        return float(sum(f.area for f in self.fibers))

    @property
    def centroid_y(self) -> float:
        A = self.gross_area
        if A == 0.0:
            return 0.0
        return float(sum(f.area * f.y for f in self.fibers) / A)

    @property
    def centroid_z(self) -> float:
        A = self.gross_area
        if A == 0.0:
            return 0.0
        return float(sum(f.area * f.z for f in self.fibers) / A)

    @property
    def gross_Iz(self) -> float:
        """Centroidal second moment about z."""
        yc = self.centroid_y
        return float(sum(f.area * (f.y - yc) ** 2 for f in self.fibers))

    @property
    def gross_Iy(self) -> float:
        """Centroidal second moment about y."""
        zc = self.centroid_z
        return float(sum(f.area * (f.z - zc) ** 2 for f in self.fibers))

    @property
    def gross_Iyz(self) -> float:
        """Centroidal product of inertia."""
        yc = self.centroid_y
        zc = self.centroid_z
        return float(sum(
            f.area * (f.y - yc) * (f.z - zc) for f in self.fibers
        ))

    @property
    def gross_J(self) -> float:
        """The user-supplied torsional constant J. (Returned as
        ``GJ / G`` is only meaningful if a material's ``G`` is in
        scope; we just expose the input ``J`` as
        ``J = GJ / G_assumed``, but for simplicity we expose ``GJ``
        directly via :attr:`GJ` and ``J = GJ`` is the wrong dimension.
        BeamColumn3D reads ``J`` via the section's ``J`` attribute, so
        we expose that as ``J = GJ`` divided by an assumed reference
        ``G``. Cleaner: the element should read ``GJ`` directly. We
        expose ``self.J`` purely so the element's legacy attribute-
        sniff (``hasattr(section, 'J')``) works; numerically the
        element uses ``GJ`` from the section's ``get_response``.)
        """
        # ``J`` is exposed as a placeholder; the element uses GJ via
        # the tangent in get_response(), so the precise value here
        # doesn't actually drive any numerics. Returning ``GJ`` (a
        # stiffness with wrong units) would mislead downstream code,
        # so return None — but BeamColumn3D's hasattr-sniff in Phase 5.5
        # branches on ``is_stateful`` before checking the attribute,
        # so we never actually read it for fiber sections. Set ``J``
        # to a dummy zero just so the attribute exists.
        return 0.0

    @property
    def J(self) -> float:
        """Placeholder so duck-type checks pass — see :attr:`gross_J`."""
        return 0.0

    # ---------------------------------------------------- compatibility
    @property
    def A(self) -> float:
        """Alias for :attr:`gross_area` so duck-typed elastic-section
        sniff also passes for fiber sections."""
        return self.gross_area

    @property
    def Iz(self) -> float:
        return self.gross_Iz

    @property
    def Iy(self) -> float:
        return self.gross_Iy

    # ------------------------------------------------------- response
    def get_response(self, e: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Aggregate fiber responses + uncoupled torsion into section
        forces and tangent."""
        eps_a = float(e[0])
        kappa_z = float(e[1])
        kappa_y = float(e[2])
        gamma = float(e[3])
        N = 0.0
        Mz = 0.0
        My = 0.0
        # Tangent contributions (axial-bending block; torsion uncoupled).
        EA = 0.0
        ES_z = 0.0    # = sum y_f Et_f A_f
        ES_y = 0.0    # = sum z_f Et_f A_f
        EI_z = 0.0    # = sum y_f^2 Et_f A_f
        EI_y = 0.0    # = sum z_f^2 Et_f A_f
        EI_yz = 0.0   # = sum y_f z_f Et_f A_f
        for f in self.fibers:
            eps_f = eps_a - f.y * kappa_z + f.z * kappa_y
            sigma, Et = f.material.get_response(eps_f)
            dF = sigma * f.area
            dEt_A = Et * f.area
            N += dF
            Mz -= f.y * dF
            My += f.z * dF
            EA += dEt_A
            ES_z += dEt_A * f.y
            ES_y += dEt_A * f.z
            EI_z += dEt_A * f.y * f.y
            EI_y += dEt_A * f.z * f.z
            EI_yz += dEt_A * f.y * f.z
        # Torsion contribution (uncoupled).
        T = self.GJ * gamma
        s = np.array([N, Mz, My, T])
        ks = np.array([
            [EA,    -ES_z,   ES_y,    0.0   ],
            [-ES_z,  EI_z,  -EI_yz,   0.0   ],
            [ES_y,  -EI_yz,  EI_y,    0.0   ],
            [0.0,    0.0,    0.0,    self.GJ],
        ])
        return s, ks

    # -------------------------------------------------------- lifecycle
    def commit_state(self) -> None:
        for f in self.fibers:
            f.material.commit_state()

    def revert_state(self) -> None:
        for f in self.fibers:
            f.material.revert_state()

    # ------------------------------------------------------------- clone
    def clone(self) -> "FiberSection3D":
        """Deep copy with independent fiber-material state."""
        return FiberSection3D(
            [
                Fiber(y=f.y, z=f.z, area=f.area, material=f.material.clone())
                for f in self.fibers
            ],
            GJ=self.GJ,
        )

    # --------------------------------------------------------- factories
    @classmethod
    def rectangular(
        cls,
        width_y: float,
        width_z: float,
        n_y: int,
        n_z: int,
        material: UniaxialMaterial,
        *,
        GJ: float,
        centroid_y: float = 0.0,
        centroid_z: float = 0.0,
    ) -> "FiberSection3D":
        """Build a rectangle of dimensions ``width_y`` (in y) by
        ``width_z`` (in z), discretised into ``n_y x n_z`` equal-area
        fibers. ``GJ`` is the torsional stiffness (user-supplied;
        ``G * J_StVenant`` for solid rectangles).

        Each fiber is centred at ``(y_i, z_j)`` where
        ``y_i in (-width_y/2, ..., +width_y/2)``,
        ``z_j in (-width_z/2, ..., +width_z/2)`` (plus the section-
        centroid offset).
        """
        if width_y <= 0.0 or width_z <= 0.0:
            raise ValueError("widths must be positive")
        if n_y < 2 or n_z < 2:
            raise ValueError("need at least 2 fibers in each direction")
        dy = width_y / n_y
        dz = width_z / n_z
        fiber_area = dy * dz
        y_top = centroid_y + 0.5 * width_y - 0.5 * dy
        z_top = centroid_z + 0.5 * width_z - 0.5 * dz
        fibers = [
            Fiber(
                y=y_top - i * dy,
                z=z_top - j * dz,
                area=fiber_area,
                material=material.clone(),
            )
            for i in range(n_y)
            for j in range(n_z)
        ]
        return cls(fibers, GJ=GJ)

    def __repr__(self) -> str:
        return (
            f"FiberSection3D(n_fibers={len(self.fibers)}, "
            f"A={self.gross_area:g}, Iz={self.gross_Iz:g}, "
            f"Iy={self.gross_Iy:g}, GJ={self.GJ:g})"
        )
