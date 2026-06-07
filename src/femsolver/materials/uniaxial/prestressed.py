"""Pre-strain wrapper for prestressing strand materials (Phase II.13).

A bonded prestressing tendon sits in a concrete cross-section with an
initial **pre-strain** ``epsilon_pe = f_pe / E_p`` (where ``f_pe`` is
the effective prestress after all losses). When the surrounding
concrete strain at the tendon location is zero, the tendon already
carries ``sigma = f_pe`` in tension.

In a fiber section, this is captured by wrapping the strand material
so that the strain it sees is shifted by ``epsilon_pe``:

    sigma = strand_material.get_response(epsilon_concrete + epsilon_pe)

The class below provides exactly that. It is conceptually identical
to OpenSees ``InitStrainMaterial`` and to the ``f_pi`` parameter on
the AASHTO-LRFD strand model.
"""
from __future__ import annotations

from femsolver.materials.uniaxial.base import UniaxialMaterial


class PrestressedUniaxial(UniaxialMaterial):
    """Wrap a uniaxial strand material with a constant strain offset
    representing the effective pre-strain.

    Parameters
    ----------
    base : UniaxialMaterial
        The underlying strand constitutive (Grade 1860 / 270 bilinear,
        Ramberg-Osgood, etc.).
    eps_pe : float
        Pre-strain (positive, tension). For a Grade 270 strand with
        ``f_pe = 1100 MPa`` and ``E_p = 195 GPa``, ``eps_pe = 0.00564``.

    Notes
    -----
    The wrapper is **strain-shift only** -- it does NOT add an initial
    stress to the section response by itself. When the fiber section's
    Newton iteration starts at zero overall strain (``eps_0 = kappa
    = 0``), the strand contribution to the axial force is exactly
    ``A_p * f_pe`` (compression on the concrete), and the user-side
    axial-load balance then drives ``eps_0`` to the correct
    decompression value.
    """

    def __init__(self, base: UniaxialMaterial, eps_pe: float):
        if eps_pe < 0:
            raise ValueError(
                f"eps_pe must be >= 0 (tension), got {eps_pe}"
            )
        self.base = base
        self.eps_pe = float(eps_pe)

    def get_response(self, eps: float) -> tuple[float, float]:
        return self.base.get_response(eps + self.eps_pe)

    def commit_state(self) -> None:
        self.base.commit_state()

    def revert_state(self) -> None:
        self.base.revert_state()

    def clone(self) -> "PrestressedUniaxial":
        return PrestressedUniaxial(self.base.clone(), self.eps_pe)

    def __repr__(self) -> str:
        return (
            f"PrestressedUniaxial(base={self.base!r}, "
            f"eps_pe={self.eps_pe:.5f})"
        )
