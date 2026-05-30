"""Reduced Beam Section (RBS / "dog-bone") -- AISC 358 Cl. 5.

The RBS is a post-Northridge connection detail that intentionally
reduces the flange width of the beam over a localised region near the
column face, so the plastic hinge forms in the reduced region instead
of in the welded beam-to-column joint. The reduced section has a
lower plastic moment ``M_p_RBS < M_p`` and therefore yields first,
protecting the connection.

AISC 358 prescribes the dimensional limits::

    0.50 b_f <= a <= 0.75 b_f       (offset from column face)
    0.65 d   <= b <= 0.85 d         (length of cut)
    0.10 b_f <= c <= 0.25 b_f       (depth of cut on each flange)

The reduced flange width at the centre of the RBS is
``b_f_red = b_f - 2 c``. The reduced plastic section modulus is::

    Z_RBS = Z_x - 2 c t_f (d - t_f)

(Subtracting the missing flange area's contribution to Z_x.)

The reduced plastic moment is ``M_p_RBS = R_y · f_y · Z_RBS`` (with
the expected-yield strength used per AISC 358 / 341).

Maximum probable moment at the column face (used for capacity-design
shear, column moment, and panel-zone strength) is::

    M_pr_face = M_p_RBS · (L_c / (L_c - 2(a + b/2)))

where ``L_c`` is the clear beam span and ``(a + b/2)`` is the distance
from the column face to the centre of the RBS region.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RBSGeometry:
    """RBS dimensions and resulting section properties.

    Attributes
    ----------
    a, b, c : float
        AISC 358 dimensions (m).
    b_f_reduced : float
        Reduced flange width at the centre of the RBS (m).
    Z_RBS : float
        Reduced plastic section modulus (m^3).
    M_p_RBS : float
        Reduced plastic moment using expected yield strength (N·m).
    M_pr_face : float
        Probable moment at the column face (N·m).
    aisc_a_ok, aisc_b_ok, aisc_c_ok : bool
        Whether each dimension lies in the AISC 358 limits.
    """

    a: float
    b: float
    c: float
    b_f_reduced: float
    Z_RBS: float
    M_p_RBS: float
    M_pr_face: float
    aisc_a_ok: bool
    aisc_b_ok: bool
    aisc_c_ok: bool


def reduced_beam_section(
    *,
    f_y: float,
    Z_x: float,
    d: float, b_f: float, t_f: float,
    a: float, b: float, c: float,
    L_clear: float,
    R_y: float = 1.1,
    Cpr: float = 1.15,
) -> RBSGeometry:
    """Compute reduced-section properties for an RBS detail.

    Parameters
    ----------
    f_y : float
        Steel yield stress (Pa).
    Z_x : float
        Plastic section modulus of the full beam about the strong
        axis (m^3).
    d, b_f, t_f : float
        Beam total depth, flange width, flange thickness (m).
    a, b, c : float
        RBS offset, cut length, and cut depth (m).
    L_clear : float
        Beam clear span between column faces (m).
    R_y : float, default 1.1
        Expected-yield ratio (AISC 341 Table A3.1).
    Cpr : float, default 1.15
        Probable-strength factor accounting for strain hardening
        (AISC 358 Eq. 2.4.3-2).

    Returns
    -------
    RBSGeometry
    """
    for name, val in [("f_y", f_y), ("Z_x", Z_x), ("d", d),
                        ("b_f", b_f), ("t_f", t_f),
                        ("a", a), ("b", b), ("c", c),
                        ("L_clear", L_clear), ("R_y", R_y), ("Cpr", Cpr)]:
        if val <= 0.0:
            raise ValueError(f"{name} must be > 0, got {val}")

    b_f_red = b_f - 2.0 * c
    if b_f_red <= 0.0:
        raise ValueError(
            f"cut depth c = {c} too large for flange width {b_f}"
        )
    # Reduced plastic-section modulus (subtract removed flange area)
    Z_RBS = Z_x - 2.0 * c * t_f * (d - t_f)
    M_p_RBS = R_y * f_y * Z_RBS
    # Centre of RBS region distance from column face
    x_RBS = a + b / 2.0
    L_eff = L_clear - 2.0 * x_RBS
    if L_eff <= 0.0:
        raise ValueError(
            "RBS centres overlap (L_eff <= 0); reduce a/b or "
            "lengthen the beam"
        )
    M_pr_face = Cpr * M_p_RBS * L_clear / L_eff
    aisc_a_ok = (0.50 * b_f <= a <= 0.75 * b_f)
    aisc_b_ok = (0.65 * d <= b <= 0.85 * d)
    aisc_c_ok = (0.10 * b_f <= c <= 0.25 * b_f)
    return RBSGeometry(
        a=a, b=b, c=c, b_f_reduced=b_f_red,
        Z_RBS=Z_RBS, M_p_RBS=M_p_RBS, M_pr_face=M_pr_face,
        aisc_a_ok=aisc_a_ok, aisc_b_ok=aisc_b_ok, aisc_c_ok=aisc_c_ok,
    )


def aisc358_recommended_RBS(
    *,
    d: float, b_f: float, t_f: float,
    f_y: float, Z_x: float, L_clear: float,
    R_y: float = 1.1, Cpr: float = 1.15,
) -> RBSGeometry:
    """Build an AISC 358-compliant RBS using midpoint values of the
    permissible dimensional ranges::

        a = 0.625 b_f, b = 0.75 d, c = 0.20 b_f.
    """
    return reduced_beam_section(
        f_y=f_y, Z_x=Z_x,
        d=d, b_f=b_f, t_f=t_f,
        a=0.625 * b_f, b=0.75 * d, c=0.20 * b_f,
        L_clear=L_clear, R_y=R_y, Cpr=Cpr,
    )
