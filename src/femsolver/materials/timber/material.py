"""TimberMaterial -- orthotropic timber with characteristic strengths.

Sign convention follows the rest of femsolver: stress and strain
are tension-positive. Compression strengths are stored as positive
magnitudes for design-code clarity (matching the NDS / EC5 / IS 883
convention where ``F_c`` is a positive number used as an allowable
limit).

Property naming follows EC5 §3 and NDS §4.2:
* Subscript ``0`` = parallel to grain (longitudinal direction)
* Subscript ``90`` = perpendicular to grain (radial / tangential
  combined; EC5 and NDS don't distinguish radial vs tangential for
  most engineering purposes)
* Subscript ``k`` = characteristic (5th-percentile, EC5 convention)
* Subscript ``m`` = mean (used for deflection / serviceability)

For NDS the published "reference" values F_b, F_t, F_c, F_v are
already the 5th-percentile (adjusted) allowables; they correspond
to EC5's ``f_k`` family.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TimberMaterial:
    """Orthotropic timber material with reference design values.

    Parameters
    ----------
    name : str
        Friendly identifier (e.g. "DFL #1", "C24", "Class II").
    species : str
        Species or class family (e.g. "Douglas Fir-Larch", "EC5 C-class").
    grade : str
        Grading designation (e.g. "Select Structural", "C24", "Class II").
    code : {"NDS", "EC5", "IS 883"}
        Source design code -- determines which safety-factor
        convention applies downstream.

    Elastic (mean values)
    ---------------------
    E_0_mean : float (Pa)
        Modulus parallel to grain at mean reference strain.
    E_0_05 : float (Pa), optional
        5th-percentile modulus parallel to grain. Used by EC5 for
        stability checks. Defaults to ``0.67 * E_0_mean`` per EC5 §3.
    E_90_mean : float (Pa)
        Modulus perpendicular to grain. Typically ~1/30 of E_0.
    G_mean : float (Pa)
        Mean shear modulus. Typically ~E_0/16 for softwoods.

    Characteristic strengths (5th-percentile)
    -----------------------------------------
    f_b_k : float (Pa)
        Characteristic bending strength.
    f_t_0_k : float (Pa)
        Tension parallel to grain.
    f_t_90_k : float (Pa)
        Tension perpendicular to grain (usually very low).
    f_c_0_k : float (Pa)
        Compression parallel to grain.
    f_c_90_k : float (Pa)
        Compression perpendicular to grain (bearing).
    f_v_k : float (Pa)
        Shear strength.

    Density
    -------
    density_k : float (kg/m^3)
        Characteristic density (5th-percentile, dry).
    density_mean : float (kg/m^3)
        Mean density.

    Service-class / moisture
    ------------------------
    moisture_content_pct : float
        Reference equilibrium moisture content (typically 12% for
        EC5 service class 1, 19% for NDS dry-service).
    """

    name: str
    species: str
    grade: str
    code: str

    E_0_mean: float
    E_90_mean: float
    G_mean: float

    f_b_k: float
    f_t_0_k: float
    f_t_90_k: float
    f_c_0_k: float
    f_c_90_k: float
    f_v_k: float

    density_k: float
    density_mean: float

    E_0_05: Optional[float] = None
    moisture_content_pct: float = 12.0

    # Aliases for compatibility with the rest of femsolver (E, density)
    # plus convenient design-code formulas.
    def __post_init__(self) -> None:
        if self.E_0_mean <= 0:
            raise ValueError(f"E_0_mean must be positive, got {self.E_0_mean}")
        if self.code not in ("NDS", "EC5", "IS 883"):
            raise ValueError(
                f"code must be 'NDS', 'EC5', or 'IS 883', got {self.code!r}"
            )
        if self.E_0_05 is None:
            # EC5 §3.1.2: E_0,05 = 0.67 * E_0,mean for solid timber,
            # 0.85 for glulam (we use the conservative 0.67 default).
            self.E_0_05 = 0.67 * self.E_0_mean

    @property
    def E(self) -> float:
        """Primary modulus (parallel to grain, mean). Compatible with
        the rest of femsolver's elastic-section API."""
        return self.E_0_mean

    @property
    def nu(self) -> float:
        """Effective Poisson ratio. Wood orthotropic Poisson is
        complex; for fiber-section elastic analysis a typical value
        of 0.3 is used as a default."""
        return 0.30

    @property
    def density(self) -> float:
        """Mean density -- used for BOM / weight per length."""
        return self.density_mean

    @property
    def G(self) -> float:
        """Shear modulus alias."""
        return self.G_mean

    @property
    def f_c_0_to_f_b_ratio(self) -> float:
        """Ratio of compression-parallel to bending strength. Useful
        diagnostic: ~1.0 for high-grade structural lumber, ~0.7 for
        lower grades."""
        return self.f_c_0_k / max(self.f_b_k, 1e-9)

    def __repr__(self) -> str:
        return (
            f"TimberMaterial({self.name!r}, {self.code}, "
            f"E_0={self.E_0_mean/1e9:.1f} GPa, "
            f"f_b={self.f_b_k/1e6:.1f} MPa, "
            f"rho={self.density_mean:.0f} kg/m3)"
        )
