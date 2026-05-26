"""Concrete uniaxial constitutive models.

Two models are implemented, both compatible with the
:class:`UniaxialMaterial` interface (and hence with
:class:`FiberSection2D` / :class:`FiberSection3D`):

* :class:`ConcreteKentPark` -- Kent-Park-Scott monotonic envelope
  (parabolic ascent, linear descent to a residual crushing stress)
  combined with Karsan-Jirsa cyclic unloading / reloading and zero
  tensile strength. The workhorse for unconfined or confined RC
  fiber sections in seismic analysis (OpenSees `Concrete01`-equivalent).

* :class:`ConcreteMander` -- Popovics monotonic curve (Mander 1988)
  parameterised by confined peak stress and strain. Smoother than
  Kent-Park and better suited to confined concrete in transverse
  reinforcement. Tension is taken as zero. Cyclic behaviour shares
  the Karsan-Jirsa unloading scheme.

Sign convention
---------------
Both classes accept their parameters as **positive magnitudes**
(``fpc = 30e6`` for 30 MPa peak compression). Internally the model
stores signed quantities (compression negative), so the user gets
stresses and tangents in the standard solid-mechanics convention:

    eps < 0 (compression)  ->  sigma <= 0
    eps > 0 (tension)      ->  sigma = 0

The model state remembers the most-compressive strain ever reached;
loading beyond that point follows the monotonic envelope, smaller
strains follow the Karsan-Jirsa unloading line, and any positive
(tensile) strain returns zero stress.
"""
from __future__ import annotations

import math

from femsolver.materials.uniaxial.base import UniaxialMaterial


# ============================================================ Kent-Park

class ConcreteKentPark(UniaxialMaterial):
    """Kent-Park-Scott concrete with Karsan-Jirsa cyclic behaviour.

    Parameters
    ----------
    fpc : float
        Peak compressive strength (positive magnitude).
    eps_c0 : float
        Strain at the peak (positive magnitude, typically 0.002).
    fpcu : float
        Residual crushing stress at ``eps_cu`` (positive magnitude;
        ``0 <= fpcu <= fpc``). Pass ``0`` for full crushing (concrete
        loses all strength once it crushes).
    eps_cu : float
        Crushing strain (positive magnitude, typically 0.003-0.005).
        Must satisfy ``eps_cu > eps_c0``.

    Notes
    -----
    The initial modulus implied by the parabola is ``E0 = 2 fpc / eps_c0``,
    which for typical RC values is in the 25-35 GPa range -- consistent
    with the ACI ``57000 sqrt(fc')`` rule of thumb.
    """

    def __init__(self, fpc: float, eps_c0: float, fpcu: float,
                 eps_cu: float):
        if fpc <= 0.0:
            raise ValueError(f"fpc must be positive, got {fpc}")
        if eps_c0 <= 0.0:
            raise ValueError(f"eps_c0 must be positive, got {eps_c0}")
        if not (0.0 <= fpcu <= fpc):
            raise ValueError(
                f"fpcu must satisfy 0 <= fpcu <= fpc, got fpcu={fpcu} "
                f"and fpc={fpc}"
            )
        if eps_cu <= eps_c0:
            raise ValueError(
                f"eps_cu must be strictly greater than eps_c0, got "
                f"eps_cu={eps_cu}, eps_c0={eps_c0}"
            )
        # Store signed (compression negative)
        self.fpc = -float(fpc)
        self.eps_c0 = -float(eps_c0)
        self.fpcu = -float(fpcu)
        self.eps_cu = -float(eps_cu)
        # Initial modulus (positive). The chord-to-tangent ratio at the
        # parabola origin is exactly 2 (twice the secant to the peak).
        self.E0 = 2.0 * fpc / eps_c0
        # ----- state -----
        # Most-compressive strain ever committed (negative or zero).
        self.eps_min_committed: float = 0.0
        self.eps_min_trial: float = 0.0
        self.sigma_trial: float = 0.0
        self.Et_trial: float = self.E0

    # ------------------------------------------------------ envelope
    def _envelope(self, eps: float) -> tuple[float, float]:
        """Monotonic compression envelope. Returns ``(sigma, Et)``.

        ``eps`` is *signed* (negative in compression). For strictly
        tensile strain returns ``(0, 0)``. At ``eps = 0`` returns the
        compression-side tangent ``E0`` so callers (e.g. the assembler
        building the initial K) see a non-singular initial stiffness.
        """
        if eps > 0.0:
            return 0.0, 0.0
        if eps >= self.eps_c0:
            # Parabolic ascent. r = eps/eps_c0 is positive (both negative).
            r = eps / self.eps_c0
            sigma = self.fpc * (2.0 * r - r * r)
            Et = self.fpc * 2.0 * (1.0 - r) / self.eps_c0     # >= 0
            return sigma, Et
        if eps >= self.eps_cu:
            slope = (self.fpcu - self.fpc) / (self.eps_cu - self.eps_c0)
            sigma = self.fpc + slope * (eps - self.eps_c0)
            return sigma, slope
        return self.fpcu, 0.0

    # ------------------------------------------------------ plastic offset
    def _plastic_offset(self, eps_min: float) -> float:
        """Karsan-Jirsa plastic strain at which the unloading line
        reaches zero stress.

        ``eps_p / eps_c0 = 0.145 r^2 + 0.13 r``, where ``r = eps_min/eps_c0``.
        Clamped at ``r <= 2`` to avoid pathological extrapolation past
        the crushing strain.
        """
        if eps_min >= 0.0:
            return 0.0
        r = eps_min / self.eps_c0
        if r > 2.0:
            r = 2.0
        return self.eps_c0 * (0.145 * r * r + 0.13 * r)

    # ------------------------------------------------------ get_response
    def get_response(self, eps: float) -> tuple[float, float]:
        eps = float(eps)
        eps_min = self.eps_min_committed

        # No prior compression history
        if eps_min == 0.0:
            if eps > 0.0:
                sigma, Et = 0.0, 0.0
                self.eps_min_trial = 0.0
            else:
                # eps <= 0: follow envelope (eps == 0 returns E0)
                sigma, Et = self._envelope(eps)
                self.eps_min_trial = eps
            self.sigma_trial = sigma
            self.Et_trial = Et
            return sigma, Et

        # Prior compression: cycle through three regions.
        sigma_min, _ = self._envelope(eps_min)
        eps_p = self._plastic_offset(eps_min)

        if eps <= eps_min:
            # Further into compression -> on the envelope.
            sigma, Et = self._envelope(eps)
            self.eps_min_trial = eps
        elif eps >= eps_p:
            # Above the unloading line: open gap or tension.
            sigma, Et = 0.0, 0.0
            self.eps_min_trial = eps_min
        else:
            # On the linear unload/reload line from (eps_min, sigma_min)
            # to (eps_p, 0).
            E_unload = sigma_min / (eps_min - eps_p)
            sigma = E_unload * (eps - eps_p)
            Et = E_unload
            self.eps_min_trial = eps_min

        self.sigma_trial = sigma
        self.Et_trial = Et
        return sigma, Et

    # ------------------------------------------------------ state
    def commit_state(self) -> None:
        self.eps_min_committed = self.eps_min_trial

    def revert_state(self) -> None:
        self.eps_min_trial = self.eps_min_committed

    def __repr__(self) -> str:
        return (
            f"ConcreteKentPark(fpc={-self.fpc:g}, eps_c0={-self.eps_c0:g}, "
            f"fpcu={-self.fpcu:g}, eps_cu={-self.eps_cu:g})"
        )


# ============================================================ Mander

class ConcreteMander(UniaxialMaterial):
    """Mander confined-concrete model (Popovics monotonic curve) with
    optional cyclic strength degradation (Chang-Mander style).

    Parameters
    ----------
    fpc : float
        (Confined or unconfined) peak compressive strength (positive
        magnitude). For confined concrete this is ``fcc'``; for
        unconfined the value typically rises 30-100% above the
        cylinder strength ``fc'``.
    eps_c0 : float
        Strain at the peak (positive magnitude). For Mander confined
        concrete this is typically larger than the unconfined value:
        ``eps_cc = eps_c0 * [1 + 5 (fcc/fc - 1)]``.
    Ec : float, optional
        Initial elastic modulus (positive). If ``None`` uses
        ``4700 sqrt(fpc[MPa]) * 1e6`` (ACI rule of thumb assuming
        ``fpc`` is in Pa).
    damage_factor : float, default 0.0
        Cyclic strength-degradation rate. The effective peak
        compressive strength is

            fpc_eff = fpc / (1 + damage_factor * alpha / eps_c0)

        where ``alpha`` is the cumulative compressive plastic
        excursion magnitude beyond ``eps_c0``. ``damage_factor = 0``
        disables degradation (back-compat). Typical values for
        cyclic confined concrete: 0.05--0.2.
    min_strength_ratio : float, default 0.2
        Lower bound on ``fpc_eff / fpc``. Prevents the degraded
        strength from collapsing below a physical residual.

    Notes
    -----
    The envelope is

        sigma = fpc * x * r / (r - 1 + x^r)

    with ``x = eps / eps_c0`` (ratio of compressive strains, both
    negative -> positive number), ``r = Ec / (Ec - E_sec)``, and
    ``E_sec = fpc / eps_c0``. The Popovics curve smoothly transitions
    from initial-tangent ``Ec`` at the origin, peaks at ``(eps_c0,
    fpc)``, and softens beyond. Tension and cyclic unloading match the
    Kent-Park model.

    With ``damage_factor > 0``, each new compressive peak deeper than
    the previous one adds to ``alpha`` and shrinks ``fpc`` for the
    next envelope evaluation -- the canonical Chang-Mander cyclic
    softening for confined concrete under repeated seismic loading.
    """

    def __init__(self, fpc: float, eps_c0: float, Ec: float | None = None,
                 damage_factor: float = 0.0,
                 min_strength_ratio: float = 0.2):
        if fpc <= 0.0:
            raise ValueError(f"fpc must be positive, got {fpc}")
        if eps_c0 <= 0.0:
            raise ValueError(f"eps_c0 must be positive, got {eps_c0}")
        if damage_factor < 0.0:
            raise ValueError(f"damage_factor must be >= 0, got {damage_factor}")
        if not (0.0 < min_strength_ratio <= 1.0):
            raise ValueError(
                f"min_strength_ratio must be in (0, 1], "
                f"got {min_strength_ratio}"
            )
        if Ec is None:
            Ec = 4700.0 * math.sqrt(fpc / 1.0e6) * 1.0e6
        if Ec <= 0.0:
            raise ValueError(f"Ec must be positive, got {Ec}")
        E_sec = fpc / eps_c0
        if Ec <= E_sec:
            raise ValueError(
                f"Ec ({Ec:g}) must exceed secant E_sec = fpc/eps_c0 "
                f"({E_sec:g}). Equivalently, the Popovics shape "
                f"parameter r must be finite."
            )
        # Signed storage
        self.fpc = -float(fpc)
        self.eps_c0 = -float(eps_c0)
        self.E0 = float(Ec)
        self.r = Ec / (Ec - E_sec)
        self.damage_factor = float(damage_factor)
        self.min_strength_ratio = float(min_strength_ratio)
        # State
        self.eps_min_committed: float = 0.0
        self.eps_min_trial: float = 0.0
        self.alpha_committed: float = 0.0     # cumulative compressive excursion
        self.alpha_trial: float = 0.0
        self.sigma_trial: float = 0.0
        self.Et_trial: float = self.E0

    # ------------------------------------------------------ degradation
    def _strength_ratio(self, alpha: float) -> float:
        if self.damage_factor == 0.0 or alpha <= 0.0:
            return 1.0
        ratio = 1.0 / (1.0 + self.damage_factor * alpha / abs(self.eps_c0))
        return max(ratio, self.min_strength_ratio)

    # ------------------------------------------------------ envelope
    def _envelope(self, eps: float, alpha: float = 0.0) -> tuple[float, float]:
        if eps > 0.0:
            return 0.0, 0.0
        if eps == 0.0:
            return 0.0, self.E0
        ratio = self._strength_ratio(alpha)
        fpc_eff = self.fpc * ratio       # signed (negative), reduced in magnitude
        x = eps / self.eps_c0
        r = self.r
        denom = r - 1.0 + x ** r
        sigma = fpc_eff * x * r / denom
        df_dx = r * (r - 1.0) * (1.0 - x ** r) / (denom * denom)
        Et = fpc_eff * df_dx / self.eps_c0
        return sigma, Et

    def _plastic_offset(self, eps_min: float) -> float:
        if eps_min >= 0.0:
            return 0.0
        r_ratio = eps_min / self.eps_c0
        if r_ratio > 2.0:
            r_ratio = 2.0
        return self.eps_c0 * (0.145 * r_ratio * r_ratio + 0.13 * r_ratio)

    def get_response(self, eps: float) -> tuple[float, float]:
        eps = float(eps)
        eps_min = self.eps_min_committed
        alpha = self.alpha_committed
        # Default trial = committed
        self.alpha_trial = alpha

        if eps_min == 0.0:
            if eps < 0.0:
                # First excursion -- accumulate damage past eps_c0.
                if eps < self.eps_c0:
                    self.alpha_trial = alpha + (self.eps_c0 - eps)
                sigma, Et = self._envelope(eps, alpha=self.alpha_trial)
                self.eps_min_trial = eps
            else:
                sigma, Et = 0.0, 0.0
                self.eps_min_trial = 0.0
            self.sigma_trial = sigma
            self.Et_trial = Et
            return sigma, Et

        sigma_min, _ = self._envelope(eps_min, alpha=alpha)
        eps_p = self._plastic_offset(eps_min)

        if eps <= eps_min:
            # Pushing the envelope further; damage accumulates by the
            # extra excursion beyond eps_min (and only past eps_c0,
            # since pre-peak loading is largely elastic).
            extra = eps_min - eps
            if eps < self.eps_c0:
                self.alpha_trial = alpha + extra
            sigma, Et = self._envelope(eps, alpha=self.alpha_trial)
            self.eps_min_trial = eps
        elif eps >= eps_p:
            sigma, Et = 0.0, 0.0
            self.eps_min_trial = eps_min
        else:
            E_unload = sigma_min / (eps_min - eps_p)
            sigma = E_unload * (eps - eps_p)
            Et = E_unload
            self.eps_min_trial = eps_min

        self.sigma_trial = sigma
        self.Et_trial = Et
        return sigma, Et

    def commit_state(self) -> None:
        self.eps_min_committed = self.eps_min_trial
        self.alpha_committed = self.alpha_trial

    def revert_state(self) -> None:
        self.eps_min_trial = self.eps_min_committed
        self.alpha_trial = self.alpha_committed

    def __repr__(self) -> str:
        return (
            f"ConcreteMander(fpc={-self.fpc:g}, eps_c0={-self.eps_c0:g}, "
            f"Ec={self.E0:g}, damage_factor={self.damage_factor:g})"
        )
