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
        # Guard the deep softening tail: for an absurd compressive strain
        # (only reached by a diverging element-state-determination probe, never
        # a physical step) ``x ** r`` overflows. There the Popovics stress
        # ``fpc*x*r/(r-1+x^r)`` tends to ``fpc*r*x^(1-r) -> 0``; return that
        # finite value (Et -> 0) so the solver fails gracefully instead of
        # crashing with OverflowError. The threshold is far outside any real
        # response, so normal results are unchanged.
        if x > 1.0 and r * math.log10(x) > 250.0:
            return fpc_eff * r * x ** (1.0 - r), 0.0
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


# ==================================================== parabola-rectangle (EC2)

class ConcreteParabolaRectangle(UniaxialMaterial):
    """Parabola-rectangle concrete design curve (Eurocode 2 EN 1992-1-1
    §3.1.7, also IRC 112 / IRS): a parabolic ascent to the peak at
    ``eps_c2`` followed by a constant plateau to the ultimate strain
    ``eps_cu2``, then zero (crushed). Tension is zero.

        0 >= eps >= eps_c2 :  sigma = fpc * [1 - (1 - eps/eps_c2)^n]   (parabola)
        eps_c2 > eps >= eps_cu2 :  sigma = fpc                          (plateau)
        eps < eps_cu2 :  sigma = 0                                      (crushed)

    Off-envelope unloading reuses the Karsan-Jirsa plastic-offset line
    (shared with :class:`ConcreteKentPark`), so the model is well-behaved
    if probed cyclically, though its intended use is monotonic section
    analysis (moment-curvature, ULS capacity).

    Parameters
    ----------
    fpc : float
        Peak compressive strength (positive magnitude); the design value
        ``f_cd`` or the characteristic ``f_ck`` depending on the caller.
    eps_c2 : float, default 0.002
        Strain at the peak (positive magnitude). EC2: 0.002 for f_ck<=50.
    eps_cu2 : float, default 0.0035
        Ultimate (crushing) strain (positive magnitude). EC2: 0.0035 for
        f_ck<=50. Must satisfy ``eps_cu2 > eps_c2``.
    n : float, default 2.0
        Parabola exponent. EC2: 2.0 for f_ck<=50 (lower for high strength).

    Notes
    -----
    The initial tangent implied by the parabola is ``E0 = n fpc / eps_c2``
    (= ``2 fpc / eps_c2`` for the usual ``n = 2``).
    """

    def __init__(self, fpc: float, eps_c2: float = 0.002,
                 eps_cu2: float = 0.0035, n: float = 2.0):
        if fpc <= 0.0:
            raise ValueError(f"fpc must be positive, got {fpc}")
        if eps_c2 <= 0.0:
            raise ValueError(f"eps_c2 must be positive, got {eps_c2}")
        if eps_cu2 <= eps_c2:
            raise ValueError(
                f"eps_cu2 must be strictly greater than eps_c2, got "
                f"eps_cu2={eps_cu2}, eps_c2={eps_c2}"
            )
        if n <= 0.0:
            raise ValueError(f"n must be positive, got {n}")
        # Signed storage (compression negative).
        self.fpc = -float(fpc)
        self.eps_c2 = -float(eps_c2)
        self.eps_cu2 = -float(eps_cu2)
        self.n = float(n)
        self.E0 = self.n * fpc / eps_c2
        # ----- state (most-compressive strain ever committed) -----
        self.eps_min_committed: float = 0.0
        self.eps_min_trial: float = 0.0
        self.sigma_trial: float = 0.0
        self.Et_trial: float = self.E0

    # ------------------------------------------------------ envelope
    def _envelope(self, eps: float) -> tuple[float, float]:
        if eps > 0.0:
            return 0.0, 0.0
        if eps >= self.eps_c2:                       # parabolic ascent
            one_minus = 1.0 - eps / self.eps_c2      # in [0, 1]
            sigma = self.fpc * (1.0 - one_minus ** self.n)
            Et = self.fpc * self.n * one_minus ** (self.n - 1.0) / self.eps_c2
            return sigma, Et
        if eps >= self.eps_cu2:                      # constant plateau
            return self.fpc, 0.0
        return 0.0, 0.0                              # crushed past eps_cu2

    # ------------------------------------------------------ plastic offset
    def _plastic_offset(self, eps_min: float) -> float:
        if eps_min >= 0.0:
            return 0.0
        r = eps_min / self.eps_c2
        if r > 2.0:
            r = 2.0
        return self.eps_c2 * (0.145 * r * r + 0.13 * r)

    # ------------------------------------------------------ get_response
    def get_response(self, eps: float) -> tuple[float, float]:
        eps = float(eps)
        eps_min = self.eps_min_committed

        if eps_min == 0.0:
            if eps > 0.0:
                sigma, Et = 0.0, 0.0
                self.eps_min_trial = 0.0
            else:
                sigma, Et = self._envelope(eps)
                self.eps_min_trial = eps
            self.sigma_trial = sigma
            self.Et_trial = Et
            return sigma, Et

        sigma_min, _ = self._envelope(eps_min)
        eps_p = self._plastic_offset(eps_min)

        if eps <= eps_min:
            sigma, Et = self._envelope(eps)
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

    # ------------------------------------------------------ state
    def commit_state(self) -> None:
        self.eps_min_committed = self.eps_min_trial

    def revert_state(self) -> None:
        self.eps_min_trial = self.eps_min_committed

    def __repr__(self) -> str:
        return (
            f"ConcreteParabolaRectangle(fpc={-self.fpc:g}, "
            f"eps_c2={-self.eps_c2:g}, eps_cu2={-self.eps_cu2:g}, n={self.n:g})"
        )


# ============================================================ trilinear

class ConcreteTrilinear(UniaxialMaterial):
    """Trilinear concrete compression model: three straight segments -- an
    initial elastic rise to the first knee ``(eps1, sigma1)``, a second rise
    to the peak ``(eps_c0, fpc)``, and a descending branch to the residual
    ``(eps_cu, fpcu)``, then a constant residual plateau. Tension is zero.

    The piecewise-linear counterpart of :class:`ConcreteKentPark` (which
    uses a parabolic ascent). The first knee is set by ``f1_ratio`` = the
    elastic-limit stress as a fraction of ``fpc`` (EC2 uses ~0.4); with the
    initial tangent ``E0 = 2 fpc / eps_c0`` this puts the knee at
    ``eps1 = f1_ratio * eps_c0 / 2``. Off-envelope unloading reuses the
    Karsan-Jirsa plastic-offset line (shared with Kent-Park).

    Parameters
    ----------
    fpc, eps_c0, fpcu, eps_cu : float
        Peak strength, peak strain, residual crushing stress, and crushing
        strain (positive magnitudes; ``0 <= fpcu <= fpc``, ``eps_cu >
        eps_c0``) -- as in :class:`ConcreteKentPark`.
    f1_ratio : float, default 0.4
        First-knee (elastic-limit) stress as a fraction of ``fpc`` (0..1).
    """

    def __init__(self, fpc: float, eps_c0: float, fpcu: float,
                 eps_cu: float, f1_ratio: float = 0.4):
        if fpc <= 0.0:
            raise ValueError(f"fpc must be positive, got {fpc}")
        if eps_c0 <= 0.0:
            raise ValueError(f"eps_c0 must be positive, got {eps_c0}")
        if not (0.0 <= fpcu <= fpc):
            raise ValueError(f"fpcu must satisfy 0 <= fpcu <= fpc, got "
                             f"fpcu={fpcu}, fpc={fpc}")
        if eps_cu <= eps_c0:
            raise ValueError(f"eps_cu must exceed eps_c0, got eps_cu={eps_cu}, "
                             f"eps_c0={eps_c0}")
        if not (0.0 < f1_ratio < 1.0):
            raise ValueError(f"f1_ratio must be in (0, 1), got {f1_ratio}")
        self.E0 = 2.0 * fpc / eps_c0
        # Signed storage (compression negative).
        self.fpc = -float(fpc)
        self.eps_c0 = -float(eps_c0)
        self.fpcu = -float(fpcu)
        self.eps_cu = -float(eps_cu)
        self.sigma1 = -float(f1_ratio * fpc)
        self.eps1 = self.sigma1 / self.E0             # negative
        # ----- state -----
        self.eps_min_committed: float = 0.0
        self.eps_min_trial: float = 0.0
        self.sigma_trial: float = 0.0
        self.Et_trial: float = self.E0

    def _envelope(self, eps: float) -> tuple[float, float]:
        if eps > 0.0:
            return 0.0, 0.0
        if eps >= self.eps1:                          # initial elastic
            return self.E0 * eps, self.E0
        if eps >= self.eps_c0:                        # rise to peak
            k2 = (self.fpc - self.sigma1) / (self.eps_c0 - self.eps1)
            return self.sigma1 + k2 * (eps - self.eps1), k2
        if eps >= self.eps_cu:                        # descending
            k3 = (self.fpcu - self.fpc) / (self.eps_cu - self.eps_c0)
            return self.fpc + k3 * (eps - self.eps_c0), k3
        return self.fpcu, 0.0                         # residual plateau

    def _plastic_offset(self, eps_min: float) -> float:
        if eps_min >= 0.0:
            return 0.0
        r = eps_min / self.eps_c0
        if r > 2.0:
            r = 2.0
        return self.eps_c0 * (0.145 * r * r + 0.13 * r)

    def get_response(self, eps: float) -> tuple[float, float]:
        eps = float(eps)
        eps_min = self.eps_min_committed
        if eps_min == 0.0:
            if eps > 0.0:
                sigma, Et = 0.0, 0.0
                self.eps_min_trial = 0.0
            else:
                sigma, Et = self._envelope(eps)
                self.eps_min_trial = eps
            self.sigma_trial, self.Et_trial = sigma, Et
            return sigma, Et
        sigma_min, _ = self._envelope(eps_min)
        eps_p = self._plastic_offset(eps_min)
        if eps <= eps_min:
            sigma, Et = self._envelope(eps)
            self.eps_min_trial = eps
        elif eps >= eps_p:
            sigma, Et = 0.0, 0.0
            self.eps_min_trial = eps_min
        else:
            E_unload = sigma_min / (eps_min - eps_p)
            sigma = E_unload * (eps - eps_p)
            Et = E_unload
            self.eps_min_trial = eps_min
        self.sigma_trial, self.Et_trial = sigma, Et
        return sigma, Et

    def commit_state(self) -> None:
        self.eps_min_committed = self.eps_min_trial

    def revert_state(self) -> None:
        self.eps_min_trial = self.eps_min_committed

    def __repr__(self) -> str:
        return (
            f"ConcreteTrilinear(fpc={-self.fpc:g}, eps_c0={-self.eps_c0:g}, "
            f"fpcu={-self.fpcu:g}, eps_cu={-self.eps_cu:g}, "
            f"sigma1={-self.sigma1:g})"
        )


# ============================================================ tension stiffening

class ConcreteTensionStiffening(UniaxialMaterial):
    """Add a tension branch to a compression-only concrete model.

    Compression is delegated unchanged to the wrapped ``compression``
    model (Kent-Park, Mander, ...). Tension follows:

    * ``0 < eps <= eps_cr``: linear elastic, ``sigma = E_ct * eps``
      (uncracked concrete), where ``eps_cr = f_ct / E_ct``.
    * ``eps > eps_cr``: exponentially-decaying **tension stiffening**,
      ``sigma = f_ct * exp(-(eps - eps_cr) / eps_decay)`` -- the bond-
      carried tension the concrete keeps between cracks, fading as the
      crack opens. Continuous at ``eps_cr`` (both give ``f_ct``).

    This restores the pre-cracking (uncracked) flexural stiffness and a
    smooth cracking transition in fiber moment-curvature, so the cracking
    point lies on the computed curve -- matching Midas GSD / MCFT-style
    RC section analysis. The decay makes the tension negligible well
    before yield, so yield / ultimate are essentially unchanged.

    Parameters
    ----------
    compression : UniaxialMaterial
        Compression-only concrete model (its state/commit is delegated).
    f_ct : float
        Tensile (cracking) strength, positive magnitude -- e.g. the
        modulus of rupture ``0.62 sqrt(fc' [MPa]) MPa``.
    E_ct : float
        Tension elastic modulus (typically the concrete ``E_c``).
    eps_decay : float, default 1e-3
        Post-crack decay strain of the tension-stiffening branch. Larger
        keeps concrete tension active further into the cracked regime.
    """

    def __init__(self, compression: UniaxialMaterial, f_ct: float,
                 E_ct: float, eps_decay: float = 1.0e-3):
        if f_ct < 0.0:
            raise ValueError(f"f_ct must be >= 0, got {f_ct}")
        if E_ct <= 0.0:
            raise ValueError(f"E_ct must be positive, got {E_ct}")
        if eps_decay <= 0.0:
            raise ValueError(f"eps_decay must be positive, got {eps_decay}")
        self.compression = compression
        self.f_ct = float(f_ct)
        self.E_ct = float(E_ct)
        self.eps_cr = (self.f_ct / self.E_ct) if self.f_ct > 0.0 else 0.0
        self.eps_decay = float(eps_decay)
        # Expose an initial modulus for callers that probe E0.
        self.E0 = float(getattr(compression, "E0", E_ct))

    def get_response(self, eps: float) -> tuple[float, float]:
        eps = float(eps)
        if eps <= 0.0:
            return self.compression.get_response(eps)
        if self.f_ct <= 0.0:
            return 0.0, 0.0
        if eps <= self.eps_cr:
            return self.E_ct * eps, self.E_ct
        sigma = self.f_ct * math.exp(-(eps - self.eps_cr) / self.eps_decay)
        Et = -sigma / self.eps_decay
        return sigma, Et

    def commit_state(self) -> None:
        self.compression.commit_state()

    def revert_state(self) -> None:
        self.compression.revert_state()

    def __repr__(self) -> str:
        return (
            f"ConcreteTensionStiffening(f_ct={self.f_ct:g}, "
            f"E_ct={self.E_ct:g}, eps_decay={self.eps_decay:g})"
        )
