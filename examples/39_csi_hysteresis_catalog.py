"""Phase 24 -- CSI-style fiber-hinge hysteresis catalog.

Side-by-side comparison of all the uniaxial hysteresis types mirroring
CSI's PERFORM-3D / SAP2000 fiber-hinge catalog:

* **Elastic**           -- :class:`UniaxialElastic`
* **Kinematic**         -- :class:`UniaxialBilinear`
* **Isotropic**         -- :class:`UniaxialIsotropicHardening`  (Phase 24.1)
* **Takeda**            -- :class:`UniaxialTakeda`              (Phase 24.2)
* **Pivot**             -- :class:`UniaxialPivot`               (Phase 24.3)
* **Concrete (KP)**     -- :class:`ConcreteKentPark`
* **Hysteretic (deg.)** -- :class:`UniaxialHysteretic`
* **IMK**               -- :class:`UniaxialIMK`                 (Phase 24.4)
* **BRB**               -- :class:`UniaxialBRB`                 (Phase 24.5)

Each material is driven by the same reversed-cyclic strain history;
the printed table reports peak stresses per cycle and the total
dissipated energy (loop area, computed by trapezoidal integration).
This makes the modelling-choice impact immediately visible.

Run::

    python examples/39_csi_hysteresis_catalog.py
"""
from __future__ import annotations

import numpy as np

from femsolver import (
    ConcreteKentPark,
    UniaxialBilinear,
    UniaxialBRB,
    UniaxialElastic,
    UniaxialHysteretic,
    UniaxialIMK,
    UniaxialIsotropicHardening,
    UniaxialPivot,
    UniaxialTakeda,
)


# ============================================================ helpers

def cycle_material(mat, strain_history) -> tuple[np.ndarray, np.ndarray]:
    """Drive ``mat`` through a strain history, returning (eps, sigma)
    arrays for each loading step. The material is committed after
    every step (treating each strain as a converged increment)."""
    sigmas = []
    for eps in strain_history:
        sigma, _ = mat.get_response(float(eps))
        mat.commit_state()
        sigmas.append(sigma)
    return np.asarray(strain_history), np.asarray(sigmas)


def hysteresis_energy(eps: np.ndarray, sigma: np.ndarray) -> float:
    """Closed-loop dissipated energy by trapezoidal area in (ε, σ)
    space: ∮ σ dε."""
    # Trapezoidal sum of sigma*deps over the closed path.
    # For a closed loop this gives the enclosed area (with sign
    # convention dictated by traversal direction).
    return float(abs(np.trapezoid(sigma, eps)))


def build_strain_history(*, n_cycles: int = 3, eps_max: float = 0.005,
                          steps_per_quarter: int = 25) -> np.ndarray:
    """Build a reversed-cyclic triangular strain history.

    Each cycle: 0 -> +eps_max -> 0 -> -eps_max -> 0
    """
    hist = []
    for _ in range(n_cycles):
        hist.extend(np.linspace(0.0, eps_max, steps_per_quarter, endpoint=False))
        hist.extend(np.linspace(eps_max, 0.0, steps_per_quarter, endpoint=False))
        hist.extend(np.linspace(0.0, -eps_max, steps_per_quarter, endpoint=False))
        hist.extend(np.linspace(-eps_max, 0.0, steps_per_quarter, endpoint=False))
    # Close the path back at 0 so the integral closes cleanly
    hist.append(0.0)
    return np.asarray(hist)


# ============================================================ main

def main() -> None:
    print("CSI-style hysteresis catalog -- reversed cyclic loading")
    print("=" * 65)

    # Common material parameters (steel-like for kinematic / iso / BRB,
    # and adjusted as appropriate for concrete / RC).
    E = 2.0e11
    sigma_y = 4.0e8

    history = build_strain_history(n_cycles=3, eps_max=0.005)
    print(f"  Strain history: 3 cycles, peak |eps| = 0.005, "
          f"{len(history)} steps total")
    print(f"  Common parameters: E = {E:.1e} Pa, sigma_y = "
          f"{sigma_y/1e6:.0f} MPa")
    print()

    catalog = [
        ("Elastic",     UniaxialElastic(E=E)),
        ("Kinematic",   UniaxialBilinear(E=E, sigma_y=sigma_y, b=0.03)),
        ("Isotropic",   UniaxialIsotropicHardening(E=E, sigma_y=sigma_y,
                                                      b=0.03)),
        ("Takeda",      UniaxialTakeda(E=E, sigma_y=sigma_y, b=0.03,
                                         alpha=0.5)),
        ("Pivot",       UniaxialPivot(E=E, sigma_y=sigma_y, b=0.03,
                                        alpha=5.0)),
        ("Concrete-KP", ConcreteKentPark(fpc=30.0e6, eps_c0=0.002,
                                            fpcu=8.0e6, eps_cu=0.006)),
        ("Hysteretic",  UniaxialHysteretic(E=E, sigma_y=sigma_y, b=0.03,
                                             pinch_x=0.5, pinch_y=0.5,
                                             damage_factor=0.0)),
        ("IMK",         UniaxialIMK(E=E, sigma_y=sigma_y, b=0.03,
                                      eps_cap=0.016, alpha_pc=-0.1)),
        ("BRB",         UniaxialBRB(E=E, sigma_y=sigma_y, b=0.02,
                                      beta=1.10, a_iso=30.0)),
    ]

    print(f"  {'Material':<14} | {'Peak +sigma (MPa)':>17} | "
          f"{'Peak -sigma (MPa)':>17} | {'Loop area (J/m^3)':>17}")
    print("  " + "-" * 70)
    for name, mat in catalog:
        eps_arr, sig_arr = cycle_material(mat, history)
        peak_pos = float(np.max(sig_arr))
        peak_neg = float(np.min(sig_arr))
        energy = hysteresis_energy(eps_arr, sig_arr)
        print(f"  {name:<14} | {peak_pos/1e6:>17.2f} | "
              f"{peak_neg/1e6:>17.2f} | {energy:>17.3e}")
    print()
    print("Reading the result:")
    print("* Elastic: loop area = 0 (no dissipation), perfectly linear.")
    print("* Kinematic / Isotropic: similar peak stresses on monotonic;")
    print("  they diverge on REVERSAL -- isotropic over-predicts because")
    print("  the yield surface expanded (no Bauschinger).")
    print("* Takeda: similar peak strength to bilinear, but smaller loop")
    print("  area thanks to the stiffness-degrading unloading.")
    print("* Pivot: characteristic narrower loops (pinched toward the")
    print("  pivot points); less energy per cycle than fat kinematic loops.")
    print("* Concrete-KP: very asymmetric -- strong in compression, weak")
    print("  in tension, with crack-closure behaviour.")
    print("* Hysteretic w/ pinching: pinched loops similar to Pivot.")
    print("* IMK: post-cap negative-slope branch (if eps_max past cap)")
    print("  produces strength drop -- the FEMA P695 collapse signature.")
    print("* BRB: asymmetric peaks (compression > tension by beta = 1.10)")
    print("  + cyclic strength growth from isotropic hardening.")
    print()
    print("Choosing a material:")
    print("* Steel rebar in beam fibers           -> MenegottoPinto / Kinematic")
    print("* Confined concrete fibers              -> ConcreteMander")
    print("* RC beam-column plastic hinge          -> Takeda or Hysteretic")
    print("* RC column with pinched cyclic loops  -> Pivot")
    print("* Collapse-capacity analysis           -> IMK")
    print("* BRB diagonal core                    -> BRB")


if __name__ == "__main__":
    main()
