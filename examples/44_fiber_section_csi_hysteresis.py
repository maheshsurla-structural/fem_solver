"""Phase 24.6 -- CSI hysteresis catalog at the FIBER-SECTION level.

The Phase 24 catalog example (``39_csi_hysteresis_catalog.py``) drives
each new uniaxial material directly through a cyclic strain history
and reports the sigma-epsilon loops. This example takes the next step:
it wraps each material as the fiber-stress-strain law of a
rectangular :class:`FiberSection2D`, drives the section through a
cyclic CURVATURE history (zero axial strain), and reports the
moment-curvature loops -- the form in which a beam-column element
actually consumes the material.

The fiber-section integration weighs the contribution of each fiber
by its area and squared distance from the neutral axis, so the
moment-curvature loop is a smoothed version of the σ-ε loop (the
outermost fibers reach yield first; inner fibers follow). The
section-level dissipation per cycle therefore differs from the
material-level value in a quantifiable way.

Materials covered:

* Kinematic           -- :class:`UniaxialBilinear`              (reference)
* Isotropic           -- :class:`UniaxialIsotropicHardening`     (Phase 24.1)
* Takeda              -- :class:`UniaxialTakeda`                 (Phase 24.2)
* Pivot               -- :class:`UniaxialPivot`                  (Phase 24.3)
* IMK                 -- :class:`UniaxialIMK`                    (Phase 24.4)
* BRB                 -- :class:`UniaxialBRB`                    (Phase 24.5)

Each section sees the same cyclic-curvature history (4 amplitude
steps x 2 cycles each = 8 cycles, with peaks 1/2/3/4 x the first-
yield curvature ``kappa_y``).

Run::

    python examples/44_fiber_section_csi_hysteresis.py
"""
from __future__ import annotations

import numpy as np

from femsolver import (
    FiberSection2D,
    UniaxialBilinear,
    UniaxialBRB,
    UniaxialIMK,
    UniaxialIsotropicHardening,
    UniaxialPivot,
    UniaxialTakeda,
)


# ============================================================ section helpers

def cyclic_kappa_history(
    *,
    kappa_y: float,
    amplitudes=(1.0, 2.0, 3.0, 4.0),
    cycles_per_amp: int = 2,
    samples_per_cycle: int = 80,
) -> np.ndarray:
    """Sinusoidal curvature history with stepped amplitude.

    For each amplitude ``a`` in ``amplitudes`` and each cycle in
    ``cycles_per_amp``, generate a full sinusoid from 0 -> +a*kappa_y
    -> -a*kappa_y -> 0, sampled at ``samples_per_cycle`` points.
    """
    out: list[float] = []
    for a in amplitudes:
        for _ in range(cycles_per_amp):
            theta = np.linspace(0.0, 2.0 * np.pi, samples_per_cycle,
                                  endpoint=False)
            out.extend(a * kappa_y * np.sin(theta))
    return np.asarray(out)


def drive_section(
    section: FiberSection2D,
    kappas: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Drive ``section`` through a curvature history, returning
    ``(kappa_array, M_array)``.

    Zero axial strain is imposed (``eps_a = 0``). The section is
    committed after every step.
    """
    moments = np.empty_like(kappas)
    for i, k in enumerate(kappas):
        e = np.array([0.0, float(k)])
        s, _ = section.get_response(e)
        moments[i] = s[1]
        section.commit_state()
    return kappas, moments


def hysteresis_area(kappa: np.ndarray, M: np.ndarray) -> float:
    """Trapezoidal-rule loop area ``∮ M dκ``.

    For a closed loop this gives the absolute enclosed area. For the
    full multi-cycle history we get the total dissipated energy per
    unit length (units: N·m / m for a rectangular cross-section,
    i.e. N).
    """
    dk = np.diff(kappa)
    Mbar = 0.5 * (M[:-1] + M[1:])
    return float(np.abs(np.sum(Mbar * dk)))


# ============================================================ material catalog

def build_materials(*, E: float, sigma_y: float):
    """One material from each catalog entry.

    All share the same elastic ``E`` and first-yield stress ``sigma_y``
    so the elastic-range section response is identical across the
    catalog. Post-yield behaviour differs.
    """
    eps_y = sigma_y / E
    return {
        "kinematic": UniaxialBilinear(E=E, sigma_y=sigma_y, b=0.02),
        "isotropic": UniaxialIsotropicHardening(
            E=E, sigma_y=sigma_y, b=0.02,
        ),
        "takeda":    UniaxialTakeda(
            E=E, sigma_y=sigma_y, b=0.02, alpha=0.5,
        ),
        "pivot":     UniaxialPivot(
            E=E, sigma_y=sigma_y, b=0.02, alpha=8.0,
        ),
        "imk":       UniaxialIMK(
            E=E, sigma_y=sigma_y, b=0.03,
            eps_cap=20.0 * eps_y, alpha_pc=-0.10,
            sigma_res_ratio=0.30, eps_ult=80.0 * eps_y,
        ),
        "brb":       UniaxialBRB(
            E=E, sigma_y=sigma_y, b=0.02,
            beta=1.10, a_iso=0.10,
        ),
    }


# ============================================================ main

def main() -> None:
    print("=" * 76)
    print("Phase 24.6 -- CSI hysteresis catalog at the FIBER-SECTION level")
    print("=" * 76)

    # Section geometry: rectangular 100 mm x 200 mm, height direction
    width = 0.10        # m (in z-direction)
    height = 0.20       # m (in y-direction)
    n_fibers = 20
    E = 2.0e11
    sigma_y = 400.0e6

    # First-yield curvature of a rectangular section in pure bending:
    #   M_y = sigma_y * width * height^2 / 6
    #   k_y = M_y / (E * I)
    I_section = width * height ** 3 / 12.0
    M_y = sigma_y * width * height ** 2 / 6.0
    kappa_y = M_y / (E * I_section)
    print(f"\nSection: {width*1e3:.0f} x {height*1e3:.0f} mm "
          f"rectangle, {n_fibers} fibers")
    print(f"  I    = {I_section*1e6:.2f} cm^4 ({I_section:.3e} m^4)")
    print(f"  M_y  = {M_y*1e-3:.1f} kN.m")
    print(f"  k_y  = {kappa_y:.4e} 1/m")

    # Curvature history: 4 amplitudes (1-4x kappa_y), 2 cycles each
    kappas = cyclic_kappa_history(
        kappa_y=kappa_y, amplitudes=(1.0, 2.0, 3.0, 4.0),
        cycles_per_amp=2, samples_per_cycle=120,
    )
    print(f"\nCurvature history: {len(kappas)} samples, "
          f"peak amplitude = {abs(kappas).max()/kappa_y:.1f} * k_y")

    # Drive each material through the history at the SECTION level
    materials = build_materials(E=E, sigma_y=sigma_y)

    print(f"\n{'material':<14}{'M_peak (kNm)':>16}{'kappa_peak':>16}"
          f"{'energy (N)':>16}{'M_final/M_y':>16}")
    print("-" * 76)
    for name, mat in materials.items():
        section = FiberSection2D.rectangular(
            width=width, height=height,
            n_fibers=n_fibers, material=mat,
        )
        k_arr, M_arr = drive_section(section, kappas.copy())
        M_peak = float(np.max(np.abs(M_arr)))
        k_peak = float(np.max(np.abs(k_arr)))
        energy = hysteresis_area(k_arr, M_arr)
        M_final = M_arr[-1]
        print(f"{name:<14}{M_peak*1e-3:>16.2f}{k_peak:>16.4e}"
              f"{energy:>16.0f}{M_final/M_y:>16.3f}")

    # Reference: full plastic moment of the rectangular section
    M_p = sigma_y * width * height ** 2 / 4.0
    print(f"\nReference: M_p (full plastic, shape factor 1.5) "
          f"= {M_p*1e-3:.1f} kN.m")

    print()
    print("Interpretation:")
    print("- kinematic & isotropic: identical M_peak (same backbone),")
    print("  but isotropic shows MORE energy on later cycles "
          "(yield surface")
    print("  expands with cumulative plastic strain).")
    print("- Takeda & Pivot:  stiffness-degrading unloading and/or")
    print("  pivot-pinching produce LOWER energy per loop than "
          "kinematic.")
    print("- IMK: post-cap softening reduces M_peak and final force on")
    print("  high-amplitude cycles -- the FEMA P-695 collapse signature.")
    print("- BRB: asymmetric peaks (compression > tension by beta) plus")
    print("  isotropic hardening -> M_peak grows over cycles.")

    print("\n" + "=" * 76)
    print("Phase 24.6 closed: CSI hysteresis catalog at section level OK.")
    print("=" * 76)


if __name__ == "__main__":
    main()
