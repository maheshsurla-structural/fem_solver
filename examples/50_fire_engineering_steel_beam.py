"""Phase 40.6 -- Steel beam under ISO 834 fire (residual-capacity check).

A simply-supported steel I-beam carries a gravity load that produces
a moment utilisation ``mu_0 ~ 0.5`` at the fire limit state. The
beam is exposed to the ISO 834 standard fire curve through its
soffit; the steel temperature rises following the EC3 simple
"unprotected steelwork" lumped-capacitance heat-balance:

    Delta T_a = (A_m / V) / (rho_a · c_a) · h_net · Delta t,

where ``h_net = h_conv (T_g - T_a) + epsilon · sigma_b (T_g^4 - T_a^4)``
is the net flux per unit surface from the fire to the steel surface,
``A_m / V`` is the section factor (typically 100-300 1/m for
unprotected sections), and ``rho_a c_a`` is the volumetric heat
capacity of steel.

The reported quantities are:

1. Gas temperature ``T_g(t)``  (ISO 834).
2. Steel temperature ``T_a(t)`` (lumped-capacitance integration).
3. Yield-strength reduction ``k_y(T_a)`` per EN 1993-1-2 Table 3.1.
4. Time to reach the critical temperature ``T_cr`` for the given
   utilisation -- the **fire resistance** of the unprotected beam.

Run::

    python examples/50_fire_engineering_steel_beam.py
"""
from __future__ import annotations

import math

import numpy as np

from femsolver import (
    iso_834_temperature,
    steel_critical_temperature,
    steel_modulus_reduction_ec3,
    steel_strength_reduction_ec3,
)


# ============================================================ inputs

# Steel I-beam properties (typical W360x122 / IPE 400 class)
A_m_over_V = 200.0           # 1/m  -- section factor for unprotected
                              #         steel exposed on three sides
RHO_STEEL = 7850.0           # kg/m^3
C_STEEL = 600.0              # J/(kg.K) -- effective value at high T

# Boundary conditions at the steel surface
h_conv = 25.0                 # W/(m^2.K) -- convection (EC1-1-2)
epsilon = 0.7                 # emissivity (EC1-1-2 Annex B)
SIGMA_SB = 5.67e-8            # W/(m^2.K^4)

# Time discretisation
DT = 5.0                       # s
T_END = 60.0 * 60.0            # 60 minutes

# Demand
MU_0 = 0.50                    # utilisation ratio at fire limit state


# ============================================================ EC3 lumped-capacitance

def steel_temperature_history(
    *,
    A_m_over_V: float,
    h_conv: float, epsilon: float,
    rho: float, c: float,
    dt: float, t_end: float,
    T0: float = 20.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate the EC3-1-2 §4.2.5.1 lumped-capacitance heat balance
    for an unprotected steel section under the ISO 834 fire.

    Returns ``(t, T_g, T_a)`` arrays in seconds and °C.
    """
    n_steps = int(round(t_end / dt))
    t = np.linspace(0.0, n_steps * dt, n_steps + 1)
    T_g = np.empty_like(t)
    T_a = np.empty_like(t)
    T_a[0] = T0
    for i, ti in enumerate(t):
        T_g[i] = iso_834_temperature(ti / 60.0)
    for i in range(n_steps):
        # Use trapezoidal-like average gas temperature over the step
        T_g_mean = 0.5 * (T_g[i] + T_g[i + 1])
        # Net heat flux per unit surface area (W/m^2)
        # Convective + radiative
        T_g_K = T_g_mean + 273.15
        T_a_K = T_a[i] + 273.15
        h_net = h_conv * (T_g_mean - T_a[i]) + \
                epsilon * SIGMA_SB * (T_g_K ** 4 - T_a_K ** 4)
        dT_a = (A_m_over_V / (rho * c)) * h_net * dt
        T_a[i + 1] = T_a[i] + dT_a
    return t, T_g, T_a


# ============================================================ main

def main() -> None:
    print("=" * 78)
    print("Phase 40.6 -- Steel beam under ISO 834 fire")
    print("=" * 78)

    print(f"\nSection factor       A_m/V    = {A_m_over_V} 1/m")
    print(f"Convection coef      h_conv   = {h_conv} W/(m^2.K)")
    print(f"Emissivity           epsilon  = {epsilon}")
    print(f"Steel density        rho      = {RHO_STEEL} kg/m^3")
    print(f"Steel specific heat  c        = {C_STEEL} J/(kg.K)")
    print(f"Utilisation ratio    mu_0     = {MU_0}")

    # Critical steel temperature
    T_cr = steel_critical_temperature(mu_0=MU_0)
    print(f"\nCritical steel temperature  T_a,cr = {T_cr:.0f} C  (EC3 § 4.2.4)")

    # Integrate the EC3 heat balance
    t, T_g, T_a = steel_temperature_history(
        A_m_over_V=A_m_over_V,
        h_conv=h_conv, epsilon=epsilon,
        rho=RHO_STEEL, c=C_STEEL,
        dt=DT, t_end=T_END,
    )

    print(f"\nUnprotected-steel temperature history (every 5 min):")
    print(f"  {'t (min)':>10}{'T_g (C)':>12}{'T_a (C)':>12}"
          f"{'k_y':>10}{'k_E':>10}")
    print("  " + "-" * 54)
    for i in range(0, len(t), int(round(5 * 60 / DT))):
        ky = steel_strength_reduction_ec3(T_a[i])
        kE = steel_modulus_reduction_ec3(T_a[i])
        print(f"  {t[i]/60:>10.0f}{T_g[i]:>12.0f}{T_a[i]:>12.0f}"
              f"{ky:>10.3f}{kE:>10.3f}")

    # Find fire-resistance time -- first t at which T_a >= T_cr
    crossed = np.where(T_a >= T_cr)[0]
    if crossed.size > 0:
        t_fire_resistance = t[crossed[0]] / 60.0
        print(f"\nFire-resistance time (unprotected): "
              f"{t_fire_resistance:.1f} min")
        print(f"  (steel reaches T_cr = {T_cr:.0f} C, "
              f"capacity drops below demand)")
    else:
        print("\nSteel did not reach T_cr within the analysis window.")

    # Brief commentary
    print("\nInterpretation:")
    print("  An unprotected steel section with high A_m/V heats almost")
    print("  in step with the fire gas temperature. With mu_0 = 0.5 the")
    print("  beam fails around T_a = 585 C, which typically happens")
    print("  within 15-20 min under ISO 834.  Passive fire protection")
    print("  (intumescent paint, board, spray) reduces A_m/V and pushes")
    print("  the fire-resistance time to 60 / 90 / 120 min as required.")
    print()
    print("=" * 78)
    print("Theme C closed: heat conduction (steady + transient),")
    print("                thermo-mechanical coupling, fire-engineering all")
    print("                operational.")
    print("=" * 78)


if __name__ == "__main__":
    main()
