# Bridges

Bridge-engineering primitives, organised around the four sub-modules
of {mod}`femsolver.bridges`.

## Influence lines + moving loads

```python
import numpy as np
from femsolver.bridges import (
    MovingLoad,
    influence_line_simple_span_moment,
    max_response_for_moving_load,
    max_truck_envelope_simple_span,
)

# IL at midspan of a 30 m SS beam
L = 30.0
xi = np.linspace(0, L, 31)
eta = influence_line_simple_span_moment(L=L, x=L/2, xi=xi)

# Maximum moment under HL-93 truck
truck = MovingLoad.preset("hl93_truck")
M_max, head_pos = max_response_for_moving_load(
    moving_load=truck,
    influence_line=lambda x: influence_line_simple_span_moment(L=L, x=L/2, xi=x),
    L=L,
)

# Or use the HL-93 envelope helper
env = max_truck_envelope_simple_span(L=L, x=L/2, impact_factor=1.33)
print(env["M_with_impact"], "N.m")
```

Presets: `"hl93_truck"`, `"hl93_tandem"`, `"irc_class_a"`,
`"irc_70r"`.

## Post-tensioning losses

```python
from femsolver.bridges import (
    parabolic_drape_profile, friction_loss,
    anchorage_slip_loss, equivalent_uniform_load_parabolic,
)

profile = parabolic_drape_profile(L=30.0, drape=0.6, n_segments=40)
fric = friction_loss(profile, mu=0.20, k=0.0066)
slip = anchorage_slip_loss(
    profile, P_0=2.0e6, mu=0.20, k=0.0066,
    slip=0.006, A_ps=1.0e-3,
)
print(slip.P0_after_seating)

w_eq = equivalent_uniform_load_parabolic(P=1.5e6, drape=0.6, L=30.0)
```

## Creep / shrinkage / relaxation

```python
from femsolver.bridges import (
    cebfip_creep_coefficient, cebfip_shrinkage,
    steel_relaxation_loss_ratio, prestress_long_term_loss,
)

creep = cebfip_creep_coefficient(
    t_days=18250, t0_days=28,
    f_cm=38e6, RH=70.0, h_0=0.20,
)
shr = cebfip_shrinkage(
    t_days=18250, t_s_days=3,
    f_cm=38e6, RH=70.0, h_0=0.20,
)
rel = steel_relaxation_loss_ratio(t_hours=18250*24, fpi_over_fpy=0.75)

loss = prestress_long_term_loss(
    P_initial=2000e3, A_ps=1.0e-3, E_p=1.95e11,
    sigma_c_at_strand=-10e6, E_c=34e9,
    creep=creep, shrinkage=shr,
    relaxation_loss_ratio=rel, f_pi=1395e6,
)
print(loss.delta_P_total, loss.P_effective)
```

## Composite section

```python
from femsolver.bridges import (
    composite_girder_deck, composite_fiber_stresses,
)

props = composite_girder_deck(
    girder_area=0.45, girder_I=0.04,
    girder_y_centroid=0.50, girder_height=1.20,
    deck_width=2.40, deck_thickness=0.20,
    E_girder=34e9, E_deck=28e9,
)
stress = composite_fiber_stresses(
    props=props, P=-1.5e6, M=800e3,
    strand_y_from_bottom=0.10,
)
```

## Capstone

See `examples/49_psc_girder_bridge.py` for an end-to-end PSC girder
under HL-93 with 50-year time-dependent losses.
