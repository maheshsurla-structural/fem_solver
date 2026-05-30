# Connection models

Connection-level mechanics for steel moment-frame and post-yield design.

## Krawinkler panel zone

Bilinear shear backbone of the column-beam joint panel:

```python
from femsolver.design.connections import (
    krawinkler_panel_zone, build_panel_zone_material,
)

pz = krawinkler_panel_zone(
    f_y=345e6,
    d_c=0.355, t_p=0.0112,
    d_b=0.612,
    b_cf=0.368, t_cf=0.018,
)
print(pz.V_y, pz.K_e_rot, pz.M_y_joint)

# Drop straight into a ZeroLengthElement at the joint:
mat = build_panel_zone_material(pz)
```

## Reduced Beam Section (AISC 358)

```python
from femsolver.design.connections import aisc358_recommended_RBS

rbs = aisc358_recommended_RBS(
    d=0.612, b_f=0.229, t_f=0.0236,
    f_y=345e6, Z_x=3.99e-3, L_clear=6.0,
)
print(rbs.Z_RBS / 3.99e-3)         # ~ 0.68 (typical reduction)
print(rbs.M_pr_face)               # used for capacity-design of column / panel
```

## PR (semi-rigid) connections — Richard-Abbott

```python
from femsolver.design.connections import (
    Pr_preset, RichardAbbottParams,
)

# Built-in presets
ra = Pr_preset("end_plate_extended")     # extended end plate
M_at_1pct = ra.M(0.01)

# Or roll your own
custom = RichardAbbottParams(
    R_ki=8.0e7, R_kp=2.0e6, M_0=250.0e3, n=1.8,
)
```

Available presets: `"top_seat_double_web"`, `"end_plate_4_bolts"`,
`"end_plate_extended"`, `"tee_stub"`.

## Bolts + welds (AISC + IS 800)

```python
from femsolver.design.connections import (
    bolt_shear_aisc, bolt_shear_is800,
    bolt_bearing_aisc, bolt_bearing_is800,
    block_shear_aisc,
    fillet_weld_aisc, fillet_weld_is800,
)

shear = bolt_shear_aisc(n_bolts=4, A_b=3.80e-4, F_nv=0.563*825e6)
weld = fillet_weld_aisc(leg_size=0.008, F_EXX=480e6)
```

## Capstone

See `examples/48_connection_capstone.py` for a side-by-side
panel-zone + RBS + PR + bolts/welds walkthrough on a real
W14×90 / W24×84 joint.
