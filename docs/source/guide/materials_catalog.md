# Materials catalog

A walk-through of the constitutive library.

## Continuum (2D / 3D)

| Class | Behaviour |
|-------|-----------|
| {class}`~femsolver.ElasticIsotropic` | Linear elastic, isotropic |
| {class}`~femsolver.J2Plasticity3D` | von Mises with isotropic / kinematic hardening |
| {class}`~femsolver.DruckerPrager3D` | Drucker-Prager for soils + dilatancy |
| {class}`~femsolver.OrthotropicLamina` | Orthotropic ply for composites |

## Uniaxial library

Used by fiber sections (`FiberSection2D` / `FiberSection3D`) and
zero-length connectors.

### Steel-rebar style

```{eval-rst}
.. autoclass:: femsolver.UniaxialBilinear
   :no-index:
.. autoclass:: femsolver.UniaxialMenegottoPinto
   :no-index:
.. autoclass:: femsolver.UniaxialIsotropicHardening
   :no-index:
```

### Concrete

```{eval-rst}
.. autoclass:: femsolver.ConcreteKentPark
   :no-index:
.. autoclass:: femsolver.ConcreteMander
   :no-index:
```

### Hysteretic / pinching

```{eval-rst}
.. autoclass:: femsolver.UniaxialHysteretic
   :no-index:
.. autoclass:: femsolver.UniaxialTakeda
   :no-index:
.. autoclass:: femsolver.UniaxialPivot
   :no-index:
```

### Collapse / degradation

```{eval-rst}
.. autoclass:: femsolver.UniaxialIMK
   :no-index:
.. autoclass:: femsolver.UniaxialBRB
   :no-index:
```

### Misc

```{eval-rst}
.. autoclass:: femsolver.UniaxialElastic
   :no-index:
.. autoclass:: femsolver.UniaxialGap
   :no-index:
```

## Picking a material

| Situation | Recommended |
|-----------|-------------|
| Steel rebar in beam fibers | `UniaxialMenegottoPinto` or `UniaxialBilinear` |
| Confined concrete fibers | `ConcreteMander` |
| Unconfined cover concrete | `ConcreteKentPark` |
| RC beam-column plastic hinge | `UniaxialTakeda` or `UniaxialHysteretic` |
| RC column with pinched cycles | `UniaxialPivot` |
| Collapse-capacity analysis | `UniaxialIMK` |
| BRB diagonal core | `UniaxialBRB` |
| Compression-only / gap | `UniaxialGap` |

See `examples/39_csi_hysteresis_catalog.py` and
`examples/44_fiber_section_csi_hysteresis.py` for σ-ε and M-κ loop
comparisons of all CSI-style models.
