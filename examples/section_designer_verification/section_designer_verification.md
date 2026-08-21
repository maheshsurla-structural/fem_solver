# Section Designer verification -- femsolver vs Midas GSD

### S1 — S1 Rect column 400x600 (8-25M)

| Code | Quantity | Units | femsolver | Midas GSD | Diff % | Tol % |
|---|---|---|--:|--:|--:|--:|
| - | Gross area A_g | mm^2 | 2.4e+05 |  |  | 1.0 |
| - | I_zz (about centroidal z) | mm^4 | 7.2000e+09 |  |  | 1.0 |
| - | I_yy (about centroidal y) | mm^4 | 3.2000e+09 |  |  | 1.0 |
| - | Centroid above bottom fibre | mm | 300 |  |  | 1.0 |
| AASHTO LRFD 2024 | P_o squash (nominal) | kN | 7983 |  |  | 1.0 |
| AASHTO LRFD 2024 | P_n,max (0.80*P_o) | kN | 6387 |  |  | 1.0 |
| AASHTO LRFD 2024 | M_n at P=0 (nominal) | kN.m | 498.5 |  |  | 2.0 |
| AASHTO LRFD 2024 | phi*M_n at P=0 (design) | kN.m | 448.7 |  |  | 2.0 |
| AASHTO LRFD 2024 | Balanced P_b (eps_t=eps_y) | kN | 2510 |  |  | 2.0 |
| AASHTO LRFD 2024 | Balanced M_b (nominal) | kN.m | 804.9 |  |  | 2.0 |
| Eurocode 2 | P_o squash (nominal) | kN | 5721 |  |  | 1.0 |
| Eurocode 2 | M_n at P=0 (nominal) | kN.m | 431.8 |  |  | 2.0 |
| Eurocode 2 | phi*M_n at P=0 (design) | kN.m | 431.8 |  |  | 2.0 |
| Eurocode 2 | Balanced P_b (eps_t=eps_y) | kN | 1900 |  |  | 2.0 |
| Eurocode 2 | Balanced M_b (nominal) | kN.m | 617.1 |  |  | 2.0 |

### S2 — S2 Circular column D600 (8-25M spiral)

| Code | Quantity | Units | femsolver | Midas GSD | Diff % | Tol % |
|---|---|---|--:|--:|--:|--:|
| - | Gross area A_g | mm^2 | 2.827e+05 |  |  | 1.5 |
| - | I_zz (about centroidal z) | mm^4 | 6.3617e+09 |  |  | 1.5 |
| - | I_yy (about centroidal y) | mm^4 | 6.3617e+09 |  |  | 1.5 |
| - | Centroid above bottom fibre | mm | 300 |  |  | 1.5 |
| AASHTO LRFD 2024 | P_o squash (nominal) | kN | 9073 |  |  | 1.0 |
| AASHTO LRFD 2024 | P_n,max (0.85*P_o) | kN | 7712 |  |  | 1.0 |
| AASHTO LRFD 2024 | M_n at P=0 (nominal) | kN.m | 431.2 |  |  | 2.0 |
| AASHTO LRFD 2024 | phi*M_n at P=0 (design) | kN.m | 388.1 |  |  | 2.0 |
| AASHTO LRFD 2024 | Balanced P_b (eps_t=eps_y) | kN | 2800 |  |  | 2.0 |
| AASHTO LRFD 2024 | Balanced M_b (nominal) | kN.m | 677.3 |  |  | 2.0 |
| Eurocode 2 | P_o squash (nominal) | kN | 6447 |  |  | 1.0 |
| Eurocode 2 | M_n at P=0 (nominal) | kN.m | 368.1 |  |  | 2.0 |
| Eurocode 2 | phi*M_n at P=0 (design) | kN.m | 368.1 |  |  | 2.0 |
| Eurocode 2 | Balanced P_b (eps_t=eps_y) | kN | 2314 |  |  | 2.0 |
| Eurocode 2 | Balanced M_b (nominal) | kN.m | 528 |  |  | 2.0 |

### S3 — S3 RC beam 300x600 (3-25M / 2-20M)

| Code | Quantity | Units | femsolver | Midas GSD | Diff % | Tol % |
|---|---|---|--:|--:|--:|--:|
| - | Gross area A_g | mm^2 | 1.8e+05 |  |  | 1.0 |
| - | I_zz (about centroidal z) | mm^4 | 5.4000e+09 |  |  | 1.0 |
| - | I_yy (about centroidal y) | mm^4 | 1.3500e+09 |  |  | 1.0 |
| - | Centroid above bottom fibre | mm | 300 |  |  | 1.0 |
| M-phi (Kent-Park + bilinear) | Cracking moment M_cr | kN.m | 61.13 |  |  | 3.0 |
| M-phi (Kent-Park + bilinear) | Ultimate moment M_u | kN.m | 404.4 |  |  | 3.0 |
| M-phi (Kent-Park + bilinear) | First-yield moment M_y | kN.m | 373.7 |  |  | 3.0 |
| M-phi (Kent-Park + bilinear) | Yield curvature kappa_y | 1/m | 0.007 |  |  | 3.0 |
| M-phi (Kent-Park + bilinear) | Curvature ductility mu_phi | - | 4.857 |  |  | 3.0 |

### S4 — S4 L-pier 800x800 t400 (12-25M)

| Code | Quantity | Units | femsolver | Midas GSD | Diff % | Tol % |
|---|---|---|--:|--:|--:|--:|
| - | Gross area A_g | mm^2 | 4.8e+05 |  |  | 1.0 |
| - | I_zz (about centroidal z) | mm^4 | 2.3467e+10 |  |  | 1.0 |
| - | I_yy (about centroidal y) | mm^4 | 2.3467e+10 |  |  | 1.0 |
| - | Centroid above bottom fibre | mm | 333.3 |  |  | 1.0 |
| AASHTO LRFD 2024 | P_o squash (nominal) | kN | 1.504e+04 |  |  | 1.0 |
| AASHTO LRFD 2024 | P_n,max (0.80*P_o) | kN | 1.203e+04 |  |  | 1.0 |
| AASHTO LRFD 2024 | M_n at P=0 (nominal) | kN.m | 1086 |  |  | 2.0 |
| AASHTO LRFD 2024 | phi*M_n at P=0 (design) | kN.m | 977.8 |  |  | 2.0 |
| AASHTO LRFD 2024 | Balanced P_b (eps_t=eps_y) | kN | 2956 |  |  | 2.0 |
| AASHTO LRFD 2024 | Balanced M_b (nominal) | kN.m | 1556 |  |  | 2.0 |
| Eurocode 2 | P_o squash (nominal) | kN | 1.062e+04 |  |  | 1.0 |
| Eurocode 2 | M_n at P=0 (nominal) | kN.m | 911.7 |  |  | 2.0 |
| Eurocode 2 | phi*M_n at P=0 (design) | kN.m | 911.7 |  |  | 2.0 |
| Eurocode 2 | Balanced P_b (eps_t=eps_y) | kN | 2319 |  |  | 2.0 |
| Eurocode 2 | Balanced M_b (nominal) | kN.m | 1195 |  |  | 2.0 |

### S5 — S5 PSC girder 400x900 (6x0.6in strand)

| Code | Quantity | Units | femsolver | Midas GSD | Diff % | Tol % |
|---|---|---|--:|--:|--:|--:|
| - | Gross area A_g | mm^2 | 3.6e+05 |  |  | 1.0 |
| - | I_zz (about centroidal z) | mm^4 | 2.4300e+10 |  |  | 1.0 |
| - | I_yy (about centroidal y) | mm^4 | 4.8000e+09 |  |  | 1.0 |
| - | Centroid above bottom fibre | mm | 450 |  |  | 1.0 |
| AASHTO LRFD 2024 | P_o squash (incl. tendon f_pu) | kN | 1.382e+04 |  |  | 1.0 |
| AASHTO LRFD 2024 | P pure tension (rebar+tendon) | kN | -1617 |  |  | 2.0 |
| Eurocode 2 | P_o squash (incl. tendon f_pu) | kN | 9723 |  |  | 1.0 |
| Eurocode 2 | P pure tension (rebar+tendon) | kN | -1591 |  |  | 2.0 |
| M-phi (Kent-Park + bilinear) | Cracking moment M_cr | kN.m | 701.5 |  |  | 3.0 |
| M-phi (Kent-Park + bilinear) | Ultimate moment M_u | kN.m | 1111 |  |  | 3.0 |
