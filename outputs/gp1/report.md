### Screen (quarter): 9 cells read, not read: []
dot cuts [0.5894105434417725, 0.6417021155357361]; votes guard {'multihoprag': 0.0486, 'musique': 0.3977, 'squad': 0.3349, 'metaqa': 0.4516, 'webqsp': 0.5292, 'hotpotqa': 0.126, '2wiki': 0.0409} (dropped: False)

| attribute | call | score | direction | pooled AUC | multihoprag/RK | musique/RK | squad/RK | metaqa/SRK | webqsp/SRK | hotpotqa/RK | hotpotqa/SRK | 2wiki/RK | 2wiki/SRK |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A2 dot | SIGN_SPLIT | 0.000 | +1 | 0.571 | 0.694 [0.674,0.712] | 0.540 [0.524,0.557] | 0.548 [0.491,0.593] | 0.484 [0.472,0.495] | 0.700 [0.652,0.740] | 0.556 [0.531,0.580] | 0.580 [0.557,0.603] | 0.623 [0.606,0.638] | 0.640 [0.624,0.655] |
| A3 rank | SIGN_SPLIT | 0.000 | +1 | 0.520 | 0.237 [0.208,0.269] | 0.475 [0.459,0.491] | 0.383 [0.301,0.467] | 0.581 [0.569,0.595] | 0.491 [0.471,0.508] | 0.479 [0.452,0.504] | 0.458 [0.430,0.484] | 0.413 [0.394,0.432] | 0.406 [0.386,0.425] |
| A4 votes | WEAK | 0.026 | +1 | 0.576 | 0.646 [0.616,0.667] | 0.565 [0.548,0.584] | 0.581 [0.509,0.649] | 0.567 [0.553,0.582] | 0.612 [0.570,0.661] | 0.573 [0.540,0.605] | 0.545 [0.512,0.577] | 0.535 [0.517,0.554] | 0.526 [0.508,0.545] |
| A5 dir | SIGN_SPLIT | 0.000 | +1 | 0.518 | 0.639 [0.592,0.684] | 0.557 [0.541,0.575] | 0.497 [0.443,0.553] | 0.453 [0.443,0.462] | 0.514 [0.492,0.540] | 0.637 [0.616,0.659] | 0.637 [0.614,0.660] | 0.722 [0.707,0.737] | 0.720 [0.705,0.735] |
| A6 exact | SIGN_SPLIT | 0.000 | +1 | 0.549 | 0.585 [0.532,0.643] | 0.595 [0.576,0.613] | 0.521 [0.467,0.577] | 0.462 [0.452,0.471] | 0.576 [0.545,0.611] | 0.704 [0.679,0.728] | 0.675 [0.647,0.703] | 0.753 [0.740,0.766] | 0.748 [0.733,0.763] |
| A7 ntok | WEAK | 0.000 | -1 | 0.365 | 0.500 [0.500,0.500] | 0.460 [0.439,0.480] | 0.500 [0.500,0.500] | 0.358 [0.333,0.382] | 0.267 [0.219,0.323] | 0.479 [0.447,0.513] | 0.513 [0.480,0.548] | 0.499 [0.479,0.518] | 0.519 [0.499,0.539] |
| A8 caps | SIGN_SPLIT | 0.000 | +1 | 0.577 | 0.485 [0.458,0.498] | 0.507 [0.492,0.521] | 0.503 [0.502,0.505] | 0.477 [0.458,0.495] | 0.771 [0.715,0.819] | 0.544 [0.520,0.567] | 0.533 [0.506,0.559] | 0.580 [0.563,0.594] | 0.585 [0.568,0.600] |
| A9 lead | SIGN_SPLIT | 0.000 | +1 | 0.516 | 0.517 [0.472,0.578] | 0.510 [0.490,0.530] | 0.516 [0.394,0.647] | 0.500 [0.500,0.500] | 0.490 [0.469,0.501] | 0.450 [0.423,0.478] | 0.435 [0.405,0.463] | 0.568 [0.547,0.589] | 0.569 [0.548,0.590] |
| A10 off | SIGN_SPLIT | 0.000 | -1 | 0.495 | 0.472 [0.382,0.568] | 0.496 [0.468,0.522] | 0.493 [0.358,0.628] | 0.519 [0.458,0.582] | 0.462 [0.404,0.533] | 0.600 [0.567,0.628] | 0.613 [0.582,0.644] | 0.458 [0.435,0.480] | 0.460 [0.438,0.482] |
| A11 cnt | NO_SIGNAL | 0.000 | -1 | 0.479 | 0.504 [0.422,0.586] | 0.481 [0.456,0.504] | 0.509 [0.383,0.642] | 0.500 [0.500,0.500] | 0.494 [0.469,0.512] | 0.464 [0.444,0.485] | 0.453 [0.434,0.473] | 0.477 [0.466,0.489] | 0.477 [0.465,0.489] |
| A12 qov | SIGN_SPLIT | 0.000 | -1 | 0.486 | 0.611 [0.532,0.701] | 0.546 [0.531,0.561] | 0.500 [0.499,0.500] | 0.450 [0.438,0.459] | 0.510 [0.487,0.537] | 0.546 [0.516,0.577] | 0.547 [0.515,0.581] | 0.496 [0.476,0.515] | 0.503 [0.482,0.524] |

K3 = ['votes'], K4 = ['votes'], ranking ['votes']

**multihoprag/RK**: gold distance (found, 1, 2, 3, unreached) [99, 41, 1, 0, 0]; needed units 452 (weight 42.0), by step [399, 53, 0], by group S/R/K [0, 449, 3]; controls 20162 sampled
  - dot: needed low 0.00 / mid 0.02 / high 0.98; control 0.16 / 0.28 / 0.56
  - rank: needed 1-3 0.22 / 4-30 0.69 / >30 0.10; control 0.04 / 0.33 / 0.63
  - votes: needed 1 0.03 / 2 0.03 / 3+ 0.95; control 0.17 / 0.15 / 0.69
  - dir: needed none 0.43 / one-way 0.44 / mutual 0.13; control 0.68 / 0.27 / 0.05
  - exact: needed none 0.66 / folded/alias 0.34 / exact 0.00; control 0.82 / 0.18 / 0.00
  - ntok: needed 1 0.00 / 2 0.00 / 3+ 1.00; control 0.00 / 0.00 / 1.00
  - caps: needed lower 0.02 / mixed 0.98 / all caps 0.00; control 0.01 / 0.98 / 0.02
  - lead: needed no 0.30 / yes 0.04 / na 0.66; control 0.16 / 0.01 / 0.82
  - off: needed <=.10 0.15 / .10-.50 0.11 / >.50 0.09 / na 0.66; control 0.07 / 0.07 / 0.05 / 0.82
  - cnt: needed 1 0.13 / 2 0.06 / 3+ 0.16 / na 0.66; control 0.06 / 0.03 / 0.09 / 0.82
  - qov: needed 0 0.59 / 1 0.14 / 2+ 0.26; control 0.77 / 0.19 / 0.04

**musique/RK**: gold distance (found, 1, 2, 3, unreached) [512, 241, 205, 55, 4]; needed units 26889 (weight 501.0), by step [10716, 15141, 1032], by group S/R/K [0, 26611, 278]; controls 938411 sampled
  - dot: needed low 0.12 / mid 0.25 / high 0.63; control 0.52 / 0.34 / 0.14
  - rank: needed 1-3 0.10 / 4-30 0.46 / >30 0.44; control 0.00 / 0.02 / 0.98
  - votes: needed 1 0.43 / 2 0.16 / 3+ 0.41; control 0.02 / 0.01 / 0.97
  - dir: needed none 0.22 / one-way 0.66 / mutual 0.12; control 0.01 / 0.95 / 0.04
  - exact: needed none 0.46 / folded/alias 0.11 / exact 0.43; control 0.94 / 0.04 / 0.02
  - ntok: needed 1 0.28 / 2 0.36 / 3+ 0.37; control 0.08 / 0.38 / 0.54
  - caps: needed lower 0.00 / mixed 0.23 / all caps 0.77; control 0.00 / 0.23 / 0.77
  - lead: needed no 0.20 / yes 0.34 / na 0.46; control 0.03 / 0.03 / 0.94
  - off: needed <=.10 0.12 / .10-.50 0.29 / >.50 0.13 / na 0.46; control 0.02 / 0.02 / 0.01 / 0.94
  - cnt: needed 1 0.40 / 2 0.08 / 3+ 0.06 / na 0.46; control 0.03 / 0.01 / 0.02 / 0.94
  - qov: needed 0 0.79 / 1 0.13 / 2+ 0.08; control 0.97 / 0.03 / 0.00

**squad/RK**: gold distance (found, 1, 2, 3, unreached) [348, 17, 7, 3, 0]; needed units 1327 (weight 27.0), by step [714, 528, 85], by group S/R/K [0, 1317, 10]; controls 67603 sampled
  - dot: needed low 0.05 / mid 0.15 / high 0.80; control 0.19 / 0.42 / 0.38
  - rank: needed 1-3 0.22 / 4-30 0.31 / >30 0.46; control 0.01 / 0.10 / 0.89
  - votes: needed 1 0.36 / 2 0.09 / 3+ 0.56; control 0.06 / 0.03 / 0.91
  - dir: needed none 0.60 / one-way 0.31 / mutual 0.09; control 0.16 / 0.83 / 0.02
  - exact: needed none 0.72 / folded/alias 0.01 / exact 0.27; control 0.92 / 0.01 / 0.07
  - ntok: needed 1 1.00 / 2 0.00 / 3+ 0.00; control 1.00 / 0.00 / 0.00
  - caps: needed lower 0.00 / mixed 0.00 / all caps 1.00; control 0.01 / 0.00 / 0.99
  - lead: needed no 0.16 / yes 0.12 / na 0.72; control 0.05 / 0.03 / 0.92
  - off: needed <=.10 0.11 / .10-.50 0.09 / >.50 0.08 / na 0.72; control 0.02 / 0.03 / 0.03 / 0.92
  - cnt: needed 1 0.15 / 2 0.06 / 3+ 0.07 / na 0.72; control 0.05 / 0.01 / 0.01 / 0.92
  - qov: needed 0 1.00 / 1 0.00 / 2+ 0.00; control 1.00 / 0.00 / 0.00

**metaqa/SRK**: gold distance (found, 1, 2, 3, unreached) [14, 409, 1769, 1036, 0]; needed units 74373 (weight 3214.0), by step [7751, 52969, 13653], by group S/R/K [23293, 49353, 1727]; controls 841827 sampled
  - dot: needed low 0.02 / mid 0.26 / high 0.72; control 0.02 / 0.20 / 0.78
  - rank: needed 1-3 0.06 / 4-30 0.60 / >30 0.34; control 0.02 / 0.09 / 0.90
  - votes: needed 1 0.44 / 2 0.20 / 3+ 0.36; control 0.19 / 0.30 / 0.51
  - dir: needed none 0.98 / one-way 0.01 / mutual 0.01; control 0.98 / 0.01 / 0.01
  - exact: needed none 0.98 / folded/alias 0.01 / exact 0.01; control 0.99 / 0.01 / 0.00
  - ntok: needed 1 0.53 / 2 0.34 / 3+ 0.13; control 0.26 / 0.45 / 0.29
  - caps: needed lower 0.29 / mixed 0.06 / all caps 0.66; control 0.16 / 0.14 / 0.70
  - lead: needed no 0.00 / yes 0.02 / na 0.98; control 0.00 / 0.01 / 0.99
  - off: needed <=.10 0.01 / .10-.50 0.00 / >.50 0.01 / na 0.98; control 0.01 / 0.00 / 0.00 / 0.99
  - cnt: needed 1 0.02 / 2 0.00 / 3+ 0.00 / na 0.98; control 0.01 / 0.00 / 0.00 / 0.99
  - qov: needed 0 0.99 / 1 0.01 / 2+ 0.00; control 1.00 / 0.00 / 0.00

**webqsp/SRK**: gold distance (found, 1, 2, 3, unreached) [80, 1192, 704, 30, 0]; needed units 193807 (weight 1925.9999), by step [48066, 142982, 2759], by group S/R/K [4151, 189439, 217]; controls 793324 sampled
  - dot: needed low 0.01 / mid 0.14 / high 0.85; control 0.12 / 0.60 / 0.28
  - rank: needed 1-3 0.10 / 4-30 0.25 / >30 0.65; control 0.00 / 0.01 / 0.99
  - votes: needed 1 0.52 / 2 0.19 / 3+ 0.29; control 0.01 / 0.01 / 0.97
  - dir: needed none 0.76 / one-way 0.17 / mutual 0.07; control 0.96 / 0.04 / 0.00
  - exact: needed none 0.78 / folded/alias 0.04 / exact 0.19; control 1.00 / 0.00 / 0.00
  - ntok: needed 1 0.45 / 2 0.32 / 3+ 0.24; control 0.09 / 0.46 / 0.45
  - caps: needed lower 0.04 / mixed 0.13 / all caps 0.82; control 0.04 / 0.21 / 0.75
  - lead: needed no 0.07 / yes 0.15 / na 0.78; control 0.00 / 0.00 / 1.00
  - off: needed <=.10 0.05 / .10-.50 0.07 / >.50 0.10 / na 0.78; control 0.00 / 0.00 / 0.00 / 1.00
  - cnt: needed 1 0.18 / 2 0.04 / 3+ 0.00 / na 0.78; control 0.00 / 0.00 / 0.00 / 1.00
  - qov: needed 0 0.80 / 1 0.16 / 2+ 0.04; control 0.99 / 0.01 / 0.00

**hotpotqa/RK**: gold distance (found, 1, 2, 3, unreached) [553, 146, 45, 10, 0]; needed units 1405 (weight 201.0), by step [982, 341, 82], by group S/R/K [0, 1353, 52]; controls 66969 sampled
  - dot: needed low 0.10 / mid 0.11 / high 0.79; control 0.61 / 0.26 / 0.13
  - rank: needed 1-3 0.14 / 4-30 0.64 / >30 0.22; control 0.00 / 0.00 / 1.00
  - votes: needed 1 0.55 / 2 0.17 / 3+ 0.27; control 0.54 / 0.27 / 0.19
  - dir: needed none 0.04 / one-way 0.87 / mutual 0.09; control 0.00 / 0.99 / 0.01
  - exact: needed none 0.21 / folded/alias 0.11 / exact 0.68; control 0.99 / 0.01 / 0.00
  - ntok: needed 1 0.20 / 2 0.46 / 3+ 0.34; control 0.06 / 0.44 / 0.50
  - caps: needed lower 0.00 / mixed 0.14 / all caps 0.86; control 0.00 / 0.21 / 0.78
  - lead: needed no 0.45 / yes 0.35 / na 0.21; control 0.00 / 0.01 / 0.99
  - off: needed <=.10 0.08 / .10-.50 0.32 / >.50 0.39 / na 0.21; control 0.01 / 0.00 / 0.00 / 0.99
  - cnt: needed 1 0.71 / 2 0.06 / 3+ 0.02 / na 0.21; control 0.00 / 0.00 / 0.00 / 0.99
  - qov: needed 0 0.70 / 1 0.20 / 2+ 0.10; control 0.96 / 0.03 / 0.00

**hotpotqa/SRK**: gold distance (found, 1, 2, 3, unreached) [553, 172, 22, 7, 0]; needed units 918 (weight 201.0), by step [623, 224, 71], by group S/R/K [382, 502, 34]; controls 45054 sampled
  - dot: needed low 0.06 / mid 0.09 / high 0.85; control 0.68 / 0.21 / 0.11
  - rank: needed 1-3 0.14 / 4-30 0.68 / >30 0.17; control 0.00 / 0.00 / 1.00
  - votes: needed 1 0.59 / 2 0.19 / 3+ 0.22; control 0.50 / 0.26 / 0.24
  - dir: needed none 0.07 / one-way 0.84 / mutual 0.09; control 0.03 / 0.97 / 0.01
  - exact: needed none 0.22 / folded/alias 0.19 / exact 0.60; control 0.99 / 0.01 / 0.00
  - ntok: needed 1 0.14 / 2 0.47 / 3+ 0.39; control 0.04 / 0.46 / 0.50
  - caps: needed lower 0.00 / mixed 0.17 / all caps 0.83; control 0.00 / 0.22 / 0.78
  - lead: needed no 0.47 / yes 0.31 / na 0.22; control 0.00 / 0.00 / 0.99
  - off: needed <=.10 0.07 / .10-.50 0.30 / >.50 0.41 / na 0.22; control 0.00 / 0.00 / 0.00 / 0.99
  - cnt: needed 1 0.71 / 2 0.05 / 3+ 0.01 / na 0.22; control 0.00 / 0.00 / 0.00 / 0.99
  - qov: needed 0 0.69 / 1 0.22 / 2+ 0.10; control 0.97 / 0.02 / 0.01

**2wiki/RK**: gold distance (found, 1, 2, 3, unreached) [552, 301, 55, 8, 0]; needed units 1744 (weight 364.0), by step [1067, 550, 127], by group S/R/K [0, 1696, 48]; controls 102772 sampled
  - dot: needed low 0.09 / mid 0.05 / high 0.86; control 0.70 / 0.23 / 0.07
  - rank: needed 1-3 0.15 / 4-30 0.66 / >30 0.18; control 0.00 / 0.00 / 1.00
  - votes: needed 1 0.71 / 2 0.14 / 3+ 0.15; control 0.37 / 0.25 / 0.39
  - dir: needed none 0.02 / one-way 0.85 / mutual 0.13; control 0.00 / 0.99 / 0.01
  - exact: needed none 0.09 / folded/alias 0.08 / exact 0.82; control 0.99 / 0.01 / 0.00
  - ntok: needed 1 0.11 / 2 0.63 / 3+ 0.26; control 0.06 / 0.45 / 0.48
  - caps: needed lower 0.00 / mixed 0.10 / all caps 0.89; control 0.00 / 0.21 / 0.79
  - lead: needed no 0.30 / yes 0.61 / na 0.09; control 0.00 / 0.00 / 0.99
  - off: needed <=.10 0.09 / .10-.50 0.55 / >.50 0.26 / na 0.09; control 0.00 / 0.00 / 0.00 / 0.99
  - cnt: needed 1 0.83 / 2 0.06 / 3+ 0.01 / na 0.09; control 0.00 / 0.00 / 0.00 / 0.99
  - qov: needed 0 0.81 / 1 0.12 / 2+ 0.06; control 0.98 / 0.02 / 0.00

**2wiki/SRK**: gold distance (found, 1, 2, 3, unreached) [552, 322, 34, 8, 0]; needed units 1540 (weight 364.0), by step [939, 470, 131], by group S/R/K [501, 997, 42]; controls 86181 sampled
  - dot: needed low 0.06 / mid 0.04 / high 0.90; control 0.67 / 0.25 / 0.07
  - rank: needed 1-3 0.15 / 4-30 0.69 / >30 0.17; control 0.00 / 0.00 / 1.00
  - votes: needed 1 0.73 / 2 0.15 / 3+ 0.12; control 0.34 / 0.24 / 0.43
  - dir: needed none 0.03 / one-way 0.84 / mutual 0.13; control 0.01 / 0.99 / 0.01
  - exact: needed none 0.08 / folded/alias 0.13 / exact 0.78; control 0.99 / 0.01 / 0.00
  - ntok: needed 1 0.08 / 2 0.63 / 3+ 0.29; control 0.07 / 0.44 / 0.48
  - caps: needed lower 0.00 / mixed 0.11 / all caps 0.89; control 0.00 / 0.21 / 0.79
  - lead: needed no 0.30 / yes 0.62 / na 0.08; control 0.00 / 0.00 / 0.99
  - off: needed <=.10 0.09 / .10-.50 0.57 / >.50 0.26 / na 0.08; control 0.00 / 0.00 / 0.00 / 0.99
  - cnt: needed 1 0.84 / 2 0.06 / 3+ 0.01 / na 0.08; control 0.00 / 0.00 / 0.00 / 0.99
  - qov: needed 0 0.79 / 1 0.14 / 2+ 0.06; control 0.99 / 0.01 / 0.00

### Program (quarter)
- K3 ['votes']: tau L 0.9515458436077957 <= U 0.9999698465739594 <= M 0.9999698465739594 (OPTIMAL, kept 8) <= T 0.9999698465739594; greedy (old form) 1.0
- K0 ['dot']: tau L 0.9661114157066317 <= U 1.0 <= M 1.0 (OPTIMAL, kept 7) <= T 1.0; greedy (old form) 1.0
- K4 ['votes']: tau L 0.9515458436077957 <= U 0.9999698465739594 <= M 0.9999698465739594 (OPTIMAL, kept 8) <= T 0.9999698465739594; greedy (old form) 1.0
- S free: {'status': 'OPTIMAL', 'tau': 0.9999698465739594, 'cost_of_A1': 0.0}; rho: {'0.9': {'status': 'OPTIMAL', 'tau': 0.9999298509490893}, '0.99': {'status': 'OPTIMAL', 'tau': 1.0}, 'L_0.94': {'tau': 0.9418550123293549}, 'L_0.96': {'tau': 0.9612366748862365}}; size: {'12': {'status': 'OPTIMAL', 'tau': 0.9999698465739594}, '24': {'status': 'OPTIMAL', 'tau': 0.9999698465739594}, '48': {'status': 'OPTIMAL', 'tau': 0.9999698465739594}}; cost of universality 3.999562487011499e-05
- own optima: {'multihoprag/RK': {'status': 'OPTIMAL', 'tau': 0.832298437954788}, 'musique/RK': {'status': 'OPTIMAL', 'tau': 0.9995193419734762}, 'squad/RK': {'status': 'OPTIMAL', 'tau': 0.9980141490004724}, 'metaqa/SRK': {'status': 'OPTIMAL', 'tau': 0.9773845813185748}, 'webqsp/SRK': {'status': 'OPTIMAL', 'tau': 0.9996667547758645}, 'hotpotqa/RK': {'status': 'OPTIMAL', 'tau': 0.9998629925076116}, 'hotpotqa/SRK': {'status': 'OPTIMAL', 'tau': 0.5262213946410556}, '2wiki/RK': {'status': 'OPTIMAL', 'tau': 0.9999298509490893}, '2wiki/SRK': {'status': 'OPTIMAL', 'tau': 0.9999160664945151}}
- frontier: {'0.3': {'status': 'INFEASIBLE', 'reach': None}, '0.4': {'status': 'OPTIMAL', 'reach': 0.0}, '0.5': {'status': 'OPTIMAL', 'reach': 0.012030738358905287}, '0.6': {'status': 'OPTIMAL', 'reach': 0.012030738358905287}, '0.7': {'status': 'OPTIMAL', 'reach': 0.012030738358905287}, '0.8': {'status': 'OPTIMAL', 'reach': 0.012030738358905287}}
- improvement over K0 (tau points): 3.015342604062443e-05

### Grade (rest of dev)
- rule: NOT_UNIVERSAL (0/9 cells SCREEN_IN)
  - multihoprag/RK: reach 0.9998, price 0.999, path retention 1.0 (121 golds) -> False
  - musique/RK: reach 0.9686, price 0.9995, path retention 0.9646 (1469 golds) -> False
  - squad/RK: reach 0.9529, price 0.9982, path retention 0.9481 (77 golds) -> False
  - metaqa/SRK: reach 0.9777, price 0.9926, path retention 0.982 (8105 golds) -> False
  - webqsp/SRK: reach 0.9925, price 1.0, path retention 0.9928 (6632 golds) -> False
  - hotpotqa/RK: reach 0.9806, price 0.9999, path retention 0.9798 (594 golds) -> False
  - hotpotqa/SRK: reach 0.9939, price 0.9998, path retention 0.995 (595 golds) -> False
  - 2wiki/RK: reach 0.9941, price 1.0, path retention 0.9946 (1120 golds) -> False
  - 2wiki/SRK: reach 0.9941, price 1.0, path retention 0.9946 (1120 golds) -> False
- rule_K0: NOT_UNIVERSAL (0/9 cells SCREEN_IN)
  - multihoprag/RK: reach 1.0, price 1.0, path retention 1.0 (121 golds) -> False
  - musique/RK: reach 1.0, price 1.0, path retention 1.0 (1469 golds) -> False
  - squad/RK: reach 1.0, price 1.0, path retention 1.0 (77 golds) -> False
  - metaqa/SRK: reach 1.0, price 1.0, path retention 1.0 (8105 golds) -> False
  - webqsp/SRK: reach 1.0, price 1.0, path retention 0.9995 (6632 golds) -> False
  - hotpotqa/RK: reach 1.0, price 1.0, path retention 1.0 (594 golds) -> False
  - hotpotqa/SRK: reach 1.0, price 1.0, path retention 1.0 (595 golds) -> False
  - 2wiki/RK: reach 1.0, price 1.0, path retention 1.0 (1120 golds) -> False
  - 2wiki/SRK: reach 1.0, price 1.0, path retention 1.0 (1120 golds) -> False