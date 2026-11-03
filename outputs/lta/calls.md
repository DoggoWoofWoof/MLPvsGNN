# LTA: zrckpa (zrckp + LTA's next-hop columns (NEXT over every pool row: each anchor's learned next-hop term on every row outside the first stage's top five, linked or not, cross-fitted)) against LTM's zrckpn, on PKO's pools, on the host's CPU

## Screen: NOT_ADOPTED

counts: {"GAIN": 0, "LOSS": 1, "WITHIN": 9}

| split | read | zrckpn s0, s1, s2 | zrckpa s0 | delta | floor | call |
|---|---|---|---:|---:|---:|---|
| WQ-musique | squad | 0.9138, 0.9125, 0.9149 | 0.9137 | -0.0001 | 0.0075 | WITHIN |
| WQ-musique | hotpotqa | 0.8460, 0.8469, 0.8491 | 0.8448 | -0.0012 | 0.0075 | WITHIN |
| WQ-musique | 2wiki | 0.9155, 0.9159, 0.9133 | 0.9119 | -0.0036 | 0.0075 | WITHIN |
| WQ-musique | webqsp | 0.6457, 0.6498, 0.6552 | 0.6470 | +0.0013 | 0.0146 | WITHIN |
| WQ-hotpotqa | squad | 0.9132, 0.9120, 0.9132 | 0.9138 | +0.0006 | 0.0075 | WITHIN |
| WQ-hotpotqa | musique | 0.5806, 0.5803, 0.5777 | 0.5761 | -0.0045 | 0.0075 | WITHIN |
| WQ-hotpotqa | 2wiki | 0.9135, 0.9166, 0.9135 | 0.9122 | -0.0013 | 0.0075 | WITHIN |
| WQ-hotpotqa | webqsp | 0.6469, 0.6597, 0.6512 | 0.6498 | +0.0029 | 0.0191 | WITHIN |
| WQ-hotpotqa | musique-3hop | 0.5351, 0.5333, 0.5263 | 0.5298 | -0.0053 | 0.0127 | WITHIN |
| WQ-hotpotqa | musique-4hop | 0.3574, 0.3562, 0.3562 | 0.3463 | -0.0111 | 0.0075 | LOSS |
| WQ-musique | musique | 0.5190, 0.5227, 0.5381 | 0.5295 | +0.0105 | 0.0275 | WITHIN (beside) |
| WQ-musique | metaqa | 0.4126, 0.4256, 0.4167 | 0.3693 | -0.0433 | 0.0193 | LOSS (beside) |
| WQ-hotpotqa | hotpotqa | 0.8193, 0.8182, 0.8136 | 0.8134 | -0.0059 | 0.0082 | WITHIN (beside) |
| WQ-hotpotqa | metaqa | 0.3668, 0.4261, 0.4131 | 0.3711 | +0.0043 | 0.1064 | WITHIN (beside) |
| WQ-musique | musique-3hop | 0.4544, 0.4596, 0.4702 | 0.4654 | +0.0110 | 0.0235 | WITHIN (beside) |
| WQ-musique | musique-4hop | 0.3086, 0.3148, 0.3352 | 0.3179 | +0.0093 | 0.0385 | WITHIN (beside) |

next (section 7): the term over every row closes; next form: LTA's rows with each A list's negatives drawn from rrf's leading rows outside the top five, not uniformly

Beside, deciding nothing: the A cases' T2 under bil, LTA's scorer among every row outside the top five (linked and unlinked targets apart; rrf's on the same cases), LTM's among linked rows (carve records):

| split | carve | A ranked | LTA bil | linked | unlinked | LTA rrf | LTM bil | musique 2/3/4-hop, LTA |
|---|---|---:|---:|---:|---:|---:|---:|---|
| WQ-musique | webqsp/fit | 663 | 0.009 | 0.045 | 0.003 | 0.032 | 0.023 |  |
| WQ-musique | squad/fit | 0 | - | - | - | - | - |  |
| WQ-musique | hotpotqa/fit | 2830 | 0.639 | 0.757 | 0.209 | 0.088 | 0.793 |  |
| WQ-musique | 2wiki/fit | 5661 | 0.599 | 0.857 | 0.193 | 0.046 | 0.915 |  |
| WQ-musique | metaqa/s1eval | 1495 | 0.001 | 0.053 | 0.000 | 0.007 | 0.105 |  |
| WQ-musique | squad/s1eval | 0 | - | - | - | - | - |  |
| WQ-musique | musique/s1eval | 2547 | 0.172 | 0.356 | 0.067 | 0.067 | 0.454 | 0.317, 0.128, 0.079 |
| WQ-musique | hotpotqa/s1eval | 3632 | 0.621 | 0.748 | 0.191 | 0.096 | 0.809 |  |
| WQ-musique | 2wiki/s1eval | 12096 | 0.557 | 0.795 | 0.152 | 0.057 | 0.865 |  |
| WQ-musique | webqsp/s1eval | 1113 | 0.005 | 0.051 | 0.001 | 0.015 | 0.061 |  |
| WQ-hotpotqa | webqsp/fit | 663 | 0.005 | 0.011 | 0.003 | 0.032 | 0.023 |  |
| WQ-hotpotqa | squad/fit | 0 | - | - | - | - | - |  |
| WQ-hotpotqa | musique/s1fit | 3798 | 0.174 | 0.331 | 0.096 | 0.066 | 0.542 | 0.224, 0.129, 0.091 |
| WQ-hotpotqa | 2wiki/fit | 5661 | 0.580 | 0.827 | 0.191 | 0.046 | 0.915 |  |
| WQ-hotpotqa | metaqa/s1eval | 1495 | 0.001 | 0.053 | 0.001 | 0.007 | 0.053 |  |
| WQ-hotpotqa | squad/s1eval | 0 | - | - | - | - | - |  |
| WQ-hotpotqa | musique/s1eval | 2547 | 0.179 | 0.346 | 0.084 | 0.067 | 0.493 | 0.331, 0.137, 0.076 |
| WQ-hotpotqa | hotpotqa/s1eval | 3632 | 0.473 | 0.579 | 0.116 | 0.096 | 0.679 |  |
| WQ-hotpotqa | 2wiki/s1eval | 12096 | 0.533 | 0.760 | 0.147 | 0.057 | 0.855 |  |
| WQ-hotpotqa | webqsp/s1eval | 1113 | 0.002 | 0.010 | 0.001 | 0.015 | 0.061 |  |

LTS's zrckps (screen NO_GAIN, full run None) - zrckpn at seed 0, as filed: WQ-musique squad -0.0005 WITHIN; WQ-musique hotpotqa -0.0012 WITHIN; WQ-musique 2wiki -0.0018 WITHIN; WQ-musique webqsp -0.0050 WITHIN; WQ-hotpotqa squad +0.0001 WITHIN; WQ-hotpotqa musique -0.0047 WITHIN; WQ-hotpotqa 2wiki -0.0033 WITHIN; WQ-hotpotqa webqsp +0.0051 WITHIN; WQ-hotpotqa musique-3hop -0.0114 WITHIN; WQ-hotpotqa musique-4hop -0.0031 WITHIN

| fit | train s | read s |
|---|---:|---:|
| scr-lta-zrckpa-s0 | 2518.4 | 3536.9 |
| scr-lta-zrckpa-s0-hp | 4739.5 | 3270.5 |
