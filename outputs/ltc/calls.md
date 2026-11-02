# LTC: zrckpc (zrckp + LTM's next-hop columns (NEXT) + LTC's chained beam of hop-wise softmaxes two and three links out from the first stage's top five rows (CHAIN)) against DOB's CPU zrckp, on PKO's pools, on the host's CPU

## Screen: NO_GAIN

counts: {"GAIN": 0, "LOSS": 0, "WITHIN": 10}
DEP's sixteen-read rule beside: screen ADOPT_FOR_FULL_RUN, full run None
LTM's verdicts: screen ADOPT_FOR_FULL_RUN, full run ADOPT

| split | read | base s0, s1, s2 | zrckpc s0 | delta | floor | call |
|---|---|---|---:|---:|---:|---|
| WQ-musique | squad | 0.9127, 0.9122, 0.9138 | 0.9142 | +0.0015 | 0.0075 | WITHIN |
| WQ-musique | hotpotqa (C1R) | 0.8411, 0.8424, 0.8388 | 0.8431 | +0.0020 | 0.0075 | WITHIN |
| WQ-musique | 2wiki | 0.9070, 0.9079, 0.9049 | 0.9113 | +0.0043 | 0.0075 | WITHIN |
| WQ-musique | webqsp | 0.6522, 0.6583, 0.6566 | 0.6433 | -0.0089 | 0.0106 | WITHIN |
| WQ-hotpotqa | squad | 0.9138, 0.9129, 0.9133 | 0.9129 | -0.0009 | 0.0075 | WITHIN |
| WQ-hotpotqa | musique (C1R) | 0.5749, 0.5718, 0.5752 | 0.5751 | +0.0002 | 0.0075 | WITHIN |
| WQ-hotpotqa | 2wiki | 0.8987, 0.9082, 0.9063 | 0.9103 | +0.0116 | 0.0172 | WITHIN |
| WQ-hotpotqa | webqsp | 0.6446, 0.6524, 0.6514 | 0.6393 | -0.0053 | 0.0146 | WITHIN |
| WQ-hotpotqa | musique-3hop | 0.5241, 0.5237, 0.5232 | 0.5246 | +0.0004 | 0.0075 | WITHIN |
| WQ-hotpotqa | musique-4hop | 0.3475, 0.3556, 0.3531 | 0.3475 | +0.0000 | 0.0138 | WITHIN |
| WQ-musique | musique | 0.5152, 0.5086, 0.5134 | 0.5323 | +0.0171 | 0.0097 | GAIN (beside) |
| WQ-musique | metaqa | 0.3406, 0.3670, 0.3860 | 0.3828 | +0.0422 | 0.0743 | WITHIN (beside) |
| WQ-hotpotqa | hotpotqa | 0.8103, 0.8099, 0.8101 | 0.8169 | +0.0066 | 0.0075 | WITHIN (beside) |
| WQ-hotpotqa | metaqa | 0.3847, 0.4037, 0.3797 | 0.4065 | +0.0218 | 0.0278 | WITHIN (beside) |
| WQ-musique | musique-3hop | 0.4434, 0.4434, 0.4390 | 0.4649 | +0.0215 | 0.0075 | GAIN (beside) |
| WQ-musique | musique-4hop | 0.3142, 0.2969, 0.3154 | 0.3154 | +0.0012 | 0.0245 | WITHIN (beside) |

## Head-to-head (section 7): NOT_RUN, candidate None

LTC's full run is not ADOPT

## Next form by rule (section 7, applies: True): cumulative_residual

WQ-hotpotqa musique/s1eval chain AUC d1 0.851, d2 0.864 (margin 0.1)

Beside, deciding nothing: zrckpc - zrckpn (LTM) per read and seed:

| split | read | seed | zrckpc | zrckpn | delta |
|---|---|---:|---:|---:|---:|
| WQ-musique | squad | 0 | 0.9142 | 0.9138 | +0.0004 |
| WQ-musique | hotpotqa (C1R) | 0 | 0.8431 | 0.8460 | -0.0029 |
| WQ-musique | 2wiki | 0 | 0.9113 | 0.9155 | -0.0042 |
| WQ-musique | webqsp | 0 | 0.6433 | 0.6457 | -0.0024 |
| WQ-hotpotqa | squad | 0 | 0.9129 | 0.9132 | -0.0003 |
| WQ-hotpotqa | musique (C1R) | 0 | 0.5751 | 0.5806 | -0.0055 |
| WQ-hotpotqa | 2wiki | 0 | 0.9103 | 0.9135 | -0.0032 |
| WQ-hotpotqa | webqsp | 0 | 0.6393 | 0.6469 | -0.0076 |
| WQ-hotpotqa | musique-3hop | 0 | 0.5246 | 0.5351 | -0.0105 |
| WQ-hotpotqa | musique-4hop | 0 | 0.3475 | 0.3574 | -0.0099 |
| WQ-musique | musique | 0 | 0.5323 | 0.5190 | +0.0133 |
| WQ-musique | metaqa | 0 | 0.3828 | 0.4126 | -0.0298 |
| WQ-hotpotqa | hotpotqa | 0 | 0.8169 | 0.8193 | -0.0024 |
| WQ-hotpotqa | metaqa | 0 | 0.4065 | 0.3668 | +0.0397 |
| WQ-musique | musique-3hop | 0 | 0.4649 | 0.4544 | +0.0105 |
| WQ-musique | musique-4hop | 0 | 0.3154 | 0.3086 | +0.0068 |

Beside, deciding nothing: the chain columns per carve (ltccols' records; depth 1 is LTM's NEXT):

| split | carve | training | rows d1, d2, d3 | missed golds d1, d2, d3 | any | AUC d1, d2, d3 | hop one | s |
|---|---|---|---|---|---:|---|---:|---:|
| WQ-musique | webqsp/fit | True | 0.050, 0.164, 0.144 | 0.525, 0.484, 0.553 | 0.859 | 0.570, 0.698, 0.675 | 0.002 | 151.044 |
| WQ-musique | squad/fit | True | 0.473, 0.493, 0.495 | 0.634, 0.626, 0.624 | 0.690 | 0.654, 0.687, 0.685 | 0.002 | 58.254 |
| WQ-musique | hotpotqa/fit | True | 0.115, 0.290, 0.247 | 0.766, 0.635, 0.786 | 0.893 | 0.927, 0.800, 0.905 | 0.002 | 225.407 |
| WQ-musique | 2wiki/fit | True | 0.089, 0.223, 0.172 | 0.730, 0.371, 0.733 | 0.794 | 0.980, 0.719, 0.950 | 0.002 | 244.550 |
| WQ-musique | metaqa/s1eval | False | 0.008, 0.107, 0.076 | 0.115, 0.284, 0.190 | 0.441 | 0.613, 0.701, 0.621 | 0.002 | 755.468 |
| WQ-musique | squad/s1eval | False | 0.430, 0.452, 0.456 | 0.586, 0.590, 0.613 | 0.670 | 0.661, 0.654, 0.662 | 0.002 | 105.477 |
| WQ-musique | musique/s1eval | False | 0.047, 0.173, 0.142 | 0.511, 0.595, 0.590 | 0.736 | 0.813, 0.841, 0.852 | 0.002 | 355.500 |
| WQ-musique | hotpotqa/s1eval | False | 0.114, 0.294, 0.250 | 0.755, 0.651, 0.787 | 0.887 | 0.926, 0.802, 0.903 | 0.002 | 283.529 |
| WQ-musique | 2wiki/s1eval | False | 0.093, 0.211, 0.170 | 0.775, 0.470, 0.770 | 0.833 | 0.975, 0.779, 0.944 | 0.002 | 540.811 |
| WQ-musique | webqsp/s1eval | False | 0.048, 0.163, 0.139 | 0.442, 0.455, 0.470 | 0.764 | 0.589, 0.689, 0.703 | 0.002 | 152.462 |
| WQ-hotpotqa | webqsp/fit | True | 0.050, 0.147, 0.122 | 0.525, 0.445, 0.432 | 0.835 | 0.491, 0.612, 0.678 | 0.002 | 130.369 |
| WQ-hotpotqa | squad/fit | True | 0.473, 0.494, 0.497 | 0.634, 0.629, 0.634 | 0.690 | 0.628, 0.669, 0.656 | 0.002 | 56.647 |
| WQ-hotpotqa | musique/s1fit | True | 0.045, 0.160, 0.128 | 0.469, 0.592, 0.551 | 0.702 | 0.870, 0.886, 0.890 | 0.002 | 677.903 |
| WQ-hotpotqa | 2wiki/fit | True | 0.089, 0.218, 0.171 | 0.730, 0.390, 0.732 | 0.795 | 0.979, 0.740, 0.945 | 0.002 | 260.191 |
| WQ-hotpotqa | metaqa/s1eval | False | 0.008, 0.101, 0.048 | 0.115, 0.237, 0.151 | 0.370 | 0.570, 0.610, 0.555 | 0.002 | 687.789 |
| WQ-hotpotqa | squad/s1eval | False | 0.430, 0.454, 0.458 | 0.586, 0.593, 0.613 | 0.668 | 0.642, 0.642, 0.635 | 0.002 | 112.159 |
| WQ-hotpotqa | musique/s1eval | False | 0.047, 0.153, 0.127 | 0.511, 0.574, 0.580 | 0.727 | 0.851, 0.864, 0.871 | 0.002 | 336.753 |
| WQ-hotpotqa | hotpotqa/s1eval | False | 0.114, 0.289, 0.254 | 0.755, 0.649, 0.750 | 0.882 | 0.889, 0.794, 0.863 | 0.002 | 325.039 |
| WQ-hotpotqa | 2wiki/s1eval | False | 0.093, 0.208, 0.168 | 0.775, 0.490, 0.768 | 0.834 | 0.974, 0.785, 0.938 | 0.002 | 548.618 |
| WQ-hotpotqa | webqsp/s1eval | False | 0.048, 0.147, 0.124 | 0.442, 0.417, 0.425 | 0.737 | 0.514, 0.615, 0.651 | 0.002 | 186.098 |

| fit | train s | read s |
|---|---:|---:|
| scr-ltc-zrckpc-s0 | 2759.0 | 3518.5 |
| scr-ltc-zrckpc-s0-hp | 5009.8 | 4160.9 |
