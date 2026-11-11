# ISB: zg6r-m16 (zg6r from its own encoder start, 16 epochs, SWA from epoch 10) against zg6r, on PKO's pools

## Screen (seed 0): KEEP

counts: {"GAIN": 3, "LOSS": 2, "WITHIN": 7}

| split | dataset | zg6r s0, s1, s2 | ISB s0 | delta | floor | call |
|---|---|---|---:|---:|---:|---|
| WQ-musique | metaqa | 0.3615, 0.3477, 0.3461 | 0.2679 | -0.0936 | 0.0621 | LOSS |
| WQ-musique | squad | 0.9142, 0.9145, 0.9163 | 0.9136 | -0.0006 | 0.0075 | WITHIN |
| WQ-musique | musique | 0.5086, 0.5039, 0.5134 | 0.5234 | +0.0148 | 0.0110 | GAIN |
| WQ-musique | hotpotqa | 0.8593, 0.8596, 0.8562 | 0.8604 | +0.0011 | 0.0075 | WITHIN |
| WQ-musique | 2wiki | 0.9269, 0.9297, 0.9215 | 0.9341 | +0.0072 | 0.0075 | WITHIN |
| WQ-musique | webqsp | 0.6632, 0.6586, 0.6593 | 0.6844 | +0.0212 | 0.0142 | GAIN |
| WQ-hotpotqa | metaqa | 0.3519, 0.3903, 0.3651 | 0.2415 | -0.1104 | 0.0637 | LOSS |
| WQ-hotpotqa | squad | 0.9139, 0.9140, 0.9127 | 0.9125 | -0.0014 | 0.0075 | WITHIN |
| WQ-hotpotqa | musique | 0.5912, 0.5988, 0.5939 | 0.6066 | +0.0154 | 0.0126 | GAIN |
| WQ-hotpotqa | hotpotqa | 0.8004, 0.7943, 0.7840 | 0.7825 | -0.0179 | 0.0200 | WITHIN |
| WQ-hotpotqa | 2wiki | 0.9296, 0.9290, 0.9276 | 0.9305 | +0.0009 | 0.0075 | WITHIN |
| WQ-hotpotqa | webqsp | 0.6658, 0.6512, 0.6523 | 0.6906 | +0.0248 | 0.0292 | WITHIN |

## Full run (ISO's rule): KEEP

counts: {"GAIN": 1, "LOSS": 3, "UNCONFIRMED_GAIN": 0, "WITHIN": 8, "ISO_LOSS": 3}

| split | dataset | d0 | d1 | d2 | mean | floor | ML1 call | ISO loss |
|---|---|---:|---:|---:|---:|---:|---|---|
| WQ-musique | metaqa | -0.0936 | -0.0697 | -0.0603 | -0.0745 | 0.0292 | LOSS | yes |
| WQ-musique | squad | -0.0006 | -0.0013 | -0.0024 | -0.0014 | 0.0075 | WITHIN | no |
| WQ-musique | musique | +0.0148 | +0.0184 | +0.0267 | +0.0200 | 0.0180 | WITHIN | no |
| WQ-musique | hotpotqa | +0.0011 | -0.0027 | +0.0042 | +0.0009 | 0.0075 | WITHIN | no |
| WQ-musique | 2wiki | +0.0072 | +0.0034 | +0.0113 | +0.0073 | 0.0075 | WITHIN | no |
| WQ-musique | webqsp | +0.0212 | +0.0098 | +0.0375 | +0.0228 | 0.0211 | GAIN | no |
| WQ-hotpotqa | metaqa | -0.1104 | -0.1974 | -0.1439 | -0.1506 | 0.0665 | LOSS | yes |
| WQ-hotpotqa | squad | -0.0014 | -0.0013 | -0.0016 | -0.0014 | 0.0075 | WITHIN | no |
| WQ-hotpotqa | musique | +0.0154 | -0.0037 | +0.0034 | +0.0050 | 0.0168 | WITHIN | no |
| WQ-hotpotqa | hotpotqa | -0.0179 | -0.0129 | -0.0030 | -0.0113 | 0.0176 | LOSS | yes |
| WQ-hotpotqa | 2wiki | +0.0009 | +0.0039 | +0.0042 | +0.0030 | 0.0075 | WITHIN | no |
| WQ-hotpotqa | webqsp | +0.0248 | +0.0115 | +0.0296 | +0.0220 | 0.0353 | WITHIN | no |

Beside, deciding nothing (means over ISB's seeds; kept = (zg6r-m16 - zspk) / (zg6r - zspk)):

| split | dataset | seeds | zspk | zg6r | zg6r-m | zg6r-m16 | m16 - m | kept |
|---|---|---|---:|---:|---:|---:|---:|---:|
| WQ-musique | metaqa | 0,1,2 | 0.3574 | 0.3518 | 0.3443 | 0.2772 | -0.0671 |  |
| WQ-musique | squad | 0,1,2 | 0.9139 | 0.9150 | 0.9140 | 0.9136 | -0.0005 |  |
| WQ-musique | musique | 0,1,2 | 0.4934 | 0.5086 | 0.5317 | 0.5286 | -0.0031 | 2.308 |
| WQ-musique | hotpotqa | 0,1,2 | 0.8440 | 0.8584 | 0.8530 | 0.8592 | +0.0062 | 1.06 |
| WQ-musique | 2wiki | 0,1,2 | 0.9105 | 0.9260 | 0.9268 | 0.9333 | +0.0065 | 1.469 |
| WQ-musique | webqsp | 0,1,2 | 0.6403 | 0.6604 | 0.6756 | 0.6832 | +0.0076 | 2.14 |
| WQ-hotpotqa | metaqa | 0,1,2 | 0.3199 | 0.3691 | 0.3111 | 0.2185 | -0.0926 | -2.062 |
| WQ-hotpotqa | squad | 0,1,2 | 0.9133 | 0.9135 | 0.9132 | 0.9121 | -0.0011 |  |
| WQ-hotpotqa | musique | 0,1,2 | 0.5788 | 0.5946 | 0.5923 | 0.5997 | +0.0074 | 1.317 |
| WQ-hotpotqa | hotpotqa | 0,1,2 | 0.8091 | 0.7929 | 0.7841 | 0.7816 | -0.0025 |  |
| WQ-hotpotqa | 2wiki | 0,1,2 | 0.9143 | 0.9287 | 0.9259 | 0.9317 | +0.0058 | 1.207 |
| WQ-hotpotqa | webqsp | 0,1,2 | 0.6345 | 0.6564 | 0.6902 | 0.6784 | -0.0118 | 2.002 |

musique by hop count, beside (R@5 of p@swa on s1eval; zg6r-m16 - zg6r):

- WQ-musique seed 0: all 0.5086 -> 0.5234 (+0.0148); musique-3hop 0.4329 -> 0.4360 (+0.0031); musique-4hop 0.3130 -> 0.3228 (+0.0099)
- WQ-hotpotqa seed 0: all 0.5912 -> 0.6066 (+0.0155); musique-3hop 0.5333 -> 0.5539 (+0.0206); musique-4hop 0.3765 -> 0.3809 (+0.0043)
- WQ-musique seed 1: all 0.5039 -> 0.5223 (+0.0184); musique-3hop 0.4272 -> 0.4386 (+0.0114); musique-4hop 0.3167 -> 0.3284 (+0.0117)
- WQ-hotpotqa seed 1: all 0.5988 -> 0.5951 (-0.0037); musique-3hop 0.5443 -> 0.5417 (-0.0026); musique-4hop 0.3833 -> 0.3784 (-0.0049)
- WQ-musique seed 2: all 0.5134 -> 0.5401 (+0.0267); musique-3hop 0.4404 -> 0.4662 (+0.0259); musique-4hop 0.3179 -> 0.3432 (+0.0253)
- WQ-hotpotqa seed 2: all 0.5939 -> 0.5973 (+0.0034); musique-3hop 0.5430 -> 0.5395 (-0.0035); musique-4hop 0.3772 -> 0.3870 (+0.0099)
