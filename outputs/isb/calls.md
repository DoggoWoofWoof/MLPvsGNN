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

Beside, deciding nothing (means over ISB's seeds; kept = (zg6r-m16 - zspk) / (zg6r - zspk)):

| split | dataset | seeds | zspk | zg6r | zg6r-m | zg6r-m16 | m16 - m | kept |
|---|---|---|---:|---:|---:|---:|---:|---:|
| WQ-musique | metaqa | 0 | 0.3560 | 0.3615 | 0.3169 | 0.2679 | -0.0490 | -16.018 |
| WQ-musique | squad | 0 | 0.9138 | 0.9142 | 0.9120 | 0.9136 | +0.0016 |  |
| WQ-musique | musique | 0 | 0.4879 | 0.5086 | 0.5288 | 0.5234 | -0.0054 | 1.715 |
| WQ-musique | hotpotqa | 0 | 0.8444 | 0.8593 | 0.8512 | 0.8604 | +0.0092 | 1.074 |
| WQ-musique | 2wiki | 0 | 0.9131 | 0.9269 | 0.9252 | 0.9341 | +0.0089 | 1.522 |
| WQ-musique | webqsp | 0 | 0.6403 | 0.6632 | 0.6774 | 0.6844 | +0.0070 | 1.926 |
| WQ-hotpotqa | metaqa | 0 | 0.3172 | 0.3519 | 0.2891 | 0.2415 | -0.0476 | -2.182 |
| WQ-hotpotqa | squad | 0 | 0.9131 | 0.9139 | 0.9133 | 0.9125 | -0.0008 |  |
| WQ-hotpotqa | musique | 0 | 0.5830 | 0.5912 | 0.5968 | 0.6066 | +0.0098 | 2.878 |
| WQ-hotpotqa | hotpotqa | 0 | 0.8106 | 0.8004 | 0.7887 | 0.7825 | -0.0062 |  |
| WQ-hotpotqa | 2wiki | 0 | 0.9131 | 0.9296 | 0.9265 | 0.9305 | +0.0040 | 1.055 |
| WQ-hotpotqa | webqsp | 0 | 0.6397 | 0.6658 | 0.6872 | 0.6906 | +0.0034 | 1.95 |

musique by hop count, beside (R@5 of p@swa on s1eval; zg6r-m16 - zg6r):

- WQ-musique seed 0: all 0.5086 -> 0.5234 (+0.0148); musique-3hop 0.4329 -> 0.4360 (+0.0031); musique-4hop 0.3130 -> 0.3228 (+0.0099)
- WQ-hotpotqa seed 0: all 0.5912 -> 0.6066 (+0.0155); musique-3hop 0.5333 -> 0.5539 (+0.0206); musique-4hop 0.3765 -> 0.3809 (+0.0043)
