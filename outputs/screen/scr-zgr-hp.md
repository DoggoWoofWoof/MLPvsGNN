# Screen scr-zgr-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-zrm-hp (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7801 | 0.7826 | +0.0026 [+0.0013, +0.0039] | WITHIN | 0.0047 | +0.0014 | +0.0066 |
| squad | in-domain | 11873 | 0.9116 | 0.8975 | -0.0141 [-0.0171, -0.0109] | LOSS | 0.9054 | -0.0141 | -0.0296 |
| musique | in-domain | 2417 | 0.5600 | 0.5125 | -0.0474 [-0.0556, -0.0393] | LOSS | 0.4734 | -0.0484 | -0.1233 |
| hotpotqa | zero-shot | 7405 | 0.8457 | 0.8194 | -0.0263 [-0.0308, -0.0221] | LOSS | 0.6846 | -0.0493 | -0.0446 |
| 2wiki | in-domain | 12576 | 0.8681 | 0.8656 | -0.0025 [-0.0049, +0.0001] | WITHIN | 0.6079 | -0.0029 | -0.0305 |
| webqsp | zero-shot | 1503 | 0.2416 | 0.2350 | -0.0065 [-0.0199, +0.0068] | WITHIN | 0.0535 | -0.0053 | -0.0047 |

## Against outputs\full_zret\fits\L-hotpotqa (reported only): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6419 | 0.7826 | +0.1407 [+0.1355, +0.1462] | GAIN | 0.0047 | +0.1453 | +0.2717 |
| squad | in-domain | 11873 | 0.9127 | 0.8975 | -0.0152 [-0.0182, -0.0121] | LOSS | 0.9054 | -0.0152 | -0.0290 |
| musique | in-domain | 2417 | 0.5623 | 0.5125 | -0.0498 [-0.0583, -0.0420] | LOSS | 0.4734 | -0.0480 | -0.1237 |
| hotpotqa | zero-shot | 7405 | 0.8445 | 0.8194 | -0.0251 [-0.0292, -0.0207] | LOSS | 0.6846 | -0.0469 | -0.0408 |
| 2wiki | in-domain | 12576 | 0.8728 | 0.8656 | -0.0072 [-0.0097, -0.0046] | WITHIN | 0.6079 | -0.0160 | -0.0335 |
| webqsp | zero-shot | 1503 | 0.1195 | 0.2350 | +0.1155 [+0.0944, +0.1360] | GAIN | 0.0535 | +0.0898 | +0.0632 |

## Against outputs\step1\fits\L-hotpotqa (reported only): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6540 | 0.7826 | +0.1287 [+0.1235, +0.1338] | GAIN | 0.0047 | +0.1353 | +0.2622 |
| squad | in-domain | 11873 | 0.9116 | 0.8975 | -0.0141 [-0.0171, -0.0110] | LOSS | 0.9054 | -0.0141 | -0.0276 |
| musique | in-domain | 2417 | 0.5550 | 0.5125 | -0.0425 [-0.0507, -0.0344] | LOSS | 0.4734 | -0.0306 | -0.1109 |
| hotpotqa | zero-shot | 7405 | 0.8585 | 0.8194 | -0.0392 [-0.0439, -0.0343] | LOSS | 0.6846 | -0.0728 | -0.0402 |
| 2wiki | in-domain | 12576 | 0.8672 | 0.8656 | -0.0016 [-0.0042, +0.0011] | WITHIN | 0.6079 | -0.0002 | -0.0370 |
| webqsp | zero-shot | 1503 | 0.1200 | 0.2350 | +0.1151 [+0.0942, +0.1361] | GAIN | 0.0535 | +0.0878 | +0.0585 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
