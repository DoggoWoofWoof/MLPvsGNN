# Screen L-squad (p@swa, s1eval carves)

## Against outputs\full_zret\fits\L-squad (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6472 | 0.7631 | +0.1159 [+0.1110, +0.1208] | GAIN | 0.0047 | +0.1169 | +0.2125 |
| squad | zero-shot | 11873 | 0.8853 | 0.8853 | +0.0000 [+0.0000, +0.0000] | WITHIN | 0.9054 | +0.0000 | +0.0000 |
| musique | in-domain | 2417 | 0.5650 | 0.5650 | +0.0000 [+0.0000, +0.0000] | WITHIN | 0.4734 | +0.0000 | +0.0000 |
| hotpotqa | in-domain | 7405 | 0.9028 | 0.9028 | +0.0000 [+0.0000, +0.0000] | WITHIN | 0.6846 | +0.0000 | +0.0000 |
| 2wiki | in-domain | 12576 | 0.8676 | 0.8676 | +0.0000 [+0.0000, +0.0000] | WITHIN | 0.6079 | +0.0000 | +0.0000 |
| webqsp | zero-shot | 1503 | 0.1469 | 0.2264 | +0.0795 [+0.0602, +0.0984] | GAIN | 0.0535 | +0.0605 | +0.0552 |

## Against outputs\full_rmatch\fits\L-squad (reported only): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7785 | 0.7631 | -0.0155 [-0.0178, -0.0133] | LOSS | 0.0047 | -0.0200 | -0.0424 |
| squad | zero-shot | 11873 | 0.8893 | 0.8853 | -0.0040 [-0.0072, -0.0008] | WITHIN | 0.9054 | -0.0040 | +0.0072 |
| musique | in-domain | 2417 | 0.5670 | 0.5650 | -0.0020 [-0.0082, +0.0044] | WITHIN | 0.4734 | -0.0004 | +0.0021 |
| hotpotqa | in-domain | 7405 | 0.9007 | 0.9028 | +0.0020 [-0.0009, +0.0049] | WITHIN | 0.6846 | +0.0028 | +0.0007 |
| 2wiki | in-domain | 12576 | 0.8695 | 0.8676 | -0.0019 [-0.0042, +0.0003] | WITHIN | 0.6079 | +0.0001 | -0.0036 |
| webqsp | zero-shot | 1503 | 0.2711 | 0.2264 | -0.0447 [-0.0615, -0.0286] | LOSS | 0.0535 | -0.0373 | -0.0432 |

## Against outputs\step1\fits\L-squad (reported only): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6535 | 0.7631 | +0.1096 [+0.1049, +0.1145] | GAIN | 0.0047 | +0.1066 | +0.2085 |
| squad | zero-shot | 11873 | 0.8797 | 0.8853 | +0.0056 [+0.0024, +0.0089] | WITHIN | 0.9054 | +0.0056 | +0.0196 |
| musique | in-domain | 2417 | 0.5743 | 0.5650 | -0.0093 [-0.0153, -0.0032] | LOSS | 0.4734 | -0.0112 | -0.0079 |
| hotpotqa | in-domain | 7405 | 0.9045 | 0.9028 | -0.0017 [-0.0045, +0.0010] | WITHIN | 0.6846 | -0.0028 | +0.0140 |
| 2wiki | in-domain | 12576 | 0.8724 | 0.8676 | -0.0048 [-0.0070, -0.0026] | WITHIN | 0.6079 | -0.0085 | +0.0049 |
| webqsp | zero-shot | 1503 | 0.1302 | 0.2264 | +0.0961 [+0.0769, +0.1153] | GAIN | 0.0535 | +0.0739 | +0.0519 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
