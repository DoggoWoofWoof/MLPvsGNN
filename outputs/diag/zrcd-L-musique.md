# Screen zrcd (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-zrm (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7789 | 0.7789 | +0.0000 [+0.0000, +0.0000] | WITHIN | 0.0047 | +0.0000 | +0.0000 |
| squad | in-domain | 11873 | 0.9111 | 0.9111 | +0.0000 [+0.0000, +0.0000] | WITHIN | 0.9054 | +0.0000 | +0.0000 |
| musique | zero-shot | 2417 | 0.5231 | 0.5231 | +0.0000 [+0.0000, +0.0000] | WITHIN | 0.4734 | +0.0000 | +0.0000 |
| hotpotqa | in-domain | 7405 | 0.9040 | 0.9040 | +0.0000 [+0.0000, +0.0000] | WITHIN | 0.6846 | +0.0000 | +0.0000 |
| 2wiki | in-domain | 12576 | 0.8709 | 0.8709 | +0.0000 [+0.0000, +0.0000] | WITHIN | 0.6079 | +0.0000 | +0.0000 |
| webqsp | zero-shot | 1503 | 0.3128 | 0.3342 | +0.0215 [+0.0074, +0.0358] | GAIN | 0.0535 | +0.0126 | -0.0459 |

## Against outputs\screen\fits\scr-zret (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6496 | 0.7789 | +0.1293 [+0.1241, +0.1347] | GAIN | 0.0047 | +0.1329 | +0.2532 |
| squad | in-domain | 11873 | 0.9106 | 0.9111 | +0.0004 [-0.0015, +0.0024] | WITHIN | 0.9054 | +0.0004 | +0.0023 |
| musique | zero-shot | 2417 | 0.3897 | 0.5231 | +0.1334 [+0.1234, +0.1434] | GAIN | 0.4734 | +0.0815 | +0.2305 |
| hotpotqa | in-domain | 7405 | 0.9061 | 0.9040 | -0.0021 [-0.0044, +0.0003] | WITHIN | 0.6846 | -0.0046 | -0.0046 |
| 2wiki | in-domain | 12576 | 0.8714 | 0.8709 | -0.0005 [-0.0023, +0.0014] | WITHIN | 0.6079 | -0.0011 | -0.0063 |
| webqsp | zero-shot | 1503 | 0.1766 | 0.3342 | +0.1576 [+0.1351, +0.1807] | GAIN | 0.0535 | +0.1218 | +0.0652 |

## Against outputs\step1\fits\L-musique (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6537 | 0.7789 | +0.1252 [+0.1201, +0.1306] | GAIN | 0.0047 | +0.1273 | +0.2521 |
| squad | in-domain | 11873 | 0.9100 | 0.9111 | +0.0011 [-0.0011, +0.0032] | WITHIN | 0.9054 | +0.0011 | -0.0017 |
| musique | zero-shot | 2417 | 0.2696 | 0.5231 | +0.2535 [+0.2422, +0.2648] | GAIN | 0.4734 | +0.1498 | +0.3844 |
| hotpotqa | in-domain | 7405 | 0.9015 | 0.9040 | +0.0025 [-0.0000, +0.0050] | WITHIN | 0.6846 | +0.0030 | -0.0012 |
| 2wiki | in-domain | 12576 | 0.8732 | 0.8709 | -0.0023 [-0.0042, -0.0004] | WITHIN | 0.6079 | -0.0066 | -0.0021 |
| webqsp | zero-shot | 1503 | 0.1779 | 0.3342 | +0.1564 [+0.1363, +0.1789] | GAIN | 0.0535 | +0.1224 | +0.0599 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
