# Screen scr-a3z-zrckpz (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-p1c3p-zrckp (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8019 | 0.8001 | -0.0018 [-0.0028, -0.0009] | WITHIN | 0.0047 | -0.0012 | -0.0098 |
| squad | in-domain | 11873 | 0.9116 | 0.9111 | -0.0006 [-0.0023, +0.0010] | WITHIN | 0.9054 | -0.0006 | -0.0013 |
| musique | zero-shot | 2417 | 0.5121 | 0.5107 | -0.0015 [-0.0063, +0.0037] | WITHIN | 0.4734 | -0.0004 | +0.0248 |
| hotpotqa | in-domain | 7405 | 0.9065 | 0.9048 | -0.0018 [-0.0038, +0.0005] | WITHIN | 0.6846 | -0.0023 | -0.0024 |
| 2wiki | in-domain | 12576 | 0.9347 | 0.9305 | -0.0042 [-0.0059, -0.0026] | WITHIN | 0.6079 | -0.0108 | +0.0033 |
| webqsp | zero-shot | 1503 | 0.3627 | 0.3474 | -0.0153 [-0.0252, -0.0058] | LOSS | 0.0535 | -0.0180 | -0.0113 |

## Against outputs\step1\fits\L-musique (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6537 | 0.8001 | +0.1463 [+0.1408, +0.1516] | GAIN | 0.0047 | +0.1442 | +0.2713 |
| squad | in-domain | 11873 | 0.9100 | 0.9111 | +0.0011 [-0.0011, +0.0033] | WITHIN | 0.9054 | +0.0011 | -0.0024 |
| musique | zero-shot | 2417 | 0.2696 | 0.5107 | +0.2411 [+0.2290, +0.2537] | GAIN | 0.4734 | +0.1204 | +0.4381 |
| hotpotqa | in-domain | 7405 | 0.9015 | 0.9048 | +0.0033 [-0.0001, +0.0068] | WITHIN | 0.6846 | +0.0019 | -0.0099 |
| 2wiki | in-domain | 12576 | 0.8732 | 0.9305 | +0.0573 [+0.0541, +0.0605] | GAIN | 0.6079 | +0.1155 | +0.0759 |
| webqsp | zero-shot | 1503 | 0.1779 | 0.3474 | +0.1696 [+0.1484, +0.1922] | GAIN | 0.0535 | +0.1404 | +0.0612 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
