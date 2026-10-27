# Screen scr-a2xi-zspkxi-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-p1c3-zspk-hp (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8040 | 0.7937 | -0.0103 [-0.0117, -0.0089] | LOSS | 0.0047 | -0.0095 | -0.0463 |
| squad | in-domain | 11873 | 0.9135 | 0.9106 | -0.0029 [-0.0053, -0.0006] | WITHIN | 0.9054 | -0.0029 | -0.0147 |
| musique | in-domain | 2417 | 0.5754 | 0.5703 | -0.0051 [-0.0109, +0.0003] | WITHIN | 0.4734 | -0.0087 | -0.0207 |
| hotpotqa | zero-shot | 7405 | 0.8499 | 0.8367 | -0.0132 [-0.0167, -0.0097] | LOSS | 0.6846 | -0.0235 | -0.0170 |
| 2wiki | in-domain | 12576 | 0.9333 | 0.9348 | +0.0015 [-0.0003, +0.0033] | WITHIN | 0.6079 | +0.0051 | -0.0104 |
| webqsp | zero-shot | 1503 | 0.2680 | 0.2747 | +0.0068 [-0.0078, +0.0204] | WITHIN | 0.0535 | +0.0000 | +0.0047 |

## Against outputs\step1\fits\L-hotpotqa (reported only): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6540 | 0.7937 | +0.1398 [+0.1342, +0.1451] | GAIN | 0.0047 | +0.1443 | +0.2435 |
| squad | in-domain | 11873 | 0.9116 | 0.9106 | -0.0011 [-0.0036, +0.0014] | WITHIN | 0.9054 | -0.0011 | -0.0124 |
| musique | in-domain | 2417 | 0.5550 | 0.5703 | +0.0153 [+0.0074, +0.0233] | GAIN | 0.4734 | +0.0153 | +0.0294 |
| hotpotqa | zero-shot | 7405 | 0.8585 | 0.8367 | -0.0219 [-0.0270, -0.0168] | LOSS | 0.6846 | -0.0406 | -0.0566 |
| 2wiki | in-domain | 12576 | 0.8672 | 0.9348 | +0.0675 [+0.0642, +0.0711] | GAIN | 0.6079 | +0.1392 | +0.0631 |
| webqsp | zero-shot | 1503 | 0.1200 | 0.2747 | +0.1548 [+0.1285, +0.1796] | GAIN | 0.0535 | +0.1104 | +0.0958 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
