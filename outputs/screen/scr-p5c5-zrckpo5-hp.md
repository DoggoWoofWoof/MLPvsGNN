# Screen scr-p5c5-zrckpo5-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-p1c3p-zrckp-hp (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8044 | 0.8036 | -0.0008 [-0.0015, -0.0002] | WITHIN | 0.0047 | -0.0009 | -0.0020 |
| squad | in-domain | 11873 | 0.9127 | 0.9121 | -0.0007 [-0.0024, +0.0013] | WITHIN | 0.9054 | -0.0007 | +0.0009 |
| musique | in-domain | 2417 | 0.5745 | 0.5775 | +0.0030 [-0.0019, +0.0080] | WITHIN | 0.4734 | +0.0054 | -0.0108 |
| hotpotqa | zero-shot | 7405 | 0.8513 | 0.8544 | +0.0031 [+0.0005, +0.0058] | WITHIN | 0.6846 | +0.0089 | +0.0011 |
| 2wiki | in-domain | 12576 | 0.9350 | 0.9347 | -0.0004 [-0.0018, +0.0010] | WITHIN | 0.6079 | -0.0002 | -0.0002 |
| webqsp | zero-shot | 1503 | 0.3475 | 0.3355 | -0.0120 [-0.0223, -0.0014] | LOSS | 0.0535 | -0.0086 | -0.0013 |

## Against outputs\step1\fits\L-hotpotqa (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6540 | 0.8036 | +0.1496 [+0.1440, +0.1549] | GAIN | 0.0047 | +0.1530 | +0.2864 |
| squad | in-domain | 11873 | 0.9116 | 0.9121 | +0.0004 [-0.0019, +0.0028] | WITHIN | 0.9054 | +0.0004 | +0.0034 |
| musique | in-domain | 2417 | 0.5550 | 0.5775 | +0.0225 [+0.0146, +0.0304] | GAIN | 0.4734 | +0.0236 | +0.0223 |
| hotpotqa | zero-shot | 7405 | 0.8585 | 0.8544 | -0.0041 [-0.0088, +0.0004] | WITHIN | 0.6846 | -0.0062 | -0.0334 |
| 2wiki | in-domain | 12576 | 0.8672 | 0.9347 | +0.0674 [+0.0642, +0.0709] | GAIN | 0.6079 | +0.1407 | +0.0643 |
| webqsp | zero-shot | 1503 | 0.1200 | 0.3355 | +0.2155 [+0.1916, +0.2411] | GAIN | 0.0535 | +0.1763 | +0.0672 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
