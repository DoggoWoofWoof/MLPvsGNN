# Screen scr-r1-zspkr-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-p1c3-zspk-hp (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8040 | 0.8024 | -0.0016 [-0.0024, -0.0006] | WITHIN | 0.0047 | -0.0009 | -0.0081 |
| squad | in-domain | 11873 | 0.9135 | 0.9132 | -0.0003 [-0.0021, +0.0017] | WITHIN | 0.9054 | -0.0003 | -0.0032 |
| musique | in-domain | 2417 | 0.5754 | 0.5765 | +0.0011 [-0.0042, +0.0064] | WITHIN | 0.4734 | -0.0058 | +0.0008 |
| hotpotqa | zero-shot | 7405 | 0.8499 | 0.8467 | -0.0032 [-0.0061, -0.0003] | WITHIN | 0.6846 | -0.0074 | +0.0181 |
| 2wiki | in-domain | 12576 | 0.9333 | 0.9324 | -0.0009 [-0.0025, +0.0007] | WITHIN | 0.6079 | -0.0040 | +0.0006 |
| webqsp | zero-shot | 1503 | 0.2680 | 0.2758 | +0.0078 [-0.0012, +0.0169] | WITHIN | 0.0535 | +0.0060 | -0.0047 |

## Against outputs\step1\fits\L-hotpotqa (reported only): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6540 | 0.8024 | +0.1484 [+0.1428, +0.1539] | GAIN | 0.0047 | +0.1529 | +0.2818 |
| squad | in-domain | 11873 | 0.9116 | 0.9132 | +0.0016 [-0.0005, +0.0039] | WITHIN | 0.9054 | +0.0016 | -0.0008 |
| musique | in-domain | 2417 | 0.5550 | 0.5765 | +0.0215 [+0.0139, +0.0293] | GAIN | 0.4734 | +0.0182 | +0.0509 |
| hotpotqa | zero-shot | 7405 | 0.8585 | 0.8467 | -0.0119 [-0.0166, -0.0070] | LOSS | 0.6846 | -0.0246 | -0.0215 |
| 2wiki | in-domain | 12576 | 0.8672 | 0.9324 | +0.0652 [+0.0618, +0.0688] | GAIN | 0.6079 | +0.1302 | +0.0742 |
| webqsp | zero-shot | 1503 | 0.1200 | 0.2758 | +0.1558 [+0.1316, +0.1796] | GAIN | 0.0535 | +0.1164 | +0.0865 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
