# Screen scr-u1f-zrc-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-zrct-hp (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7793 | 0.7852 | +0.0059 [+0.0039, +0.0079] | WITHIN | 0.0047 | +0.0061 | -0.0077 |
| squad | in-domain | 11873 | 0.9112 | 0.9135 | +0.0023 [+0.0000, +0.0045] | WITHIN | 0.9054 | +0.0023 | -0.0014 |
| musique | in-domain | 2417 | 0.5645 | 0.5547 | -0.0098 [-0.0167, -0.0030] | LOSS | 0.4734 | -0.0103 | +0.0070 |
| hotpotqa | zero-shot | 7405 | 0.8451 | 0.7999 | -0.0452 [-0.0499, -0.0406] | LOSS | 0.6846 | -0.0868 | -0.0030 |
| 2wiki | in-domain | 12576 | 0.8731 | 0.8623 | -0.0108 [-0.0141, -0.0074] | LOSS | 0.6079 | -0.0277 | +0.0111 |
| webqsp | zero-shot | 1503 | 0.2865 | 0.3007 | +0.0142 [-0.0001, +0.0286] | WITHIN | 0.0535 | +0.0086 | -0.0040 |

## Against outputs\step1\fits\L-hotpotqa (reported only): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6540 | 0.7852 | +0.1312 [+0.1257, +0.1366] | GAIN | 0.0047 | +0.1396 | +0.2476 |
| squad | in-domain | 11873 | 0.9116 | 0.9135 | +0.0019 [-0.0005, +0.0044] | WITHIN | 0.9054 | +0.0019 | +0.0017 |
| musique | in-domain | 2417 | 0.5550 | 0.5547 | -0.0003 [-0.0069, +0.0064] | WITHIN | 0.4734 | +0.0054 | +0.0074 |
| hotpotqa | zero-shot | 7405 | 0.8585 | 0.7999 | -0.0586 [-0.0638, -0.0533] | LOSS | 0.6846 | -0.1111 | -0.0030 |
| 2wiki | in-domain | 12576 | 0.8672 | 0.8623 | -0.0049 [-0.0083, -0.0015] | WITHIN | 0.6079 | -0.0103 | +0.0023 |
| webqsp | zero-shot | 1503 | 0.1200 | 0.3007 | +0.1807 [+0.1578, +0.2040] | GAIN | 0.0535 | +0.1450 | +0.0432 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
