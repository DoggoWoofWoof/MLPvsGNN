# Screen scr-a8ex-zspxqe-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-g5-zspx-hp (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8039 | 0.8054 | +0.0015 [+0.0006, +0.0024] | WITHIN | 0.0047 | +0.0011 | +0.0030 |
| squad | in-domain | 11873 | 0.9129 | 0.9129 | +0.0000 [-0.0021, +0.0022] | WITHIN | 0.9054 | +0.0000 | +0.0000 |
| musique | in-domain | 2417 | 0.5901 | 0.5832 | -0.0069 [-0.0120, -0.0016] | WITHIN | 0.4734 | -0.0108 | -0.0095 |
| hotpotqa | zero-shot | 7405 | 0.8547 | 0.8473 | -0.0074 [-0.0103, -0.0043] | WITHIN | 0.6846 | -0.0123 | -0.0143 |
| 2wiki | in-domain | 12576 | 0.9312 | 0.9341 | +0.0029 [+0.0014, +0.0043] | WITHIN | 0.6079 | +0.0077 | -0.0021 |
| webqsp | zero-shot | 1503 | 0.2914 | 0.2825 | -0.0089 [-0.0179, +0.0003] | WITHIN | 0.0535 | -0.0047 | +0.0047 |

## Against outputs\step1\fits\L-hotpotqa (reported only): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6540 | 0.8054 | +0.1515 [+0.1459, +0.1570] | GAIN | 0.0047 | +0.1560 | +0.2955 |
| squad | in-domain | 11873 | 0.9116 | 0.9129 | +0.0013 [-0.0013, +0.0040] | WITHIN | 0.9054 | +0.0013 | +0.0019 |
| musique | in-domain | 2417 | 0.5550 | 0.5832 | +0.0282 [+0.0207, +0.0358] | GAIN | 0.4734 | +0.0314 | +0.0455 |
| hotpotqa | zero-shot | 7405 | 0.8585 | 0.8473 | -0.0112 [-0.0163, -0.0061] | LOSS | 0.6846 | -0.0189 | -0.0381 |
| 2wiki | in-domain | 12576 | 0.8672 | 0.9341 | +0.0669 [+0.0636, +0.0706] | GAIN | 0.6079 | +0.1388 | +0.0727 |
| webqsp | zero-shot | 1503 | 0.1200 | 0.2825 | +0.1626 [+0.1385, +0.1866] | GAIN | 0.0535 | +0.1224 | +0.0958 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
