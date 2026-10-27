# Screen scr-a9b-zspxt-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-g5-zspx-hp (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8039 | 0.8032 | -0.0008 [-0.0016, +0.0001] | WITHIN | 0.0047 | -0.0021 | -0.0064 |
| squad | in-domain | 11873 | 0.9129 | 0.9134 | +0.0005 [-0.0013, +0.0025] | WITHIN | 0.9054 | +0.0005 | -0.0019 |
| musique | in-domain | 2417 | 0.5901 | 0.5786 | -0.0116 [-0.0164, -0.0065] | LOSS | 0.4734 | -0.0161 | -0.0091 |
| hotpotqa | zero-shot | 7405 | 0.8547 | 0.8513 | -0.0034 [-0.0062, -0.0004] | WITHIN | 0.6846 | -0.0068 | -0.0194 |
| 2wiki | in-domain | 12576 | 0.9312 | 0.9308 | -0.0004 [-0.0020, +0.0011] | WITHIN | 0.6079 | -0.0045 | -0.0021 |
| webqsp | zero-shot | 1503 | 0.2914 | 0.2677 | -0.0237 [-0.0322, -0.0155] | LOSS | 0.0535 | -0.0173 | -0.0133 |

## Against outputs\step1\fits\L-hotpotqa (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6540 | 0.8032 | +0.1492 [+0.1434, +0.1545] | GAIN | 0.0047 | +0.1527 | +0.2861 |
| squad | in-domain | 11873 | 0.9116 | 0.9134 | +0.0018 [-0.0004, +0.0040] | WITHIN | 0.9054 | +0.0018 | +0.0000 |
| musique | in-domain | 2417 | 0.5550 | 0.5786 | +0.0235 [+0.0158, +0.0312] | GAIN | 0.4734 | +0.0261 | +0.0459 |
| hotpotqa | zero-shot | 7405 | 0.8585 | 0.8513 | -0.0072 [-0.0122, -0.0026] | WITHIN | 0.6846 | -0.0134 | -0.0432 |
| 2wiki | in-domain | 12576 | 0.8672 | 0.9308 | +0.0636 [+0.0602, +0.0672] | GAIN | 0.6079 | +0.1267 | +0.0726 |
| webqsp | zero-shot | 1503 | 0.1200 | 0.2677 | +0.1477 [+0.1237, +0.1717] | GAIN | 0.0535 | +0.1098 | +0.0778 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
