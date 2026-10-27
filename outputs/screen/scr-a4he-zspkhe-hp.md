# Screen scr-a4he-zspkhe-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-p1c3-zspk-hp (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8040 | 0.8039 | -0.0001 [-0.0010, +0.0008] | WITHIN | 0.0047 | +0.0004 | -0.0018 |
| squad | in-domain | 11873 | 0.9135 | 0.9132 | -0.0003 [-0.0023, +0.0014] | WITHIN | 0.9054 | -0.0003 | -0.0003 |
| musique | in-domain | 2417 | 0.5754 | 0.5760 | +0.0006 [-0.0044, +0.0059] | WITHIN | 0.4734 | -0.0037 | -0.0099 |
| hotpotqa | zero-shot | 7405 | 0.8499 | 0.8598 | +0.0099 [+0.0068, +0.0130] | GAIN | 0.6846 | +0.0192 | -0.0026 |
| 2wiki | in-domain | 12576 | 0.9333 | 0.9307 | -0.0025 [-0.0042, -0.0010] | WITHIN | 0.6079 | -0.0062 | -0.0029 |
| webqsp | zero-shot | 1503 | 0.2680 | 0.2992 | +0.0312 [+0.0211, +0.0415] | GAIN | 0.0535 | +0.0233 | +0.0060 |

## Against outputs\step1\fits\L-hotpotqa (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6540 | 0.8039 | +0.1499 [+0.1442, +0.1554] | GAIN | 0.0047 | +0.1542 | +0.2880 |
| squad | in-domain | 11873 | 0.9116 | 0.9132 | +0.0015 [-0.0008, +0.0036] | WITHIN | 0.9054 | +0.0015 | +0.0020 |
| musique | in-domain | 2417 | 0.5550 | 0.5760 | +0.0210 [+0.0136, +0.0282] | GAIN | 0.4734 | +0.0203 | +0.0401 |
| hotpotqa | zero-shot | 7405 | 0.8585 | 0.8598 | +0.0012 [-0.0034, +0.0059] | WITHIN | 0.6846 | +0.0020 | -0.0421 |
| 2wiki | in-domain | 12576 | 0.8672 | 0.9307 | +0.0635 [+0.0602, +0.0670] | GAIN | 0.6079 | +0.1279 | +0.0707 |
| webqsp | zero-shot | 1503 | 0.1200 | 0.2992 | +0.1792 [+0.1542, +0.2048] | GAIN | 0.0535 | +0.1337 | +0.0971 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
