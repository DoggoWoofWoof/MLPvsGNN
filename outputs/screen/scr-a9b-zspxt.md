# Screen scr-a9b-zspxt (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-g5-zspx (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8030 | 0.8018 | -0.0012 [-0.0021, -0.0003] | WITHIN | 0.0047 | -0.0010 | -0.0056 |
| squad | in-domain | 11873 | 0.9092 | 0.9116 | +0.0024 [+0.0005, +0.0043] | WITHIN | 0.9054 | +0.0024 | +0.0003 |
| musique | zero-shot | 2417 | 0.5073 | 0.5197 | +0.0124 [+0.0065, +0.0179] | GAIN | 0.4734 | +0.0037 | +0.0600 |
| hotpotqa | in-domain | 7405 | 0.9074 | 0.9070 | -0.0004 [-0.0027, +0.0018] | WITHIN | 0.6846 | +0.0003 | +0.0011 |
| 2wiki | in-domain | 12576 | 0.9351 | 0.9327 | -0.0023 [-0.0037, -0.0010] | WITHIN | 0.6079 | -0.0068 | +0.0077 |
| webqsp | zero-shot | 1503 | 0.3204 | 0.3150 | -0.0054 [-0.0130, +0.0024] | WITHIN | 0.0535 | -0.0007 | -0.0140 |

## Against outputs\step1\fits\L-musique (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6537 | 0.8018 | +0.1480 [+0.1424, +0.1533] | GAIN | 0.0047 | +0.1463 | +0.2838 |
| squad | in-domain | 11873 | 0.9100 | 0.9116 | +0.0016 [-0.0007, +0.0039] | WITHIN | 0.9054 | +0.0016 | -0.0015 |
| musique | zero-shot | 2417 | 0.2696 | 0.5197 | +0.2501 [+0.2384, +0.2617] | GAIN | 0.4734 | +0.1312 | +0.4104 |
| hotpotqa | in-domain | 7405 | 0.9015 | 0.9070 | +0.0055 [+0.0022, +0.0089] | WITHIN | 0.6846 | +0.0076 | +0.0142 |
| 2wiki | in-domain | 12576 | 0.8732 | 0.9327 | +0.0595 [+0.0564, +0.0627] | GAIN | 0.6079 | +0.1205 | +0.0727 |
| webqsp | zero-shot | 1503 | 0.1779 | 0.3150 | +0.1371 [+0.1142, +0.1618] | GAIN | 0.0535 | +0.1065 | +0.1158 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
