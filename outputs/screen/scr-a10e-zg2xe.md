# Screen scr-a10e-zg2xe (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-g5-zspx (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8030 | 0.8031 | +0.0001 [-0.0007, +0.0009] | WITHIN | 0.0047 | +0.0000 | +0.0009 |
| squad | in-domain | 11873 | 0.9092 | 0.9116 | +0.0024 [+0.0005, +0.0042] | WITHIN | 0.9054 | +0.0024 | +0.0009 |
| musique | zero-shot | 2417 | 0.5073 | 0.4903 | -0.0169 [-0.0225, -0.0114] | LOSS | 0.4734 | -0.0186 | -0.0410 |
| hotpotqa | in-domain | 7405 | 0.9074 | 0.9085 | +0.0011 [-0.0010, +0.0031] | WITHIN | 0.6846 | +0.0030 | -0.0062 |
| 2wiki | in-domain | 12576 | 0.9351 | 0.9297 | -0.0054 [-0.0068, -0.0039] | WITHIN | 0.6079 | -0.0147 | +0.0015 |
| webqsp | zero-shot | 1503 | 0.3204 | 0.3167 | -0.0037 [-0.0116, +0.0040] | WITHIN | 0.0535 | -0.0013 | -0.0013 |

## Against outputs\step1\fits\L-musique (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6537 | 0.8031 | +0.1493 [+0.1437, +0.1547] | GAIN | 0.0047 | +0.1474 | +0.2903 |
| squad | in-domain | 11873 | 0.9100 | 0.9116 | +0.0016 [-0.0007, +0.0040] | WITHIN | 0.9054 | +0.0016 | -0.0009 |
| musique | zero-shot | 2417 | 0.2696 | 0.4903 | +0.2208 [+0.2090, +0.2331] | GAIN | 0.4734 | +0.1088 | +0.3095 |
| hotpotqa | in-domain | 7405 | 0.9015 | 0.9085 | +0.0070 [+0.0038, +0.0103] | WITHIN | 0.6846 | +0.0103 | +0.0069 |
| 2wiki | in-domain | 12576 | 0.8732 | 0.9297 | +0.0564 [+0.0534, +0.0597] | GAIN | 0.6079 | +0.1126 | +0.0665 |
| webqsp | zero-shot | 1503 | 0.1779 | 0.3167 | +0.1388 [+0.1151, +0.1636] | GAIN | 0.0535 | +0.1058 | +0.1284 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
