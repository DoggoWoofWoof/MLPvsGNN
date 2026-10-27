# Screen scr-a13x-zspxb (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-g5-zspx (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8030 | 0.8032 | +0.0002 [-0.0005, +0.0011] | WITHIN | 0.0047 | +0.0005 | -0.0003 |
| squad | in-domain | 11873 | 0.9092 | 0.9096 | +0.0004 [-0.0015, +0.0023] | WITHIN | 0.9054 | +0.0004 | +0.0001 |
| musique | zero-shot | 2417 | 0.5073 | 0.4954 | -0.0119 [-0.0176, -0.0063] | LOSS | 0.4734 | -0.0153 | +0.0025 |
| hotpotqa | in-domain | 7405 | 0.9074 | 0.9087 | +0.0014 [-0.0008, +0.0034] | WITHIN | 0.6846 | +0.0035 | -0.0068 |
| 2wiki | in-domain | 12576 | 0.9351 | 0.9339 | -0.0011 [-0.0025, +0.0003] | WITHIN | 0.6079 | -0.0040 | +0.0038 |
| webqsp | zero-shot | 1503 | 0.3204 | 0.3269 | +0.0065 [-0.0003, +0.0136] | WITHIN | 0.0535 | +0.0067 | -0.0040 |

## Against outputs\step1\fits\L-musique (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6537 | 0.8032 | +0.1495 [+0.1438, +0.1547] | GAIN | 0.0047 | +0.1479 | +0.2891 |
| squad | in-domain | 11873 | 0.9100 | 0.9096 | -0.0003 [-0.0028, +0.0020] | WITHIN | 0.9054 | -0.0003 | -0.0018 |
| musique | zero-shot | 2417 | 0.2696 | 0.4954 | +0.2258 [+0.2146, +0.2379] | GAIN | 0.4734 | +0.1121 | +0.3529 |
| hotpotqa | in-domain | 7405 | 0.9015 | 0.9087 | +0.0072 [+0.0039, +0.0105] | WITHIN | 0.6846 | +0.0108 | +0.0063 |
| 2wiki | in-domain | 12576 | 0.8732 | 0.9339 | +0.0607 [+0.0576, +0.0639] | GAIN | 0.6079 | +0.1233 | +0.0688 |
| webqsp | zero-shot | 1503 | 0.1779 | 0.3269 | +0.1491 [+0.1262, +0.1736] | GAIN | 0.0535 | +0.1138 | +0.1257 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
