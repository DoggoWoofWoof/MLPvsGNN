# Screen scr-a12x-zspxgs (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-g5-zspx (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8030 | 0.8042 | +0.0012 [+0.0003, +0.0022] | WITHIN | 0.0047 | +0.0015 | +0.0011 |
| squad | in-domain | 11873 | 0.9092 | 0.9090 | -0.0003 [-0.0023, +0.0019] | WITHIN | 0.9054 | -0.0003 | -0.0023 |
| musique | zero-shot | 2417 | 0.5073 | 0.4940 | -0.0133 [-0.0196, -0.0074] | LOSS | 0.4734 | -0.0116 | -0.0095 |
| hotpotqa | in-domain | 7405 | 0.9074 | 0.9101 | +0.0027 [+0.0005, +0.0050] | WITHIN | 0.6846 | +0.0039 | +0.0007 |
| 2wiki | in-domain | 12576 | 0.9351 | 0.9341 | -0.0010 [-0.0025, +0.0005] | WITHIN | 0.6079 | -0.0014 | +0.0017 |
| webqsp | zero-shot | 1503 | 0.3204 | 0.3199 | -0.0004 [-0.0099, +0.0084] | WITHIN | 0.0535 | +0.0040 | -0.0160 |

## Against outputs\step1\fits\L-musique (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6537 | 0.8042 | +0.1505 [+0.1448, +0.1559] | GAIN | 0.0047 | +0.1489 | +0.2905 |
| squad | in-domain | 11873 | 0.9100 | 0.9090 | -0.0010 [-0.0033, +0.0013] | WITHIN | 0.9054 | -0.0010 | -0.0041 |
| musique | zero-shot | 2417 | 0.2696 | 0.4940 | +0.2244 [+0.2129, +0.2360] | GAIN | 0.4734 | +0.1158 | +0.3409 |
| hotpotqa | in-domain | 7405 | 0.9015 | 0.9101 | +0.0086 [+0.0053, +0.0118] | GAIN | 0.6846 | +0.0112 | +0.0138 |
| 2wiki | in-domain | 12576 | 0.8732 | 0.9341 | +0.0609 [+0.0577, +0.0641] | GAIN | 0.6079 | +0.1260 | +0.0667 |
| webqsp | zero-shot | 1503 | 0.1779 | 0.3199 | +0.1421 [+0.1202, +0.1662] | GAIN | 0.0535 | +0.1111 | +0.1138 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
