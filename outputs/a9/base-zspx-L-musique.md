# Screen scr-g5-zspx (p@swa, s1eval carves)

## Against outputs\step1\fits\L-musique (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6537 | 0.8030 | +0.1492 [+0.1437, +0.1546] | GAIN | 0.0047 | +0.1474 | +0.2894 |
| squad | in-domain | 11873 | 0.9100 | 0.9092 | -0.0008 [-0.0030, +0.0016] | WITHIN | 0.9054 | -0.0008 | -0.0019 |
| musique | zero-shot | 2417 | 0.2696 | 0.5073 | +0.2377 [+0.2256, +0.2493] | GAIN | 0.4734 | +0.1274 | +0.3504 |
| hotpotqa | in-domain | 7405 | 0.9015 | 0.9074 | +0.0059 [+0.0025, +0.0092] | WITHIN | 0.6846 | +0.0073 | +0.0131 |
| 2wiki | in-domain | 12576 | 0.8732 | 0.9351 | +0.0618 [+0.0587, +0.0650] | GAIN | 0.6079 | +0.1273 | +0.0650 |
| webqsp | zero-shot | 1503 | 0.1779 | 0.3204 | +0.1425 [+0.1196, +0.1679] | GAIN | 0.0535 | +0.1071 | +0.1297 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
