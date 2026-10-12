# Screen scr-u1f-zrc (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-zrct (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7782 | 0.7857 | +0.0074 [+0.0054, +0.0095] | WITHIN | 0.0047 | +0.0075 | -0.0018 |
| squad | in-domain | 11873 | 0.9105 | 0.9117 | +0.0013 [-0.0008, +0.0034] | WITHIN | 0.9054 | +0.0013 | -0.0036 |
| musique | zero-shot | 2417 | 0.5235 | 0.4977 | -0.0259 [-0.0342, -0.0178] | LOSS | 0.4734 | -0.0468 | -0.0968 |
| hotpotqa | in-domain | 7405 | 0.9041 | 0.8454 | -0.0586 [-0.0631, -0.0538] | LOSS | 0.6846 | -0.1188 | -0.0026 |
| 2wiki | in-domain | 12576 | 0.8733 | 0.8635 | -0.0098 [-0.0129, -0.0065] | LOSS | 0.6079 | -0.0259 | +0.0013 |
| webqsp | zero-shot | 1503 | 0.3384 | 0.3421 | +0.0037 [-0.0106, +0.0174] | WITHIN | 0.0535 | +0.0033 | -0.0053 |

## Against outputs\step1\fits\L-musique (reported only): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6537 | 0.7857 | +0.1319 [+0.1266, +0.1370] | GAIN | 0.0047 | +0.1341 | +0.2486 |
| squad | in-domain | 11873 | 0.9100 | 0.9117 | +0.0018 [-0.0003, +0.0040] | WITHIN | 0.9054 | +0.0018 | -0.0038 |
| musique | zero-shot | 2417 | 0.2696 | 0.4977 | +0.2281 [+0.2163, +0.2404] | GAIN | 0.4734 | +0.1001 | +0.2780 |
| hotpotqa | in-domain | 7405 | 0.9015 | 0.8454 | -0.0560 [-0.0607, -0.0513] | LOSS | 0.6846 | -0.1136 | +0.0081 |
| 2wiki | in-domain | 12576 | 0.8732 | 0.8635 | -0.0097 [-0.0131, -0.0064] | LOSS | 0.6079 | -0.0271 | +0.0091 |
| webqsp | zero-shot | 1503 | 0.1779 | 0.3421 | +0.1643 [+0.1417, +0.1870] | GAIN | 0.0535 | +0.1331 | +0.0472 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
