# Screen scr-zrct (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-zrm (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7789 | 0.7782 | -0.0007 [-0.0017, +0.0003] | WITHIN | 0.0047 | -0.0007 | -0.0016 |
| squad | in-domain | 11873 | 0.9111 | 0.9105 | -0.0006 [-0.0024, +0.0013] | WITHIN | 0.9054 | -0.0006 | +0.0015 |
| musique | zero-shot | 2417 | 0.5231 | 0.5235 | +0.0004 [-0.0042, +0.0050] | WITHIN | 0.4734 | -0.0029 | -0.0095 |
| hotpotqa | in-domain | 7405 | 0.9040 | 0.9041 | +0.0001 [-0.0020, +0.0020] | WITHIN | 0.6846 | +0.0023 | +0.0119 |
| 2wiki | in-domain | 12576 | 0.8709 | 0.8733 | +0.0024 [+0.0007, +0.0041] | WITHIN | 0.6079 | +0.0054 | +0.0099 |
| webqsp | zero-shot | 1503 | 0.3128 | 0.3384 | +0.0257 [+0.0108, +0.0402] | GAIN | 0.0535 | +0.0200 | -0.0532 |

## Against outputs\screen\fits\scr-zret (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6496 | 0.7782 | +0.1286 [+0.1234, +0.1339] | GAIN | 0.0047 | +0.1321 | +0.2516 |
| squad | in-domain | 11873 | 0.9106 | 0.9105 | -0.0002 [-0.0022, +0.0019] | WITHIN | 0.9054 | -0.0002 | +0.0038 |
| musique | zero-shot | 2417 | 0.3897 | 0.5235 | +0.1338 [+0.1240, +0.1436] | GAIN | 0.4734 | +0.0786 | +0.2209 |
| hotpotqa | in-domain | 7405 | 0.9061 | 0.9041 | -0.0020 [-0.0044, +0.0003] | WITHIN | 0.6846 | -0.0023 | +0.0073 |
| 2wiki | in-domain | 12576 | 0.8714 | 0.8733 | +0.0019 [+0.0001, +0.0037] | WITHIN | 0.6079 | +0.0043 | +0.0037 |
| webqsp | zero-shot | 1503 | 0.1766 | 0.3384 | +0.1618 [+0.1391, +0.1853] | GAIN | 0.0535 | +0.1291 | +0.0579 |

## Against outputs\step1\fits\L-musique (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6537 | 0.7782 | +0.1245 [+0.1194, +0.1297] | GAIN | 0.0047 | +0.1266 | +0.2505 |
| squad | in-domain | 11873 | 0.9100 | 0.9105 | +0.0005 [-0.0018, +0.0027] | WITHIN | 0.9054 | +0.0005 | -0.0002 |
| musique | zero-shot | 2417 | 0.2696 | 0.5235 | +0.2540 [+0.2427, +0.2652] | GAIN | 0.4734 | +0.1469 | +0.3748 |
| hotpotqa | in-domain | 7405 | 0.9015 | 0.9041 | +0.0026 [-0.0001, +0.0049] | WITHIN | 0.6846 | +0.0053 | +0.0107 |
| 2wiki | in-domain | 12576 | 0.8732 | 0.8733 | +0.0001 [-0.0020, +0.0020] | WITHIN | 0.6079 | -0.0012 | +0.0078 |
| webqsp | zero-shot | 1503 | 0.1779 | 0.3384 | +0.1606 [+0.1401, +0.1831] | GAIN | 0.0535 | +0.1297 | +0.0526 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
