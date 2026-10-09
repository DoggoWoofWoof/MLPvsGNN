# Screen scr-zrct (p@swa, s1eval carves)

## Against outputs\step1\fits\L-musique (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6537 | 0.7782 | +0.1245 [+0.1194, +0.1297] | GAIN | 0.0047 | +0.1266 | +0.2505 |
| squad | in-domain | 11873 | 0.9100 | 0.9105 | +0.0005 [-0.0018, +0.0027] | WITHIN | 0.9054 | +0.0005 | -0.0002 |
| musique | zero-shot | 2417 | 0.2696 | 0.5235 | +0.2540 [+0.2427, +0.2652] | GAIN | 0.4734 | +0.1469 | +0.3748 |
| hotpotqa | in-domain | 7405 | 0.9015 | 0.9041 | +0.0026 [-0.0001, +0.0049] | WITHIN | 0.6846 | +0.0053 | +0.0107 |
| 2wiki | in-domain | 12576 | 0.8732 | 0.8733 | +0.0001 [-0.0020, +0.0020] | WITHIN | 0.6079 | -0.0012 | +0.0078 |
| webqsp | zero-shot | 1503 | 0.1779 | 0.3384 | +0.1606 [+0.1401, +0.1831] | GAIN | 0.0535 | +0.1297 | +0.0526 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
