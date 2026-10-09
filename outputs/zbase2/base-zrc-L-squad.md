# Screen L-squad (p@swa, s1eval carves)

## Against outputs\step1\fits\L-squad (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6535 | 0.7799 | +0.1265 [+0.1212, +0.1319] | GAIN | 0.0047 | +0.1281 | +0.2522 |
| squad | zero-shot | 11873 | 0.8797 | 0.8912 | +0.0115 [+0.0083, +0.0148] | GAIN | 0.9054 | +0.0115 | +0.0243 |
| musique | in-domain | 2417 | 0.5743 | 0.5710 | -0.0033 [-0.0099, +0.0026] | WITHIN | 0.4734 | +0.0000 | -0.0025 |
| hotpotqa | in-domain | 7405 | 0.9045 | 0.9041 | -0.0003 [-0.0030, +0.0022] | WITHIN | 0.6846 | +0.0001 | +0.0204 |
| 2wiki | in-domain | 12576 | 0.8724 | 0.8687 | -0.0036 [-0.0058, -0.0015] | WITHIN | 0.6079 | -0.0064 | +0.0033 |
| webqsp | zero-shot | 1503 | 0.1302 | 0.2912 | +0.1609 [+0.1390, +0.1825] | GAIN | 0.0535 | +0.1277 | +0.0486 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
