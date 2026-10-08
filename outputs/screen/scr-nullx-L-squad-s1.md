# Screen scr-nullx-L-squad-s1 (p@swa, s1eval carves)

## Against outputs\step1\fits\L-squad (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6535 | 0.6497 | -0.0037 [-0.0065, -0.0008] | WITHIN | 0.0047 | -0.0052 | -0.0062 |
| squad | zero-shot | 11873 | 0.8797 | 0.8821 | +0.0024 [-0.0005, +0.0051] | WITHIN | 0.9054 | +0.0024 | +0.0155 |
| musique | in-domain | 2417 | 0.5743 | 0.5655 | -0.0088 [-0.0146, -0.0031] | LOSS | 0.4734 | -0.0041 | -0.0103 |
| hotpotqa | in-domain | 7405 | 0.9045 | 0.9034 | -0.0010 [-0.0036, +0.0013] | WITHIN | 0.6846 | +0.0004 | +0.0162 |
| 2wiki | in-domain | 12576 | 0.8724 | 0.8726 | +0.0002 [-0.0017, +0.0022] | WITHIN | 0.6079 | +0.0054 | +0.0098 |
| webqsp | zero-shot | 1503 | 0.1302 | 0.1369 | +0.0067 [-0.0024, +0.0164] | WITHIN | 0.0535 | +0.0060 | +0.0027 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
