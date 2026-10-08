# Screen L-squad (p@swa, s1eval carves)

## Against outputs\step1\fits\L-squad (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6535 | 0.7631 | +0.1096 [+0.1049, +0.1145] | GAIN | 0.0047 | +0.1066 | +0.2085 |
| squad | zero-shot | 11873 | 0.8797 | 0.8853 | +0.0056 [+0.0024, +0.0089] | WITHIN | 0.9054 | +0.0056 | +0.0196 |
| musique | in-domain | 2417 | 0.5743 | 0.5650 | -0.0093 [-0.0153, -0.0032] | LOSS | 0.4734 | -0.0112 | -0.0079 |
| hotpotqa | in-domain | 7405 | 0.9045 | 0.9028 | -0.0017 [-0.0045, +0.0010] | WITHIN | 0.6846 | -0.0028 | +0.0140 |
| 2wiki | in-domain | 12576 | 0.8724 | 0.8676 | -0.0048 [-0.0070, -0.0026] | WITHIN | 0.6079 | -0.0085 | +0.0049 |
| webqsp | zero-shot | 1503 | 0.1302 | 0.2264 | +0.0961 [+0.0769, +0.1153] | GAIN | 0.0535 | +0.0739 | +0.0519 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
