# Screen scr-wqg-zg6r-s1-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-g6rw-zg6r-hp (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | zero-shot | 9785 | 0.3834 | 0.4355 | +0.0521 [+0.0470, +0.0570] | GAIN | 0.0047 | +0.0433 | +0.0131 |
| squad | in-domain | 11873 | 0.9111 | 0.9129 | +0.0019 [+0.0002, +0.0036] | WITHIN | 0.9054 | +0.0019 | -0.0016 |
| musique | in-domain | 2417 | 0.5941 | 0.5920 | -0.0021 [-0.0070, +0.0030] | WITHIN | 0.4734 | +0.0050 | +0.0000 |
| hotpotqa | zero-shot | 7405 | 0.8529 | 0.8488 | -0.0041 [-0.0072, -0.0011] | WITHIN | 0.6846 | -0.0063 | +0.0061 |
| 2wiki | in-domain | 12576 | 0.9420 | 0.9418 | -0.0002 [-0.0017, +0.0011] | WITHIN | 0.6079 | +0.0003 | +0.0061 |
| webqsp | in-domain | 1503 | 0.6536 | 0.6483 | -0.0053 [-0.0155, +0.0043] | WITHIN | 0.0535 | -0.0040 | -0.0106 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
