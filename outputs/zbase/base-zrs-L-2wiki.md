# Screen L-2wiki (p@swa, s1eval carves)

## Against outputs\step1\fits\L-2wiki (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6468 | 0.7642 | +0.1174 [+0.1124, +0.1224] | GAIN | 0.0047 | +0.1170 | +0.2183 |
| squad | in-domain | 11873 | 0.9112 | 0.9109 | -0.0003 [-0.0025, +0.0018] | WITHIN | 0.9054 | -0.0003 | -0.0010 |
| musique | in-domain | 2417 | 0.5601 | 0.5656 | +0.0055 [-0.0009, +0.0118] | WITHIN | 0.4734 | +0.0050 | +0.0029 |
| hotpotqa | in-domain | 7405 | 0.8977 | 0.9035 | +0.0058 [+0.0031, +0.0086] | WITHIN | 0.6846 | +0.0107 | +0.0046 |
| 2wiki | zero-shot | 12576 | 0.8074 | 0.7993 | -0.0081 [-0.0107, -0.0054] | LOSS | 0.6079 | -0.0177 | -0.0008 |
| webqsp | zero-shot | 1503 | 0.1075 | 0.2108 | +0.1032 [+0.0852, +0.1219] | GAIN | 0.0535 | +0.0838 | +0.0512 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
