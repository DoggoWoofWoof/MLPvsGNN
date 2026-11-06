# Screen scr-gnx-nx-s2-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-ml1-zg6r-s2-hp (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | zero-shot | 9785 | 0.3651 | 0.3733 | +0.0082 [+0.0028, +0.0140] | GAIN | 0.0047 | +0.0181 | +0.0100 |
| squad | in-domain | 11873 | 0.9127 | 0.9141 | +0.0014 [-0.0005, +0.0035] | WITHIN | 0.9054 | +0.0014 | +0.0001 |
| musique | in-domain | 2417 | 0.5939 | 0.5984 | +0.0045 [-0.0012, +0.0097] | WITHIN | 0.4734 | -0.0062 | +0.0050 |
| hotpotqa | zero-shot | 7405 | 0.7840 | 0.7995 | +0.0155 [+0.0119, +0.0191] | GAIN | 0.6846 | +0.0263 | +0.0030 |
| 2wiki | in-domain | 12576 | 0.9276 | 0.9279 | +0.0003 [-0.0015, +0.0021] | WITHIN | 0.6079 | -0.0013 | +0.0118 |
| webqsp | in-domain | 1503 | 0.6523 | 0.6621 | +0.0097 [+0.0006, +0.0191] | GAIN | 0.0535 | +0.0093 | +0.0133 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
