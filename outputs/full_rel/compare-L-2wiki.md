# Screen L-2wiki (p@swa, s1eval carves)

## Against outputs\step1\fits\L-2wiki (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6468 | 0.7263 | +0.0795 [+0.0752, +0.0839] | GAIN | 0.0047 | +0.0726 | +0.1545 |
| squad | in-domain | 11873 | 0.9112 | 0.9087 | -0.0025 [-0.0046, -0.0003] | WITHIN | 0.9054 | -0.0025 | -0.0012 |
| musique | in-domain | 2417 | 0.5601 | 0.5569 | -0.0033 [-0.0092, +0.0026] | WITHIN | 0.4734 | -0.0033 | -0.0062 |
| hotpotqa | in-domain | 7405 | 0.8977 | 0.9036 | +0.0059 [+0.0032, +0.0086] | WITHIN | 0.6846 | +0.0101 | -0.0051 |
| 2wiki | zero-shot | 12576 | 0.8074 | 0.7938 | -0.0136 [-0.0159, -0.0112] | LOSS | 0.6079 | -0.0115 | -0.0120 |
| webqsp | zero-shot | 1503 | 0.1075 | 0.2231 | +0.1156 [+0.0977, +0.1340] | GAIN | 0.0535 | +0.0912 | +0.0539 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
