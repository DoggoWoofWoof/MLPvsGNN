# Screen scr-u2hf-base-zspk-s1-lmetaqa (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-u2hf-base-zspk-lmetaqa (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | zero-shot | 9785 | 0.2135 | 0.2253 | +0.0117 [+0.0077, +0.0155] | GAIN | 0.0047 | +0.0125 | +0.0054 |
| squad | in-domain | 11873 | 0.9114 | 0.9106 | -0.0008 [-0.0029, +0.0013] | WITHIN | 0.9054 | -0.0008 | -0.0029 |
| musique | in-domain | 2417 | 0.5831 | 0.5813 | -0.0018 [-0.0070, +0.0036] | WITHIN | 0.4734 | -0.0021 | +0.0174 |
| hotpotqa | in-domain | 7405 | 0.9108 | 0.9074 | -0.0034 [-0.0055, -0.0011] | WITHIN | 0.6846 | -0.0055 | -0.0084 |
| 2wiki | in-domain | 12576 | 0.9373 | 0.9364 | -0.0009 [-0.0024, +0.0006] | WITHIN | 0.6079 | -0.0050 | -0.0030 |
| webqsp | zero-shot | 1503 | 0.1102 | 0.1117 | +0.0016 [-0.0060, +0.0092] | WITHIN | 0.0535 | +0.0000 | -0.0040 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
