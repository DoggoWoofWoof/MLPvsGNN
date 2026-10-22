# Screen scr-u2d-new-zspk (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-u2g-y-zspk (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8023 | 0.7991 | -0.0032 [-0.0047, -0.0018] | WITHIN | 0.0047 | -0.0025 | -0.0126 |
| squad | in-domain | 11873 | 0.9110 | 0.9137 | +0.0027 [+0.0008, +0.0047] | WITHIN | 0.9054 | +0.0027 | +0.0001 |
| musique | zero-shot | 2417 | 0.5286 | 0.5257 | -0.0029 [-0.0097, +0.0041] | WITHIN | 0.4734 | +0.0116 | +0.0153 |
| hotpotqa | in-domain | 7405 | 0.8394 | 0.8483 | +0.0089 [+0.0059, +0.0120] | GAIN | 0.6846 | +0.0174 | -0.0090 |
| 2wiki | in-domain | 12576 | 0.9072 | 0.9187 | +0.0115 [+0.0096, +0.0134] | GAIN | 0.6079 | +0.0294 | +0.0002 |
| webqsp | zero-shot | 1503 | 0.3009 | 0.3223 | +0.0214 [+0.0074, +0.0350] | GAIN | 0.0535 | +0.0126 | +0.0173 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
