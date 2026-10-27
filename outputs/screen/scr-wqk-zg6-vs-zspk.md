# Screen scr-wqk-zg6-s0 (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-wqk-zspk-s0 (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | zero-shot | 9785 | 0.3984 | 0.4393 | +0.0409 [+0.0348, +0.0464] | GAIN | 0.0047 | +0.0283 | -0.0344 |
| squad | in-domain | 11873 | 0.9110 | 0.9097 | -0.0013 [-0.0032, +0.0008] | WITHIN | 0.9054 | -0.0013 | +0.0013 |
| musique | zero-shot | 2417 | 0.4590 | 0.5018 | +0.0428 [+0.0336, +0.0518] | GAIN | 0.4734 | +0.0331 | +0.1477 |
| hotpotqa | in-domain | 7405 | 0.9058 | 0.9109 | +0.0051 [+0.0028, +0.0076] | WITHIN | 0.6846 | +0.0088 | +0.0115 |
| 2wiki | in-domain | 12576 | 0.9345 | 0.9390 | +0.0045 [+0.0030, +0.0061] | WITHIN | 0.6079 | +0.0143 | +0.0021 |
| webqsp | in-domain | 1503 | 0.6334 | 0.6423 | +0.0089 [-0.0038, +0.0215] | WITHIN | 0.0535 | +0.0106 | +0.0213 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
