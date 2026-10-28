# Screen scr-psmh-zrckph (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-wqk-zrckp-s0 (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | zero-shot | 9785 | 0.4393 | 0.4013 | -0.0380 [-0.0452, -0.0309] | LOSS | 0.0047 | -0.0363 | -0.0655 |
| squad | in-domain | 11873 | 0.9106 | 0.9094 | -0.0012 [-0.0032, +0.0007] | WITHIN | 0.9054 | -0.0012 | -0.0012 |
| musique | zero-shot | 2417 | 0.4972 | 0.5049 | +0.0077 [+0.0012, +0.0139] | GAIN | 0.4734 | +0.0074 | +0.0790 |
| hotpotqa | in-domain | 7405 | 0.9051 | 0.8989 | -0.0063 [-0.0088, -0.0038] | WITHIN | 0.6846 | -0.0124 | -0.0150 |
| 2wiki | in-domain | 12576 | 0.9315 | 0.9244 | -0.0072 [-0.0089, -0.0054] | WITHIN | 0.6079 | -0.0250 | +0.0038 |
| webqsp | in-domain | 1503 | 0.6433 | 0.6251 | -0.0182 [-0.0279, -0.0090] | LOSS | 0.0535 | -0.0240 | -0.0266 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
