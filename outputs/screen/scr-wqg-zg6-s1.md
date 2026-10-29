# Screen scr-wqg-zg6-s1 (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-wqk-zg6-s0 (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | zero-shot | 9785 | 0.4393 | 0.4239 | -0.0154 [-0.0206, -0.0099] | LOSS | 0.0047 | -0.0126 | -0.0419 |
| squad | in-domain | 11873 | 0.9097 | 0.9113 | +0.0016 [-0.0003, +0.0036] | WITHIN | 0.9054 | +0.0016 | +0.0017 |
| musique | zero-shot | 2417 | 0.5018 | 0.5169 | +0.0150 [+0.0090, +0.0211] | GAIN | 0.4734 | +0.0112 | +0.0021 |
| hotpotqa | in-domain | 7405 | 0.9109 | 0.9124 | +0.0015 [-0.0005, +0.0034] | WITHIN | 0.6846 | +0.0032 | +0.0019 |
| 2wiki | in-domain | 12576 | 0.9390 | 0.9401 | +0.0012 [-0.0003, +0.0027] | WITHIN | 0.6079 | +0.0009 | +0.0060 |
| webqsp | in-domain | 1503 | 0.6423 | 0.6411 | -0.0012 [-0.0128, +0.0095] | WITHIN | 0.0535 | -0.0073 | +0.0013 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
