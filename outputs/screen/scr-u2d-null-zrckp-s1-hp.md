# Screen scr-u2d-base-zrckp-s1-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-u2d-base-zrckp-hp (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8033 | 0.8024 | -0.0008 [-0.0019, +0.0002] | WITHIN | 0.0047 | -0.0002 | -0.0013 |
| squad | in-domain | 11873 | 0.9124 | 0.9123 | -0.0001 [-0.0020, +0.0019] | WITHIN | 0.9054 | -0.0001 | +0.0040 |
| musique | in-domain | 2417 | 0.5697 | 0.5781 | +0.0084 [+0.0027, +0.0141] | GAIN | 0.4734 | +0.0046 | +0.0141 |
| hotpotqa | zero-shot | 7405 | 0.8538 | 0.8502 | -0.0036 [-0.0068, -0.0006] | WITHIN | 0.6846 | -0.0065 | +0.0038 |
| 2wiki | in-domain | 12576 | 0.9303 | 0.9338 | +0.0036 [+0.0017, +0.0053] | WITHIN | 0.6079 | +0.0121 | -0.0002 |
| webqsp | zero-shot | 1503 | 0.3370 | 0.3212 | -0.0158 [-0.0262, -0.0047] | LOSS | 0.0535 | -0.0166 | -0.0053 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
