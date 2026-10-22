# Screen scr-u2d-new-zrckp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-u2d-base-zrckp (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8008 | 0.7999 | -0.0009 [-0.0024, +0.0006] | WITHIN | 0.0047 | -0.0001 | -0.0042 |
| squad | in-domain | 11873 | 0.9121 | 0.9129 | +0.0008 [-0.0014, +0.0030] | WITHIN | 0.9054 | +0.0008 | +0.0028 |
| musique | zero-shot | 2417 | 0.5153 | 0.5189 | +0.0036 [-0.0037, +0.0104] | WITHIN | 0.4734 | +0.0095 | +0.0079 |
| hotpotqa | in-domain | 7405 | 0.9055 | 0.8484 | -0.0571 [-0.0617, -0.0527] | LOSS | 0.6846 | -0.1113 | -0.0065 |
| 2wiki | in-domain | 12576 | 0.9345 | 0.9204 | -0.0141 [-0.0169, -0.0112] | LOSS | 0.6079 | -0.0362 | +0.0045 |
| webqsp | zero-shot | 1503 | 0.3561 | 0.3951 | +0.0390 [+0.0260, +0.0523] | GAIN | 0.0535 | +0.0353 | +0.0299 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
