# Screen scr-psms-zrckph1-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-wqk-zrckp-s0-hp (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | zero-shot | 9785 | 0.3927 | 0.4104 | +0.0177 [+0.0128, +0.0230] | GAIN | 0.0047 | +0.0145 | +0.0069 |
| squad | in-domain | 11873 | 0.9117 | 0.9111 | -0.0006 [-0.0024, +0.0013] | WITHIN | 0.9054 | -0.0006 | -0.0012 |
| musique | in-domain | 2417 | 0.5743 | 0.5733 | -0.0010 [-0.0059, +0.0038] | WITHIN | 0.4734 | -0.0021 | +0.0025 |
| hotpotqa | zero-shot | 7405 | 0.8525 | 0.8519 | -0.0005 [-0.0036, +0.0025] | WITHIN | 0.6846 | -0.0005 | -0.0178 |
| 2wiki | in-domain | 12576 | 0.9299 | 0.9303 | +0.0004 [-0.0012, +0.0019] | WITHIN | 0.6079 | +0.0008 | -0.0078 |
| webqsp | in-domain | 1503 | 0.6382 | 0.6330 | -0.0053 [-0.0127, +0.0017] | WITHIN | 0.0535 | -0.0106 | -0.0153 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
