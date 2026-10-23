# Screen scr-u2h-zspkw (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-u2g-y-zspk (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8023 | 0.8033 | +0.0009 [-0.0001, +0.0020] | WITHIN | 0.0047 | +0.0011 | +0.0000 |
| squad | in-domain | 11873 | 0.9110 | 0.9122 | +0.0013 [-0.0010, +0.0035] | WITHIN | 0.9054 | +0.0013 | -0.0043 |
| musique | zero-shot | 2417 | 0.5286 | 0.5202 | -0.0083 [-0.0153, -0.0014] | LOSS | 0.4734 | +0.0037 | -0.0244 |
| hotpotqa | in-domain | 7405 | 0.8394 | 0.9074 | +0.0679 [+0.0631, +0.0727] | GAIN | 0.6846 | +0.1344 | +0.0024 |
| 2wiki | in-domain | 12576 | 0.9072 | 0.9369 | +0.0297 [+0.0268, +0.0326] | GAIN | 0.6079 | +0.0772 | -0.0067 |
| webqsp | zero-shot | 1503 | 0.3009 | 0.3192 | +0.0183 [+0.0084, +0.0287] | GAIN | 0.0535 | +0.0133 | +0.0106 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
