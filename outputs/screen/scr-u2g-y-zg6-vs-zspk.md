# Screen scr-u2g-y-zg6 (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-u2g-y-zspk (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8023 | 0.8110 | +0.0087 [+0.0071, +0.0102] | GAIN | 0.0047 | +0.0106 | +0.0312 |
| squad | in-domain | 11873 | 0.9110 | 0.9117 | +0.0008 [-0.0013, +0.0029] | WITHIN | 0.9054 | +0.0008 | +0.0014 |
| musique | zero-shot | 2417 | 0.5286 | 0.5414 | +0.0128 [+0.0059, +0.0199] | GAIN | 0.4734 | +0.0248 | -0.0310 |
| hotpotqa | in-domain | 7405 | 0.8394 | 0.8571 | +0.0176 [+0.0144, +0.0210] | GAIN | 0.6846 | +0.0332 | +0.0076 |
| 2wiki | in-domain | 12576 | 0.9072 | 0.9266 | +0.0195 [+0.0174, +0.0216] | GAIN | 0.6079 | +0.0556 | +0.0018 |
| webqsp | zero-shot | 1503 | 0.3009 | 0.1961 | -0.1048 [-0.1261, -0.0838] | LOSS | 0.0535 | -0.0845 | -0.0379 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
