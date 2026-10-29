# Screen scr-r2ws-zrckpb-s1 (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-wqk-zrckp-s1 (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | zero-shot | 9785 | 0.4189 | 0.4455 | +0.0266 [+0.0211, +0.0318] | GAIN | 0.0047 | +0.0105 | +0.0358 |
| squad | in-domain | 11873 | 0.9091 | 0.9087 | -0.0004 [-0.0027, +0.0018] | WITHIN | 0.9054 | -0.0004 | -0.0046 |
| musique | zero-shot | 2417 | 0.4637 | 0.5010 | +0.0373 [+0.0296, +0.0452] | GAIN | 0.4734 | +0.0252 | +0.0273 |
| hotpotqa | in-domain | 7405 | 0.8998 | 0.9013 | +0.0015 [-0.0013, +0.0043] | WITHIN | 0.6846 | +0.0028 | +0.0011 |
| 2wiki | in-domain | 12576 | 0.9281 | 0.9317 | +0.0036 [+0.0020, +0.0052] | WITHIN | 0.6079 | +0.0106 | -0.0130 |
| webqsp | in-domain | 1503 | 0.6421 | 0.6555 | +0.0134 [+0.0018, +0.0247] | GAIN | 0.0535 | +0.0086 | +0.0180 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
