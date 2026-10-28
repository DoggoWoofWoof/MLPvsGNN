# Screen scr-psmu-zrckpu1-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-wqk-zrckp-s0-hp (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | zero-shot | 9785 | 0.3927 | 0.4342 | +0.0415 [+0.0363, +0.0467] | GAIN | 0.0047 | +0.0358 | +0.0487 |
| squad | in-domain | 11873 | 0.9117 | 0.9127 | +0.0010 [-0.0009, +0.0029] | WITHIN | 0.9054 | +0.0010 | -0.0002 |
| musique | in-domain | 2417 | 0.5743 | 0.5731 | -0.0012 [-0.0062, +0.0037] | WITHIN | 0.4734 | -0.0041 | -0.0050 |
| hotpotqa | zero-shot | 7405 | 0.8525 | 0.8568 | +0.0043 [+0.0014, +0.0074] | WITHIN | 0.6846 | +0.0069 | -0.0023 |
| 2wiki | in-domain | 12576 | 0.9299 | 0.9297 | -0.0003 [-0.0018, +0.0013] | WITHIN | 0.6079 | -0.0006 | +0.0009 |
| webqsp | in-domain | 1503 | 0.6382 | 0.6315 | -0.0067 [-0.0141, +0.0004] | WITHIN | 0.0535 | -0.0093 | +0.0007 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
