# Screen scr-psmu-zrckpu-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-wqk-zrckp-s0-hp (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | zero-shot | 9785 | 0.3927 | 0.4037 | +0.0111 [+0.0053, +0.0164] | GAIN | 0.0047 | +0.0079 | -0.0054 |
| squad | in-domain | 11873 | 0.9117 | 0.9132 | +0.0014 [-0.0006, +0.0035] | WITHIN | 0.9054 | +0.0014 | -0.0005 |
| musique | in-domain | 2417 | 0.5743 | 0.5742 | -0.0001 [-0.0049, +0.0048] | WITHIN | 0.4734 | +0.0008 | -0.0004 |
| hotpotqa | zero-shot | 7405 | 0.8525 | 0.8560 | +0.0035 [+0.0004, +0.0068] | WITHIN | 0.6846 | +0.0070 | -0.0120 |
| 2wiki | in-domain | 12576 | 0.9299 | 0.9278 | -0.0021 [-0.0038, -0.0005] | WITHIN | 0.6079 | -0.0064 | -0.0023 |
| webqsp | in-domain | 1503 | 0.6382 | 0.6281 | -0.0101 [-0.0184, -0.0020] | LOSS | 0.0535 | -0.0140 | -0.0240 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
