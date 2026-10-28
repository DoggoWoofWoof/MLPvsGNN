# Screen scr-psmk-zrckpg-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-wqk-zrckp-s0-hp (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | zero-shot | 9785 | 0.3927 | 0.3953 | +0.0026 [-0.0038, +0.0091] | WITHIN | 0.0047 | +0.0032 | -0.0217 |
| squad | in-domain | 11873 | 0.9117 | 0.9068 | -0.0049 [-0.0072, -0.0026] | WITHIN | 0.9054 | -0.0049 | -0.0024 |
| musique | in-domain | 2417 | 0.5743 | 0.5568 | -0.0175 [-0.0238, -0.0112] | LOSS | 0.4734 | -0.0248 | -0.0128 |
| hotpotqa | zero-shot | 7405 | 0.8525 | 0.8379 | -0.0145 [-0.0184, -0.0106] | LOSS | 0.6846 | -0.0236 | -0.0224 |
| 2wiki | in-domain | 12576 | 0.9299 | 0.9098 | -0.0201 [-0.0223, -0.0179] | LOSS | 0.6079 | -0.0431 | +0.0082 |
| webqsp | in-domain | 1503 | 0.6382 | 0.6240 | -0.0142 [-0.0240, -0.0042] | LOSS | 0.0535 | -0.0220 | -0.0193 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
