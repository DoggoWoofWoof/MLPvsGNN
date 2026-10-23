# Screen scr-u2hf-base-zrckp-s1-l2wiki (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-u2hf-base-zrckp-l2wiki (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8029 | 0.8025 | -0.0004 [-0.0013, +0.0006] | WITHIN | 0.0047 | -0.0002 | -0.0027 |
| squad | in-domain | 11873 | 0.9099 | 0.9101 | +0.0003 [-0.0018, +0.0022] | WITHIN | 0.9054 | +0.0003 | +0.0014 |
| musique | in-domain | 2417 | 0.5678 | 0.5788 | +0.0110 [+0.0056, +0.0163] | GAIN | 0.4734 | +0.0103 | +0.0058 |
| hotpotqa | in-domain | 7405 | 0.9014 | 0.9086 | +0.0073 [+0.0048, +0.0099] | WITHIN | 0.6846 | +0.0139 | -0.0031 |
| 2wiki | zero-shot | 12576 | 0.8519 | 0.8616 | +0.0097 [+0.0075, +0.0120] | GAIN | 0.6079 | +0.0176 | -0.0103 |
| webqsp | zero-shot | 1503 | 0.3580 | 0.3474 | -0.0106 [-0.0197, -0.0007] | LOSS | 0.0535 | -0.0020 | -0.0073 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
