# Screen scr-qdepth-hp (p@swa, s1eval carves)

## Against outputs\step1\fits\L-hotpotqa (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6540 | 0.6506 | -0.0034 [-0.0065, -0.0007] | WITHIN | 0.0047 | -0.0008 | +0.0022 |
| squad | in-domain | 11873 | 0.9116 | 0.9130 | +0.0013 [-0.0008, +0.0035] | WITHIN | 0.9054 | +0.0013 | +0.0024 |
| musique | in-domain | 2417 | 0.5550 | 0.5585 | +0.0035 [-0.0022, +0.0094] | WITHIN | 0.4734 | +0.0116 | -0.0029 |
| hotpotqa | zero-shot | 7405 | 0.8585 | 0.8460 | -0.0125 [-0.0161, -0.0090] | LOSS | 0.6846 | -0.0231 | -0.0055 |
| 2wiki | in-domain | 12576 | 0.8672 | 0.8707 | +0.0035 [+0.0014, +0.0057] | WITHIN | 0.6079 | +0.0121 | -0.0033 |
| webqsp | zero-shot | 1503 | 0.1200 | 0.1165 | -0.0035 [-0.0127, +0.0060] | WITHIN | 0.0535 | -0.0047 | -0.0100 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
