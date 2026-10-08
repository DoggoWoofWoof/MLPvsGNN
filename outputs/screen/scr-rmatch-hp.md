# Screen scr-rmatch-hp (p@swa, s1eval carves)

## Against outputs\step1\fits\L-hotpotqa (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6540 | 0.7790 | +0.1250 [+0.1198, +0.1300] | GAIN | 0.0047 | +0.1325 | +0.2525 |
| squad | in-domain | 11873 | 0.9116 | 0.9101 | -0.0015 [-0.0036, +0.0007] | WITHIN | 0.9054 | -0.0015 | -0.0006 |
| musique | in-domain | 2417 | 0.5550 | 0.5588 | +0.0038 [-0.0019, +0.0098] | WITHIN | 0.4734 | +0.0103 | +0.0037 |
| hotpotqa | zero-shot | 7405 | 0.8585 | 0.8513 | -0.0072 [-0.0111, -0.0034] | WITHIN | 0.6846 | -0.0132 | -0.0027 |
| 2wiki | in-domain | 12576 | 0.8672 | 0.8724 | +0.0051 [+0.0030, +0.0073] | WITHIN | 0.6079 | +0.0150 | -0.0012 |
| webqsp | zero-shot | 1503 | 0.1200 | 0.2646 | +0.1446 [+0.1217, +0.1669] | GAIN | 0.0535 | +0.1124 | +0.0705 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
