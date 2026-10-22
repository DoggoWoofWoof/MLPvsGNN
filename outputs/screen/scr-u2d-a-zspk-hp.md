# Screen scr-u2d-new-zspk-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-u2d-base-zspk-hp (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8043 | 0.7999 | -0.0044 [-0.0060, -0.0029] | WITHIN | 0.0047 | -0.0050 | -0.0173 |
| squad | in-domain | 11873 | 0.9117 | 0.9159 | +0.0042 [+0.0019, +0.0067] | WITHIN | 0.9054 | +0.0042 | -0.0007 |
| musique | in-domain | 2417 | 0.5800 | 0.5765 | -0.0034 [-0.0095, +0.0029] | WITHIN | 0.4734 | -0.0017 | +0.0037 |
| hotpotqa | zero-shot | 7405 | 0.8512 | 0.8018 | -0.0494 [-0.0541, -0.0449] | LOSS | 0.6846 | -0.0972 | +0.0086 |
| 2wiki | in-domain | 12576 | 0.9290 | 0.9095 | -0.0195 [-0.0224, -0.0164] | LOSS | 0.6079 | -0.0519 | +0.0056 |
| webqsp | zero-shot | 1503 | 0.2773 | 0.2942 | +0.0169 [+0.0034, +0.0298] | GAIN | 0.0535 | +0.0146 | +0.0153 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
