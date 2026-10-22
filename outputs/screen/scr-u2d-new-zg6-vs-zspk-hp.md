# Screen scr-u2d-new-zg6-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-u2d-new-zspk-hp (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7999 | 0.8134 | +0.0135 [+0.0117, +0.0153] | GAIN | 0.0047 | +0.0170 | +0.0469 |
| squad | in-domain | 11873 | 0.9159 | 0.9137 | -0.0023 [-0.0045, +0.0000] | WITHIN | 0.9054 | -0.0023 | +0.0070 |
| musique | in-domain | 2417 | 0.5765 | 0.5985 | +0.0219 [+0.0161, +0.0279] | GAIN | 0.4734 | +0.0285 | +0.0281 |
| hotpotqa | zero-shot | 7405 | 0.8018 | 0.7810 | -0.0208 [-0.0251, -0.0164] | LOSS | 0.6846 | -0.0261 | -0.0515 |
| 2wiki | in-domain | 12576 | 0.9095 | 0.9267 | +0.0172 [+0.0149, +0.0194] | GAIN | 0.6079 | +0.0511 | +0.0070 |
| webqsp | zero-shot | 1503 | 0.2942 | 0.1781 | -0.1161 [-0.1364, -0.0939] | LOSS | 0.0535 | -0.0892 | -0.0539 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
