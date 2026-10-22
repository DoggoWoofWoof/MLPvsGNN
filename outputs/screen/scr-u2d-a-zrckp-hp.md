# Screen scr-u2d-new-zrckp-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-u2d-base-zrckp-hp (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8033 | 0.8009 | -0.0023 [-0.0039, -0.0009] | WITHIN | 0.0047 | -0.0025 | -0.0089 |
| squad | in-domain | 11873 | 0.9124 | 0.9131 | +0.0007 [-0.0015, +0.0029] | WITHIN | 0.9054 | +0.0007 | +0.0000 |
| musique | in-domain | 2417 | 0.5697 | 0.5741 | +0.0044 [-0.0020, +0.0109] | WITHIN | 0.4734 | +0.0074 | +0.0033 |
| hotpotqa | zero-shot | 7405 | 0.8538 | 0.8064 | -0.0474 [-0.0521, -0.0428] | LOSS | 0.6846 | -0.0926 | +0.0197 |
| 2wiki | in-domain | 12576 | 0.9303 | 0.9029 | -0.0273 [-0.0304, -0.0245] | LOSS | 0.6079 | -0.0680 | +0.0071 |
| webqsp | zero-shot | 1503 | 0.3370 | 0.3870 | +0.0500 [+0.0362, +0.0644] | GAIN | 0.0535 | +0.0373 | +0.0353 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
