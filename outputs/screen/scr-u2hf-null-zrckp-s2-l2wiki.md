# Screen scr-u2hf-base-zrckp-s2-l2wiki (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-u2hf-base-zrckp-l2wiki (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8029 | 0.8021 | -0.0008 [-0.0017, +0.0001] | WITHIN | 0.0047 | -0.0007 | -0.0050 |
| squad | in-domain | 11873 | 0.9099 | 0.9096 | -0.0003 [-0.0022, +0.0018] | WITHIN | 0.9054 | -0.0003 | +0.0038 |
| musique | in-domain | 2417 | 0.5678 | 0.5782 | +0.0104 [+0.0052, +0.0156] | GAIN | 0.4734 | +0.0095 | +0.0083 |
| hotpotqa | in-domain | 7405 | 0.9014 | 0.9074 | +0.0061 [+0.0037, +0.0086] | WITHIN | 0.6846 | +0.0120 | -0.0061 |
| 2wiki | zero-shot | 12576 | 0.8519 | 0.8600 | +0.0081 [+0.0059, +0.0104] | GAIN | 0.6079 | +0.0132 | -0.0072 |
| webqsp | zero-shot | 1503 | 0.3580 | 0.3489 | -0.0091 [-0.0185, +0.0003] | WITHIN | 0.0535 | -0.0040 | +0.0027 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
