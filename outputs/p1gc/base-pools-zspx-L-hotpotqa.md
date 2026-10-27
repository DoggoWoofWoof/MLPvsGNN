# Screen scr-g5-zspx-hp (p@swa, s1eval carves)

## Against outputs\step1\fits\L-hotpotqa (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6540 | 0.8039 | +0.1500 [+0.1443, +0.1554] | GAIN | 0.0047 | +0.1548 | +0.2925 |
| squad | in-domain | 11873 | 0.9116 | 0.9129 | +0.0013 [-0.0009, +0.0035] | WITHIN | 0.9054 | +0.0013 | +0.0019 |
| musique | in-domain | 2417 | 0.5550 | 0.5901 | +0.0351 [+0.0276, +0.0428] | GAIN | 0.4734 | +0.0422 | +0.0550 |
| hotpotqa | zero-shot | 7405 | 0.8585 | 0.8547 | -0.0038 [-0.0088, +0.0009] | WITHIN | 0.6846 | -0.0066 | -0.0238 |
| 2wiki | in-domain | 12576 | 0.8672 | 0.9312 | +0.0640 [+0.0607, +0.0677] | GAIN | 0.6079 | +0.1311 | +0.0747 |
| webqsp | zero-shot | 1503 | 0.1200 | 0.2914 | +0.1714 [+0.1472, +0.1963] | GAIN | 0.0535 | +0.1271 | +0.0912 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
