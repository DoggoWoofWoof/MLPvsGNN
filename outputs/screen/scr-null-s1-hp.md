# Screen scr-null-s1-hp (p@swa, s1eval carves)

## Against outputs\step1\fits\L-hotpotqa (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6540 | 0.6448 | -0.0092 [-0.0126, -0.0058] | LOSS | 0.0047 | -0.0046 | -0.0198 |
| squad | in-domain | 11873 | 0.9116 | 0.9118 | +0.0002 [-0.0021, +0.0025] | WITHIN | 0.9054 | +0.0002 | -0.0003 |
| musique | in-domain | 2417 | 0.5550 | 0.5565 | +0.0015 [-0.0048, +0.0078] | WITHIN | 0.4734 | +0.0124 | -0.0083 |
| hotpotqa | zero-shot | 7405 | 0.8585 | 0.8531 | -0.0055 [-0.0093, -0.0018] | WITHIN | 0.6846 | -0.0109 | -0.0028 |
| 2wiki | in-domain | 12576 | 0.8672 | 0.8653 | -0.0020 [-0.0041, +0.0003] | WITHIN | 0.6079 | -0.0020 | -0.0068 |
| webqsp | zero-shot | 1503 | 0.1200 | 0.1302 | +0.0102 [+0.0004, +0.0200] | GAIN | 0.0535 | +0.0060 | -0.0040 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
