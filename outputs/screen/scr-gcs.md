# Screen scr-gcs (p@swa, s1eval carves)

## Against outputs\step1\fits\L-musique (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6537 | 0.6478 | -0.0060 [-0.0092, -0.0030] | WITHIN | 0.0047 | -0.0067 | -0.0038 |
| squad | in-domain | 11873 | 0.9100 | 0.9082 | -0.0018 [-0.0037, +0.0003] | WITHIN | 0.9054 | -0.0018 | -0.0016 |
| musique | zero-shot | 2417 | 0.2696 | 0.3132 | +0.0436 [+0.0356, +0.0516] | GAIN | 0.4734 | +0.0199 | +0.0257 |
| hotpotqa | in-domain | 7405 | 0.9015 | 0.8993 | -0.0022 [-0.0047, +0.0001] | WITHIN | 0.6846 | -0.0028 | +0.0022 |
| 2wiki | in-domain | 12576 | 0.8732 | 0.8715 | -0.0018 [-0.0036, +0.0002] | WITHIN | 0.6079 | -0.0038 | +0.0010 |
| webqsp | zero-shot | 1503 | 0.1779 | 0.1558 | -0.0220 [-0.0326, -0.0113] | LOSS | 0.0535 | -0.0193 | -0.0047 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
