# Screen scr-vshare-hp (p@swa, s1eval carves)

## Against outputs\step1\fits\L-hotpotqa (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6540 | 0.6484 | -0.0055 [-0.0087, -0.0024] | WITHIN | 0.0047 | -0.0016 | -0.0079 |
| squad | in-domain | 11873 | 0.9116 | 0.9119 | +0.0003 [-0.0017, +0.0024] | WITHIN | 0.9054 | +0.0003 | +0.0008 |
| musique | in-domain | 2417 | 0.5550 | 0.5581 | +0.0030 [-0.0027, +0.0091] | WITHIN | 0.4734 | +0.0137 | -0.0062 |
| hotpotqa | zero-shot | 7405 | 0.8585 | 0.8445 | -0.0140 [-0.0179, -0.0101] | LOSS | 0.6846 | -0.0285 | -0.0039 |
| 2wiki | in-domain | 12576 | 0.8672 | 0.8660 | -0.0013 [-0.0035, +0.0010] | WITHIN | 0.6079 | +0.0011 | -0.0022 |
| webqsp | zero-shot | 1503 | 0.1200 | 0.1143 | -0.0057 [-0.0144, +0.0027] | WITHIN | 0.0535 | -0.0067 | -0.0053 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
