# Screen scr-u2d-new-zg6-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-u2g-y-zg6-hp (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8117 | 0.8134 | +0.0016 [+0.0004, +0.0028] | WITHIN | 0.0047 | +0.0027 | +0.0049 |
| squad | in-domain | 11873 | 0.9132 | 0.9137 | +0.0005 [-0.0014, +0.0024] | WITHIN | 0.9054 | +0.0005 | -0.0016 |
| musique | in-domain | 2417 | 0.5984 | 0.5985 | +0.0001 [-0.0047, +0.0053] | WITHIN | 0.4734 | +0.0008 | +0.0021 |
| hotpotqa | zero-shot | 7405 | 0.7847 | 0.7810 | -0.0036 [-0.0069, -0.0003] | WITHIN | 0.6846 | -0.0099 | +0.0019 |
| 2wiki | in-domain | 12576 | 0.9271 | 0.9267 | -0.0004 [-0.0021, +0.0014] | WITHIN | 0.6079 | -0.0027 | +0.0015 |
| webqsp | zero-shot | 1503 | 0.2191 | 0.1781 | -0.0410 [-0.0570, -0.0248] | LOSS | 0.0535 | -0.0259 | -0.0359 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
