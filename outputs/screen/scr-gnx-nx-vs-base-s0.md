# Screen scr-gnx-nx-s0 (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-ml1-zg6r-s0 (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | zero-shot | 9785 | 0.3615 | 0.3860 | +0.0245 [+0.0191, +0.0294] | GAIN | 0.0047 | +0.0226 | +0.0028 |
| squad | in-domain | 11873 | 0.9142 | 0.9145 | +0.0003 [-0.0015, +0.0022] | WITHIN | 0.9054 | +0.0003 | -0.0003 |
| musique | zero-shot | 2417 | 0.5086 | 0.5233 | +0.0148 [+0.0091, +0.0204] | GAIN | 0.4734 | +0.0149 | -0.0046 |
| hotpotqa | in-domain | 7405 | 0.8593 | 0.8597 | +0.0004 [-0.0020, +0.0028] | WITHIN | 0.6846 | +0.0005 | -0.0095 |
| 2wiki | in-domain | 12576 | 0.9269 | 0.9284 | +0.0015 [-0.0002, +0.0032] | WITHIN | 0.6079 | +0.0017 | +0.0018 |
| webqsp | in-domain | 1503 | 0.6632 | 0.6637 | +0.0005 [-0.0087, +0.0096] | WITHIN | 0.0535 | +0.0000 | +0.0086 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
