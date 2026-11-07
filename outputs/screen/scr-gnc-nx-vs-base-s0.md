# Screen scr-gnc-nx-s0 (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-ml1-zg6r-s0 (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | zero-shot | 9785 | 0.3615 | 0.3874 | +0.0259 [+0.0211, +0.0306] | GAIN | 0.0047 | +0.0212 | +0.0374 |
| squad | in-domain | 11873 | 0.9142 | 0.9148 | +0.0006 [-0.0007, +0.0018] | WITHIN | 0.9054 | +0.0006 | -0.0032 |
| musique | zero-shot | 2417 | 0.5086 | 0.5134 | +0.0048 [-0.0001, +0.0099] | WITHIN | 0.4734 | +0.0029 | -0.0165 |
| hotpotqa | in-domain | 7405 | 0.8593 | 0.8586 | -0.0007 [-0.0024, +0.0010] | WITHIN | 0.6846 | -0.0019 | -0.0184 |
| 2wiki | in-domain | 12576 | 0.9269 | 0.9269 | -0.0000 [-0.0011, +0.0011] | WITHIN | 0.6079 | -0.0016 | -0.0055 |
| webqsp | in-domain | 1503 | 0.6632 | 0.6508 | -0.0125 [-0.0217, -0.0029] | LOSS | 0.0535 | -0.0153 | -0.0233 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
