# Screen scr-gsurg (p@swa, s1eval carves)

## Against outputs\step1\fits\L-musique (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6537 | 0.6549 | +0.0012 [-0.0016, +0.0041] | WITHIN | 0.0047 | -0.0004 | +0.0131 |
| squad | in-domain | 11873 | 0.9100 | 0.9089 | -0.0011 [-0.0031, +0.0009] | WITHIN | 0.9054 | -0.0011 | -0.0031 |
| musique | zero-shot | 2417 | 0.2696 | 0.3537 | +0.0842 [+0.0757, +0.0924] | GAIN | 0.4734 | +0.0381 | +0.0674 |
| hotpotqa | in-domain | 7405 | 0.9015 | 0.9018 | +0.0003 [-0.0021, +0.0026] | WITHIN | 0.6846 | -0.0003 | +0.0047 |
| 2wiki | in-domain | 12576 | 0.8732 | 0.8746 | +0.0014 [-0.0005, +0.0033] | WITHIN | 0.6079 | -0.0016 | +0.0013 |
| webqsp | zero-shot | 1503 | 0.1779 | 0.1771 | -0.0007 [-0.0093, +0.0079] | WITHIN | 0.0535 | -0.0007 | -0.0086 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
