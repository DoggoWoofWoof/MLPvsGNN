# Screen scr-u2d-new-zg6 (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-u2d-base-zg6 (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8123 | 0.8134 | +0.0011 [-0.0002, +0.0023] | WITHIN | 0.0047 | +0.0016 | +0.0017 |
| squad | in-domain | 11873 | 0.9106 | 0.9141 | +0.0035 [+0.0013, +0.0057] | WITHIN | 0.9054 | +0.0035 | +0.0016 |
| musique | zero-shot | 2417 | 0.5207 | 0.5426 | +0.0220 [+0.0153, +0.0287] | GAIN | 0.4734 | +0.0145 | +0.0182 |
| hotpotqa | in-domain | 7405 | 0.9118 | 0.8604 | -0.0514 [-0.0559, -0.0467] | LOSS | 0.6846 | -0.1067 | +0.0049 |
| 2wiki | in-domain | 12576 | 0.9416 | 0.9306 | -0.0110 [-0.0137, -0.0082] | LOSS | 0.6079 | -0.0285 | +0.0045 |
| webqsp | zero-shot | 1503 | 0.2368 | 0.1560 | -0.0808 [-0.0980, -0.0632] | LOSS | 0.0535 | -0.0632 | -0.0373 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
