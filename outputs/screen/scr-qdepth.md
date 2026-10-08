# Screen scr-qdepth (p@swa, s1eval carves)

## Against outputs\step1\fits\L-musique (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6537 | 0.6523 | -0.0014 [-0.0040, +0.0011] | WITHIN | 0.0047 | -0.0018 | +0.0065 |
| squad | in-domain | 11873 | 0.9100 | 0.9100 | +0.0001 [-0.0022, +0.0023] | WITHIN | 0.9054 | +0.0001 | -0.0032 |
| musique | zero-shot | 2417 | 0.2696 | 0.3802 | +0.1106 [+0.1009, +0.1205] | GAIN | 0.4734 | +0.0625 | +0.1121 |
| hotpotqa | in-domain | 7405 | 0.9015 | 0.8976 | -0.0038 [-0.0063, -0.0016] | WITHIN | 0.6846 | -0.0070 | +0.0032 |
| 2wiki | in-domain | 12576 | 0.8732 | 0.8678 | -0.0054 [-0.0073, -0.0035] | WITHIN | 0.6079 | -0.0144 | -0.0032 |
| webqsp | zero-shot | 1503 | 0.1779 | 0.1575 | -0.0204 [-0.0310, -0.0094] | LOSS | 0.0535 | -0.0173 | -0.0086 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
