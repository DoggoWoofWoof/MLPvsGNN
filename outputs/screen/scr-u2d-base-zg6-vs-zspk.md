# Screen scr-u2d-base-zg6 (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-u2d-base-zspk (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8021 | 0.8123 | +0.0102 [+0.0085, +0.0118] | GAIN | 0.0047 | +0.0118 | +0.0355 |
| squad | in-domain | 11873 | 0.9118 | 0.9106 | -0.0012 [-0.0033, +0.0010] | WITHIN | 0.9054 | -0.0012 | +0.0061 |
| musique | zero-shot | 2417 | 0.5143 | 0.5207 | +0.0063 [-0.0010, +0.0131] | WITHIN | 0.4734 | +0.0120 | +0.0116 |
| hotpotqa | in-domain | 7405 | 0.9088 | 0.9118 | +0.0030 [+0.0005, +0.0053] | WITHIN | 0.6846 | +0.0076 | +0.0059 |
| 2wiki | in-domain | 12576 | 0.9346 | 0.9416 | +0.0070 [+0.0055, +0.0087] | WITHIN | 0.6079 | +0.0192 | +0.0024 |
| webqsp | zero-shot | 1503 | 0.3162 | 0.2368 | -0.0795 [-0.1003, -0.0596] | LOSS | 0.0535 | -0.0599 | -0.0519 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
