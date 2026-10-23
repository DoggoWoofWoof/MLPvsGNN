# Screen scr-u2hf-base-zspk-s1-l2wiki (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-u2hf-base-zspk-l2wiki (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8030 | 0.8007 | -0.0024 [-0.0033, -0.0014] | WITHIN | 0.0047 | -0.0017 | -0.0088 |
| squad | in-domain | 11873 | 0.9112 | 0.9102 | -0.0010 [-0.0030, +0.0010] | WITHIN | 0.9054 | -0.0010 | +0.0023 |
| musique | in-domain | 2417 | 0.5708 | 0.5800 | +0.0091 [+0.0039, +0.0144] | GAIN | 0.4734 | +0.0153 | -0.0062 |
| hotpotqa | in-domain | 7405 | 0.9032 | 0.9101 | +0.0069 [+0.0045, +0.0095] | WITHIN | 0.6846 | +0.0131 | +0.0005 |
| 2wiki | zero-shot | 12576 | 0.8600 | 0.8568 | -0.0032 [-0.0055, -0.0011] | WITHIN | 0.6079 | -0.0015 | -0.0106 |
| webqsp | zero-shot | 1503 | 0.2756 | 0.3031 | +0.0275 [+0.0183, +0.0376] | GAIN | 0.0535 | +0.0240 | +0.0100 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
