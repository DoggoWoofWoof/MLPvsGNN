# Screen scr-u2hf-base-zspk-s1-j5 (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-u2hf-base-zspk-j5 (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8007 | 0.7996 | -0.0011 [-0.0023, +0.0000] | WITHIN | 0.0047 | -0.0005 | +0.0006 |
| squad | in-domain | 11873 | 0.9113 | 0.9109 | -0.0004 [-0.0025, +0.0017] | WITHIN | 0.9054 | -0.0004 | -0.0006 |
| musique | in-domain | 2417 | 0.5772 | 0.5811 | +0.0039 [-0.0010, +0.0089] | WITHIN | 0.4734 | +0.0091 | +0.0046 |
| hotpotqa | in-domain | 7405 | 0.9045 | 0.9065 | +0.0020 [-0.0000, +0.0041] | WITHIN | 0.6846 | +0.0047 | +0.0007 |
| 2wiki | in-domain | 12576 | 0.9358 | 0.9360 | +0.0002 [-0.0012, +0.0017] | WITHIN | 0.6079 | +0.0032 | -0.0002 |
| webqsp | zero-shot | 1503 | 0.2894 | 0.2803 | -0.0091 [-0.0177, -0.0015] | LOSS | 0.0535 | -0.0060 | -0.0027 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
