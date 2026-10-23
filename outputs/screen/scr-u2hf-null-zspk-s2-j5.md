# Screen scr-u2hf-base-zspk-s2-j5 (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-u2hf-base-zspk-j5 (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8007 | 0.8016 | +0.0009 [-0.0001, +0.0019] | WITHIN | 0.0047 | +0.0011 | +0.0012 |
| squad | in-domain | 11873 | 0.9113 | 0.9096 | -0.0017 [-0.0037, +0.0004] | WITHIN | 0.9054 | -0.0017 | -0.0004 |
| musique | in-domain | 2417 | 0.5772 | 0.5841 | +0.0069 [+0.0020, +0.0116] | WITHIN | 0.4734 | +0.0112 | -0.0004 |
| hotpotqa | in-domain | 7405 | 0.9045 | 0.9070 | +0.0025 [+0.0005, +0.0045] | WITHIN | 0.6846 | +0.0057 | -0.0018 |
| 2wiki | in-domain | 12576 | 0.9358 | 0.9354 | -0.0004 [-0.0016, +0.0010] | WITHIN | 0.6079 | +0.0002 | -0.0078 |
| webqsp | zero-shot | 1503 | 0.2894 | 0.2848 | -0.0045 [-0.0127, +0.0039] | WITHIN | 0.0535 | -0.0047 | +0.0047 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
