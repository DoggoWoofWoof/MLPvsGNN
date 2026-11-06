# Screen scr-gnx-nx-s1-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-ml1-zg6r-s1-hp (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | zero-shot | 9785 | 0.3903 | 0.3809 | -0.0094 [-0.0149, -0.0042] | LOSS | 0.0047 | -0.0086 | -0.0015 |
| squad | in-domain | 11873 | 0.9140 | 0.9136 | -0.0004 [-0.0024, +0.0015] | WITHIN | 0.9054 | -0.0004 | -0.0006 |
| musique | in-domain | 2417 | 0.5988 | 0.5944 | -0.0044 [-0.0097, +0.0010] | WITHIN | 0.4734 | -0.0141 | -0.0025 |
| hotpotqa | zero-shot | 7405 | 0.7943 | 0.8021 | +0.0078 [+0.0044, +0.0115] | GAIN | 0.6846 | +0.0054 | +0.0251 |
| 2wiki | in-domain | 12576 | 0.9290 | 0.9223 | -0.0067 [-0.0087, -0.0049] | WITHIN | 0.6079 | -0.0192 | +0.0029 |
| webqsp | in-domain | 1503 | 0.6512 | 0.6576 | +0.0064 [-0.0037, +0.0162] | WITHIN | 0.0535 | +0.0093 | +0.0086 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
