# Screen scr-u2g-y-zg6-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-u2d-base-zg6-hp (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8112 | 0.8117 | +0.0006 [-0.0003, +0.0016] | WITHIN | 0.0047 | +0.0010 | -0.0004 |
| squad | in-domain | 11873 | 0.9118 | 0.9132 | +0.0013 [-0.0008, +0.0035] | WITHIN | 0.9054 | +0.0013 | +0.0045 |
| musique | in-domain | 2417 | 0.5937 | 0.5984 | +0.0047 [-0.0008, +0.0100] | WITHIN | 0.4734 | +0.0025 | -0.0095 |
| hotpotqa | zero-shot | 7405 | 0.8309 | 0.7847 | -0.0463 [-0.0510, -0.0416] | LOSS | 0.6846 | -0.0913 | -0.0039 |
| 2wiki | in-domain | 12576 | 0.9391 | 0.9271 | -0.0121 [-0.0148, -0.0093] | LOSS | 0.6079 | -0.0278 | +0.0020 |
| webqsp | zero-shot | 1503 | 0.2189 | 0.2191 | +0.0002 [-0.0121, +0.0139] | WITHIN | 0.0535 | -0.0080 | +0.0040 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
