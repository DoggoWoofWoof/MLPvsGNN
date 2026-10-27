# Screen scr-kg-base-zspk-s2-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-kg-base-zspk-hp (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8037 | 0.8026 | -0.0011 [-0.0021, -0.0001] | WITHIN | 0.0047 | -0.0005 | -0.0058 |
| squad | in-domain | 11873 | 0.9139 | 0.9138 | -0.0001 [-0.0021, +0.0019] | WITHIN | 0.9054 | -0.0001 | +0.0006 |
| musique | in-domain | 2417 | 0.5754 | 0.5828 | +0.0074 [+0.0019, +0.0131] | WITHIN | 0.4734 | +0.0120 | -0.0124 |
| hotpotqa | zero-shot | 7405 | 0.8550 | 0.8551 | +0.0001 [-0.0028, +0.0032] | WITHIN | 0.6846 | +0.0016 | -0.0032 |
| 2wiki | in-domain | 12576 | 0.9265 | 0.9370 | +0.0105 [+0.0087, +0.0122] | GAIN | 0.6079 | +0.0332 | -0.0039 |
| webqsp | zero-shot | 1503 | 0.2640 | 0.2681 | +0.0041 [-0.0061, +0.0137] | WITHIN | 0.0535 | +0.0053 | -0.0007 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
