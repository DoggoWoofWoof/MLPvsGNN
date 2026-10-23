# Screen scr-u2hf-base-zspk-s1-lsquad (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-u2hf-base-zspk-lsquad (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8019 | 0.8034 | +0.0015 [+0.0005, +0.0025] | WITHIN | 0.0047 | +0.0007 | +0.0084 |
| squad | zero-shot | 11873 | 0.8929 | 0.8919 | -0.0009 [-0.0032, +0.0015] | WITHIN | 0.9054 | -0.0009 | +0.0027 |
| musique | in-domain | 2417 | 0.5807 | 0.5826 | +0.0019 [-0.0033, +0.0077] | WITHIN | 0.4734 | +0.0033 | -0.0145 |
| hotpotqa | in-domain | 7405 | 0.9092 | 0.9099 | +0.0007 [-0.0014, +0.0029] | WITHIN | 0.6846 | +0.0012 | +0.0022 |
| 2wiki | in-domain | 12576 | 0.9385 | 0.9375 | -0.0011 [-0.0025, +0.0004] | WITHIN | 0.6079 | -0.0033 | +0.0043 |
| webqsp | zero-shot | 1503 | 0.2957 | 0.2991 | +0.0034 [-0.0058, +0.0132] | WITHIN | 0.0535 | +0.0027 | +0.0060 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
