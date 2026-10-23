# Screen scr-u2hf-base-zrckp-s1-j5 (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-u2hf-probe-t4 (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8018 | 0.7984 | -0.0034 [-0.0046, -0.0022] | WITHIN | 0.0047 | -0.0034 | -0.0138 |
| squad | in-domain | 11873 | 0.9116 | 0.9096 | -0.0020 [-0.0040, -0.0003] | WITHIN | 0.9054 | -0.0020 | +0.0013 |
| musique | in-domain | 2417 | 0.5737 | 0.5760 | +0.0022 [-0.0027, +0.0073] | WITHIN | 0.4734 | +0.0037 | +0.0012 |
| hotpotqa | in-domain | 7405 | 0.9066 | 0.9077 | +0.0011 [-0.0010, +0.0032] | WITHIN | 0.6846 | +0.0020 | +0.0124 |
| 2wiki | in-domain | 12576 | 0.9357 | 0.9321 | -0.0036 [-0.0051, -0.0021] | WITHIN | 0.6079 | -0.0092 | +0.0037 |
| webqsp | zero-shot | 1503 | 0.3643 | 0.3553 | -0.0090 [-0.0184, +0.0006] | WITHIN | 0.0535 | -0.0027 | -0.0113 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
