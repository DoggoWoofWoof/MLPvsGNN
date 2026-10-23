# Screen scr-u2hf-zrckpw-j5 (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-u2hf-probe-t4 (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8018 | 0.7970 | -0.0048 [-0.0059, -0.0036] | WITHIN | 0.0047 | -0.0052 | -0.0208 |
| squad | in-domain | 11873 | 0.9116 | 0.9105 | -0.0012 [-0.0030, +0.0007] | WITHIN | 0.9054 | -0.0012 | +0.0010 |
| musique | in-domain | 2417 | 0.5737 | 0.5743 | +0.0006 [-0.0048, +0.0059] | WITHIN | 0.4734 | +0.0033 | +0.0137 |
| hotpotqa | in-domain | 7405 | 0.9066 | 0.9038 | -0.0028 [-0.0051, -0.0006] | WITHIN | 0.6846 | -0.0063 | +0.0134 |
| 2wiki | in-domain | 12576 | 0.9357 | 0.9364 | +0.0007 [-0.0008, +0.0022] | WITHIN | 0.6079 | -0.0002 | +0.0026 |
| webqsp | zero-shot | 1503 | 0.3643 | 0.3512 | -0.0131 [-0.0214, -0.0046] | LOSS | 0.0535 | -0.0093 | -0.0073 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
