# Screen scr-null-s2-hp (p@swa, s1eval carves)

## Against outputs\step1\fits\L-hotpotqa (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6540 | 0.6489 | -0.0051 [-0.0082, -0.0017] | WITHIN | 0.0047 | -0.0038 | +0.0024 |
| squad | in-domain | 11873 | 0.9116 | 0.9123 | +0.0007 [-0.0016, +0.0029] | WITHIN | 0.9054 | +0.0007 | +0.0024 |
| musique | in-domain | 2417 | 0.5550 | 0.5573 | +0.0023 [-0.0039, +0.0087] | WITHIN | 0.4734 | +0.0153 | +0.0000 |
| hotpotqa | zero-shot | 7405 | 0.8585 | 0.8455 | -0.0130 [-0.0165, -0.0094] | LOSS | 0.6846 | -0.0247 | -0.0046 |
| 2wiki | in-domain | 12576 | 0.8672 | 0.8683 | +0.0011 [-0.0009, +0.0033] | WITHIN | 0.6079 | +0.0042 | -0.0024 |
| webqsp | zero-shot | 1503 | 0.1200 | 0.1329 | +0.0129 [+0.0035, +0.0231] | GAIN | 0.0535 | +0.0080 | -0.0027 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
