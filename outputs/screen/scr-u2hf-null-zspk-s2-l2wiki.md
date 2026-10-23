# Screen scr-u2hf-base-zspk-s2-l2wiki (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-u2hf-base-zspk-l2wiki (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8030 | 0.8018 | -0.0012 [-0.0021, -0.0004] | WITHIN | 0.0047 | +0.0002 | -0.0010 |
| squad | in-domain | 11873 | 0.9112 | 0.9104 | -0.0008 [-0.0029, +0.0013] | WITHIN | 0.9054 | -0.0008 | +0.0024 |
| musique | in-domain | 2417 | 0.5708 | 0.5774 | +0.0066 [+0.0010, +0.0126] | WITHIN | 0.4734 | +0.0194 | -0.0046 |
| hotpotqa | in-domain | 7405 | 0.9032 | 0.9080 | +0.0049 [+0.0026, +0.0072] | WITHIN | 0.6846 | +0.0116 | +0.0061 |
| 2wiki | zero-shot | 12576 | 0.8600 | 0.8621 | +0.0021 [-0.0000, +0.0041] | WITHIN | 0.6079 | +0.0060 | -0.0095 |
| webqsp | zero-shot | 1503 | 0.2756 | 0.3011 | +0.0255 [+0.0167, +0.0350] | GAIN | 0.0535 | +0.0233 | -0.0013 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
