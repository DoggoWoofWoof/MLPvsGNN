# Screen scr-u2hf-base-zrckp-s2-j5 (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-u2hf-probe-t4 (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8018 | 0.8018 | +0.0001 [-0.0012, +0.0013] | WITHIN | 0.0047 | -0.0001 | -0.0027 |
| squad | in-domain | 11873 | 0.9116 | 0.9116 | -0.0001 [-0.0019, +0.0018] | WITHIN | 0.9054 | -0.0001 | +0.0018 |
| musique | in-domain | 2417 | 0.5737 | 0.5787 | +0.0050 [+0.0003, +0.0097] | WITHIN | 0.4734 | +0.0128 | -0.0037 |
| hotpotqa | in-domain | 7405 | 0.9066 | 0.9065 | -0.0001 [-0.0021, +0.0019] | WITHIN | 0.6846 | -0.0004 | +0.0134 |
| 2wiki | in-domain | 12576 | 0.9357 | 0.9357 | -0.0000 [-0.0015, +0.0014] | WITHIN | 0.6079 | +0.0003 | +0.0011 |
| webqsp | zero-shot | 1503 | 0.3643 | 0.3713 | +0.0070 [-0.0024, +0.0164] | WITHIN | 0.0535 | +0.0086 | +0.0040 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
