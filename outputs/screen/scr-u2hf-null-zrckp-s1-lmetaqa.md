# Screen scr-u2hf-base-zrckp-s1-lmetaqa (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-u2hf-base-zrckp-lmetaqa (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | zero-shot | 9785 | 0.2463 | 0.2561 | +0.0098 [+0.0052, +0.0145] | GAIN | 0.0047 | +0.0128 | -0.0027 |
| squad | in-domain | 11873 | 0.9095 | 0.9122 | +0.0027 [+0.0005, +0.0050] | WITHIN | 0.9054 | +0.0027 | +0.0005 |
| musique | in-domain | 2417 | 0.5757 | 0.5749 | -0.0009 [-0.0064, +0.0047] | WITHIN | 0.4734 | +0.0000 | +0.0165 |
| hotpotqa | in-domain | 7405 | 0.9063 | 0.9080 | +0.0017 [-0.0006, +0.0041] | WITHIN | 0.6846 | +0.0035 | +0.0012 |
| 2wiki | in-domain | 12576 | 0.9286 | 0.9347 | +0.0061 [+0.0046, +0.0077] | WITHIN | 0.6079 | +0.0150 | -0.0037 |
| webqsp | zero-shot | 1503 | 0.1152 | 0.1163 | +0.0010 [-0.0062, +0.0087] | WITHIN | 0.0535 | -0.0013 | -0.0047 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
