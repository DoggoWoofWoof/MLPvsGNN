# Screen L-metaqa (p@swa, s1eval carves)

## Against outputs\full_zspg\fits\L-metaqa (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | zero-shot | 9785 | 0.1707 | 0.2128 | +0.0422 [+0.0372, +0.0480] | GAIN | 0.0047 | +0.0252 | -0.0047 |
| squad | in-domain | 11873 | 0.9116 | 0.9088 | -0.0029 [-0.0054, -0.0003] | WITHIN | 0.9054 | -0.0029 | -0.0009 |
| musique | in-domain | 2417 | 0.5672 | 0.5745 | +0.0073 [-0.0001, +0.0148] | WITHIN | 0.4734 | -0.0037 | +0.0186 |
| hotpotqa | in-domain | 7405 | 0.9080 | 0.9090 | +0.0010 [-0.0023, +0.0043] | WITHIN | 0.6846 | +0.0000 | +0.0054 |
| 2wiki | in-domain | 12576 | 0.8777 | 0.9340 | +0.0563 [+0.0531, +0.0596] | GAIN | 0.6079 | +0.1076 | +0.0677 |
| webqsp | zero-shot | 1503 | 0.1141 | 0.1162 | +0.0021 [-0.0069, +0.0111] | WITHIN | 0.0535 | +0.0000 | +0.0060 |

## Against outputs\step1\fits\L-metaqa (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | zero-shot | 9785 | 0.0773 | 0.2128 | +0.1355 [+0.1289, +0.1425] | GAIN | 0.0047 | +0.1099 | +0.0056 |
| squad | in-domain | 11873 | 0.9076 | 0.9088 | +0.0012 [-0.0013, +0.0036] | WITHIN | 0.9054 | +0.0012 | -0.0007 |
| musique | in-domain | 2417 | 0.5614 | 0.5745 | +0.0131 [+0.0057, +0.0205] | GAIN | 0.4734 | +0.0033 | +0.0331 |
| hotpotqa | in-domain | 7405 | 0.9046 | 0.9090 | +0.0044 [+0.0008, +0.0079] | WITHIN | 0.6846 | +0.0059 | +0.0124 |
| 2wiki | in-domain | 12576 | 0.8639 | 0.9340 | +0.0701 [+0.0669, +0.0735] | GAIN | 0.6079 | +0.1350 | +0.0761 |
| webqsp | zero-shot | 1503 | 0.0926 | 0.1162 | +0.0236 [+0.0150, +0.0330] | GAIN | 0.0535 | +0.0206 | +0.0173 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
