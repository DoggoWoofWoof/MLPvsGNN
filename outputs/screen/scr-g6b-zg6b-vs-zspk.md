# Screen scr-g6b-zg6b (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-kg-base-zspk (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8013 | 0.8126 | +0.0114 [+0.0097, +0.0130] | GAIN | 0.0047 | +0.0134 | +0.0376 |
| squad | in-domain | 11873 | 0.9102 | 0.9113 | +0.0011 [-0.0012, +0.0033] | WITHIN | 0.9054 | +0.0011 | +0.0017 |
| musique | zero-shot | 2417 | 0.5146 | 0.5052 | -0.0094 [-0.0166, -0.0022] | LOSS | 0.4734 | +0.0017 | -0.0199 |
| hotpotqa | in-domain | 7405 | 0.9086 | 0.9128 | +0.0042 [+0.0019, +0.0065] | WITHIN | 0.6846 | +0.0080 | +0.0178 |
| 2wiki | in-domain | 12576 | 0.9364 | 0.9424 | +0.0060 [+0.0044, +0.0075] | WITHIN | 0.6079 | +0.0191 | +0.0086 |
| webqsp | zero-shot | 1503 | 0.3151 | 0.2784 | -0.0367 [-0.0571, -0.0172] | LOSS | 0.0535 | -0.0293 | -0.0093 |

## Against outputs\screen\fits\scr-kg-base-zg6 (reported only): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8127 | 0.8126 | -0.0000 [-0.0009, +0.0009] | WITHIN | 0.0047 | +0.0001 | -0.0020 |
| squad | in-domain | 11873 | 0.9120 | 0.9113 | -0.0007 [-0.0025, +0.0011] | WITHIN | 0.9054 | -0.0007 | -0.0012 |
| musique | zero-shot | 2417 | 0.5138 | 0.5052 | -0.0085 [-0.0132, -0.0038] | LOSS | 0.4734 | -0.0062 | -0.0319 |
| hotpotqa | in-domain | 7405 | 0.9127 | 0.9128 | +0.0001 [-0.0020, +0.0022] | WITHIN | 0.6846 | +0.0011 | +0.0036 |
| 2wiki | in-domain | 12576 | 0.9448 | 0.9424 | -0.0024 [-0.0037, -0.0011] | WITHIN | 0.6079 | -0.0054 | -0.0041 |
| webqsp | zero-shot | 1503 | 0.2398 | 0.2784 | +0.0386 [+0.0232, +0.0539] | GAIN | 0.0535 | +0.0286 | +0.0326 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
