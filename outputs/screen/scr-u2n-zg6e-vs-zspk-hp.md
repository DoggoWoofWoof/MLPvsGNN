# Screen scr-u2n-zg6e-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-kg-base-zspk-hp (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8037 | 0.8113 | +0.0076 [+0.0060, +0.0091] | GAIN | 0.0047 | +0.0093 | +0.0299 |
| squad | in-domain | 11873 | 0.9139 | 0.9106 | -0.0033 [-0.0055, -0.0011] | WITHIN | 0.9054 | -0.0033 | +0.0027 |
| musique | in-domain | 2417 | 0.5754 | 0.5985 | +0.0231 [+0.0169, +0.0291] | GAIN | 0.4734 | +0.0343 | +0.0186 |
| hotpotqa | zero-shot | 7405 | 0.8550 | 0.8238 | -0.0311 [-0.0352, -0.0267] | LOSS | 0.6846 | -0.0373 | -0.0435 |
| 2wiki | in-domain | 12576 | 0.9265 | 0.9437 | +0.0172 [+0.0150, +0.0193] | GAIN | 0.6079 | +0.0567 | +0.0017 |
| webqsp | zero-shot | 1503 | 0.2640 | 0.2254 | -0.0386 [-0.0603, -0.0170] | LOSS | 0.0535 | -0.0326 | +0.0033 |

## Against outputs\screen\fits\scr-kg-base-zg6-hp (reported only): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8115 | 0.8113 | -0.0002 [-0.0011, +0.0007] | WITHIN | 0.0047 | +0.0001 | +0.0025 |
| squad | in-domain | 11873 | 0.9122 | 0.9106 | -0.0016 [-0.0035, +0.0003] | WITHIN | 0.9054 | -0.0016 | -0.0026 |
| musique | in-domain | 2417 | 0.5953 | 0.5985 | +0.0032 [-0.0016, +0.0081] | WITHIN | 0.4734 | +0.0017 | +0.0029 |
| hotpotqa | zero-shot | 7405 | 0.8391 | 0.8238 | -0.0153 [-0.0184, -0.0120] | LOSS | 0.6846 | -0.0192 | -0.0105 |
| 2wiki | in-domain | 12576 | 0.9463 | 0.9437 | -0.0026 [-0.0041, -0.0012] | WITHIN | 0.6079 | -0.0049 | +0.0127 |
| webqsp | zero-shot | 1503 | 0.2430 | 0.2254 | -0.0176 [-0.0324, -0.0044] | LOSS | 0.0535 | -0.0180 | -0.0053 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
