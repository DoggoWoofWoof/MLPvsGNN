# Screen scr-p5c5-zspko5-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-p1c3-zspk-hp (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8040 | 0.8060 | +0.0020 [+0.0009, +0.0031] | WITHIN | 0.0047 | +0.0029 | +0.0049 |
| squad | in-domain | 11873 | 0.9135 | 0.9143 | +0.0008 [-0.0012, +0.0025] | WITHIN | 0.9054 | +0.0008 | +0.0024 |
| musique | in-domain | 2417 | 0.5754 | 0.5807 | +0.0053 [-0.0004, +0.0109] | WITHIN | 0.4734 | +0.0025 | -0.0037 |
| hotpotqa | zero-shot | 7405 | 0.8499 | 0.8508 | +0.0009 [-0.0021, +0.0038] | WITHIN | 0.6846 | +0.0039 | -0.0039 |
| 2wiki | in-domain | 12576 | 0.9333 | 0.9336 | +0.0003 [-0.0012, +0.0019] | WITHIN | 0.6079 | +0.0044 | -0.0041 |
| webqsp | zero-shot | 1503 | 0.2680 | 0.2726 | +0.0046 [-0.0064, +0.0161] | WITHIN | 0.0535 | +0.0013 | -0.0013 |

## Against outputs\step1\fits\L-hotpotqa (reported only): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6540 | 0.8060 | +0.1520 [+0.1463, +0.1574] | GAIN | 0.0047 | +0.1567 | +0.2947 |
| squad | in-domain | 11873 | 0.9116 | 0.9143 | +0.0026 [+0.0003, +0.0049] | WITHIN | 0.9054 | +0.0026 | +0.0047 |
| musique | in-domain | 2417 | 0.5550 | 0.5807 | +0.0257 [+0.0178, +0.0331] | GAIN | 0.4734 | +0.0265 | +0.0463 |
| hotpotqa | zero-shot | 7405 | 0.8585 | 0.8508 | -0.0077 [-0.0125, -0.0028] | LOSS | 0.6846 | -0.0132 | -0.0435 |
| 2wiki | in-domain | 12576 | 0.8672 | 0.9336 | +0.0664 [+0.0630, +0.0700] | GAIN | 0.6079 | +0.1385 | +0.0695 |
| webqsp | zero-shot | 1503 | 0.1200 | 0.2726 | +0.1526 [+0.1291, +0.1769] | GAIN | 0.0535 | +0.1118 | +0.0898 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
