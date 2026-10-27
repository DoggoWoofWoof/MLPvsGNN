# Screen scr-a5cp-zrckpc-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-p1gb-zrckp-hp (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7997 | 0.8019 | +0.0022 [+0.0012, +0.0034] | WITHIN | 0.0047 | +0.0018 | +0.0105 |
| squad | in-domain | 11873 | 0.9122 | 0.9122 | +0.0000 [-0.0019, +0.0019] | WITHIN | 0.9054 | +0.0000 | +0.0008 |
| musique | in-domain | 2417 | 0.5695 | 0.5695 | +0.0000 [-0.0053, +0.0050] | WITHIN | 0.4734 | -0.0054 | -0.0062 |
| hotpotqa | zero-shot | 7405 | 0.8546 | 0.8527 | -0.0019 [-0.0051, +0.0011] | WITHIN | 0.6846 | -0.0012 | -0.0117 |
| 2wiki | in-domain | 12576 | 0.9340 | 0.9256 | -0.0084 [-0.0100, -0.0066] | LOSS | 0.6079 | -0.0249 | -0.0015 |
| webqsp | zero-shot | 1503 | 0.3364 | 0.3730 | +0.0366 [+0.0216, +0.0522] | GAIN | 0.0535 | +0.0299 | +0.0373 |

## Against outputs\step1\fits\L-hotpotqa (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6540 | 0.8019 | +0.1479 [+0.1424, +0.1533] | GAIN | 0.0047 | +0.1517 | +0.2879 |
| squad | in-domain | 11873 | 0.9116 | 0.9122 | +0.0006 [-0.0015, +0.0029] | WITHIN | 0.9054 | +0.0006 | +0.0019 |
| musique | in-domain | 2417 | 0.5550 | 0.5695 | +0.0145 [+0.0061, +0.0226] | GAIN | 0.4734 | +0.0137 | +0.0257 |
| hotpotqa | zero-shot | 7405 | 0.8585 | 0.8527 | -0.0058 [-0.0106, -0.0011] | WITHIN | 0.6846 | -0.0113 | -0.0497 |
| 2wiki | in-domain | 12576 | 0.8672 | 0.9256 | +0.0584 [+0.0553, +0.0619] | GAIN | 0.6079 | +0.1103 | +0.0623 |
| webqsp | zero-shot | 1503 | 0.1200 | 0.3730 | +0.2530 [+0.2274, +0.2806] | GAIN | 0.0535 | +0.2029 | +0.1031 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
