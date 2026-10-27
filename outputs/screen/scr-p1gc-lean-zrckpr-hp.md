# Screen scr-p1gc-zrckpr-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-p1gb-zrckp-hp (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7997 | 0.8012 | +0.0015 [+0.0006, +0.0024] | WITHIN | 0.0047 | +0.0011 | +0.0051 |
| squad | in-domain | 11873 | 0.9122 | 0.9112 | -0.0010 [-0.0029, +0.0009] | WITHIN | 0.9054 | -0.0010 | +0.0002 |
| musique | in-domain | 2417 | 0.5695 | 0.5689 | -0.0006 [-0.0054, +0.0043] | WITHIN | 0.4734 | -0.0058 | -0.0046 |
| hotpotqa | zero-shot | 7405 | 0.8546 | 0.8519 | -0.0027 [-0.0056, +0.0001] | WITHIN | 0.6846 | -0.0050 | +0.0134 |
| 2wiki | in-domain | 12576 | 0.9340 | 0.9303 | -0.0037 [-0.0053, -0.0022] | WITHIN | 0.6079 | -0.0106 | +0.0021 |
| webqsp | zero-shot | 1503 | 0.3364 | 0.3370 | +0.0006 [-0.0098, +0.0111] | WITHIN | 0.0535 | +0.0013 | +0.0027 |

## Against outputs\step1\fits\L-hotpotqa (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6540 | 0.8012 | +0.1472 [+0.1418, +0.1524] | GAIN | 0.0047 | +0.1509 | +0.2825 |
| squad | in-domain | 11873 | 0.9116 | 0.9112 | -0.0004 [-0.0027, +0.0019] | WITHIN | 0.9054 | -0.0004 | +0.0012 |
| musique | in-domain | 2417 | 0.5550 | 0.5689 | +0.0139 [+0.0058, +0.0218] | GAIN | 0.4734 | +0.0132 | +0.0273 |
| hotpotqa | zero-shot | 7405 | 0.8585 | 0.8519 | -0.0066 [-0.0114, -0.0022] | WITHIN | 0.6846 | -0.0151 | -0.0246 |
| 2wiki | in-domain | 12576 | 0.8672 | 0.9303 | +0.0631 [+0.0599, +0.0664] | GAIN | 0.6079 | +0.1246 | +0.0659 |
| webqsp | zero-shot | 1503 | 0.1200 | 0.3370 | +0.2170 [+0.1932, +0.2406] | GAIN | 0.0535 | +0.1743 | +0.0685 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
