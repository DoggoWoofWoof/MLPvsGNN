# Screen scr-zgr-off (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-zrm (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7789 | 0.7585 | -0.0205 [-0.0225, -0.0184] | LOSS | 0.0047 | -0.0201 | -0.0597 |
| squad | in-domain | 11873 | 0.9111 | 0.9119 | +0.0008 [-0.0010, +0.0026] | WITHIN | 0.9054 | +0.0008 | +0.0000 |
| musique | zero-shot | 2417 | 0.5231 | 0.5294 | +0.0063 [+0.0013, +0.0112] | WITHIN | 0.4734 | +0.0066 | +0.0058 |
| hotpotqa | in-domain | 7405 | 0.9040 | 0.8999 | -0.0041 [-0.0065, -0.0018] | WITHIN | 0.6846 | -0.0046 | +0.0053 |
| 2wiki | in-domain | 12576 | 0.8709 | 0.8597 | -0.0113 [-0.0133, -0.0093] | LOSS | 0.6079 | -0.0257 | +0.0089 |
| webqsp | zero-shot | 1503 | 0.3128 | 0.3090 | -0.0038 [-0.0139, +0.0059] | WITHIN | 0.0535 | -0.0027 | -0.0186 |

## Against outputs\screen\fits\scr-zgr (reported only): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7823 | 0.7585 | -0.0238 [-0.0261, -0.0216] | LOSS | 0.0047 | -0.0223 | -0.0661 |
| squad | in-domain | 11873 | 0.8993 | 0.9119 | +0.0126 [+0.0097, +0.0157] | GAIN | 0.9054 | +0.0126 | +0.0280 |
| musique | zero-shot | 2417 | 0.5174 | 0.5294 | +0.0120 [+0.0064, +0.0180] | GAIN | 0.4734 | +0.0083 | +0.1001 |
| hotpotqa | in-domain | 7405 | 0.8977 | 0.8999 | +0.0022 [-0.0007, +0.0050] | WITHIN | 0.6846 | +0.0055 | +0.0200 |
| 2wiki | in-domain | 12576 | 0.8706 | 0.8597 | -0.0109 [-0.0131, -0.0089] | LOSS | 0.6079 | -0.0272 | +0.0342 |
| webqsp | zero-shot | 1503 | 0.3124 | 0.3090 | -0.0034 [-0.0155, +0.0091] | WITHIN | 0.0535 | +0.0020 | -0.0333 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
