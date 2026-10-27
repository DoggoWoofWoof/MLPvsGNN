# Screen scr-a2xi-zspkxi (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-p1c3-zspk (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8009 | 0.8036 | +0.0027 [+0.0016, +0.0038] | WITHIN | 0.0047 | +0.0036 | +0.0144 |
| squad | in-domain | 11873 | 0.9114 | 0.9109 | -0.0005 [-0.0030, +0.0019] | WITHIN | 0.9054 | -0.0005 | -0.0083 |
| musique | zero-shot | 2417 | 0.5012 | 0.5122 | +0.0110 [+0.0043, +0.0180] | GAIN | 0.4734 | +0.0074 | +0.0046 |
| hotpotqa | in-domain | 7405 | 0.9080 | 0.9026 | -0.0054 [-0.0080, -0.0027] | WITHIN | 0.6846 | -0.0092 | -0.0113 |
| 2wiki | in-domain | 12576 | 0.9383 | 0.9339 | -0.0044 [-0.0061, -0.0027] | WITHIN | 0.6079 | -0.0140 | -0.0008 |
| webqsp | zero-shot | 1503 | 0.3229 | 0.2993 | -0.0236 [-0.0356, -0.0115] | LOSS | 0.0535 | -0.0206 | -0.0093 |

## Against outputs\step1\fits\L-musique (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6537 | 0.8036 | +0.1498 [+0.1442, +0.1551] | GAIN | 0.0047 | +0.1489 | +0.2893 |
| squad | in-domain | 11873 | 0.9100 | 0.9109 | +0.0009 [-0.0018, +0.0035] | WITHIN | 0.9054 | +0.0009 | -0.0099 |
| musique | zero-shot | 2417 | 0.2696 | 0.5122 | +0.2426 [+0.2307, +0.2549] | GAIN | 0.4734 | +0.1237 | +0.3914 |
| hotpotqa | in-domain | 7405 | 0.9015 | 0.9026 | +0.0011 [-0.0024, +0.0046] | WITHIN | 0.6846 | +0.0007 | -0.0063 |
| 2wiki | in-domain | 12576 | 0.8732 | 0.9339 | +0.0606 [+0.0575, +0.0640] | GAIN | 0.6079 | +0.1214 | +0.0715 |
| webqsp | zero-shot | 1503 | 0.1779 | 0.2993 | +0.1214 [+0.0974, +0.1460] | GAIN | 0.0535 | +0.0898 | +0.1184 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
