# Screen scr-a2xmx-zrckprxm (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-r1-zrckpr (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8012 | 0.8004 | -0.0008 [-0.0018, +0.0002] | WITHIN | 0.0047 | -0.0004 | -0.0053 |
| squad | in-domain | 11873 | 0.9127 | 0.9111 | -0.0016 [-0.0035, +0.0003] | WITHIN | 0.9054 | -0.0016 | -0.0009 |
| musique | zero-shot | 2417 | 0.5197 | 0.5165 | -0.0032 [-0.0090, +0.0027] | WITHIN | 0.4734 | +0.0070 | +0.0066 |
| hotpotqa | in-domain | 7405 | 0.9083 | 0.9060 | -0.0023 [-0.0045, -0.0002] | WITHIN | 0.6846 | -0.0046 | +0.0050 |
| 2wiki | in-domain | 12576 | 0.9361 | 0.9356 | -0.0005 [-0.0019, +0.0009] | WITHIN | 0.6079 | -0.0016 | +0.0017 |
| webqsp | zero-shot | 1503 | 0.3762 | 0.3856 | +0.0094 [-0.0019, +0.0208] | WITHIN | 0.0535 | +0.0093 | +0.0040 |

## Against outputs\step1\fits\L-musique (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6537 | 0.8004 | +0.1467 [+0.1412, +0.1520] | GAIN | 0.0047 | +0.1447 | +0.2778 |
| squad | in-domain | 11873 | 0.9100 | 0.9111 | +0.0012 [-0.0010, +0.0033] | WITHIN | 0.9054 | +0.0012 | -0.0039 |
| musique | zero-shot | 2417 | 0.2696 | 0.5165 | +0.2469 [+0.2350, +0.2593] | GAIN | 0.4734 | +0.1374 | +0.4204 |
| hotpotqa | in-domain | 7405 | 0.9015 | 0.9060 | +0.0045 [+0.0012, +0.0078] | WITHIN | 0.6846 | +0.0047 | -0.0026 |
| 2wiki | in-domain | 12576 | 0.8732 | 0.9356 | +0.0624 [+0.0594, +0.0656] | GAIN | 0.6079 | +0.1292 | +0.0722 |
| webqsp | zero-shot | 1503 | 0.1779 | 0.3856 | +0.2077 [+0.1858, +0.2311] | GAIN | 0.0535 | +0.1750 | +0.0872 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
