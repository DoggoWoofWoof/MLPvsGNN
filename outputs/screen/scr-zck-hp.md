# Screen scr-zck-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-zrct-hp (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7793 | 0.7792 | -0.0001 [-0.0011, +0.0009] | WITHIN | 0.0047 | +0.0002 | -0.0012 |
| squad | in-domain | 11873 | 0.9112 | 0.9127 | +0.0015 [-0.0007, +0.0036] | WITHIN | 0.9054 | +0.0015 | +0.0004 |
| musique | in-domain | 2417 | 0.5645 | 0.5674 | +0.0030 [-0.0041, +0.0099] | WITHIN | 0.4734 | +0.0120 | -0.0058 |
| hotpotqa | zero-shot | 7405 | 0.8451 | 0.8419 | -0.0032 [-0.0068, +0.0002] | WITHIN | 0.6846 | -0.0074 | +0.0007 |
| 2wiki | in-domain | 12576 | 0.8731 | 0.8649 | -0.0082 [-0.0103, -0.0061] | LOSS | 0.6079 | -0.0183 | +0.0041 |
| webqsp | zero-shot | 1503 | 0.2865 | 0.3267 | +0.0402 [+0.0257, +0.0557] | GAIN | 0.0535 | +0.0279 | +0.0399 |

## Against outputs\screen\fits\scr-zrm-hp (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7801 | 0.7792 | -0.0009 [-0.0020, +0.0002] | WITHIN | 0.0047 | -0.0002 | -0.0015 |
| squad | in-domain | 11873 | 0.9116 | 0.9127 | +0.0012 [-0.0010, +0.0033] | WITHIN | 0.9054 | +0.0012 | +0.0015 |
| musique | in-domain | 2417 | 0.5600 | 0.5674 | +0.0075 [+0.0012, +0.0142] | WITHIN | 0.4734 | +0.0099 | -0.0178 |
| hotpotqa | zero-shot | 7405 | 0.8457 | 0.8419 | -0.0038 [-0.0072, -0.0005] | WITHIN | 0.6846 | -0.0082 | -0.0036 |
| 2wiki | in-domain | 12576 | 0.8681 | 0.8649 | -0.0031 [-0.0052, -0.0010] | WITHIN | 0.6079 | -0.0037 | +0.0017 |
| webqsp | zero-shot | 1503 | 0.2416 | 0.3267 | +0.0851 [+0.0682, +0.1035] | GAIN | 0.0535 | +0.0712 | +0.0240 |

## Against outputs\step1\fits\L-hotpotqa (reported only): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6540 | 0.7792 | +0.1252 [+0.1199, +0.1301] | GAIN | 0.0047 | +0.1337 | +0.2541 |
| squad | in-domain | 11873 | 0.9116 | 0.9127 | +0.0011 [-0.0012, +0.0034] | WITHIN | 0.9054 | +0.0011 | +0.0035 |
| musique | in-domain | 2417 | 0.5550 | 0.5674 | +0.0124 [+0.0055, +0.0194] | GAIN | 0.4734 | +0.0277 | -0.0054 |
| hotpotqa | zero-shot | 7405 | 0.8585 | 0.8419 | -0.0167 [-0.0206, -0.0128] | LOSS | 0.6846 | -0.0317 | +0.0007 |
| 2wiki | in-domain | 12576 | 0.8672 | 0.8649 | -0.0023 [-0.0046, +0.0001] | WITHIN | 0.6079 | -0.0010 | -0.0047 |
| webqsp | zero-shot | 1503 | 0.1200 | 0.3267 | +0.2067 [+0.1820, +0.2316] | GAIN | 0.0535 | +0.1643 | +0.0872 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
