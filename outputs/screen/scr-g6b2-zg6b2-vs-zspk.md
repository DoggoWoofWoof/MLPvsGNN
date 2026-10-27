# Screen scr-g6b2-zg6b2 (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-kg-base-zspk (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8013 | 0.8112 | +0.0099 [+0.0083, +0.0115] | GAIN | 0.0047 | +0.0115 | +0.0351 |
| squad | in-domain | 11873 | 0.9102 | 0.9119 | +0.0017 [-0.0007, +0.0040] | WITHIN | 0.9054 | +0.0017 | +0.0023 |
| musique | zero-shot | 2417 | 0.5146 | 0.4780 | -0.0366 [-0.0446, -0.0289] | LOSS | 0.4734 | -0.0132 | -0.2276 |
| hotpotqa | in-domain | 7405 | 0.9086 | 0.9113 | +0.0027 [+0.0004, +0.0051] | WITHIN | 0.6846 | +0.0061 | +0.0201 |
| 2wiki | in-domain | 12576 | 0.9364 | 0.9431 | +0.0067 [+0.0051, +0.0083] | WITHIN | 0.6079 | +0.0204 | +0.0053 |
| webqsp | zero-shot | 1503 | 0.3151 | 0.2222 | -0.0929 [-0.1134, -0.0726] | LOSS | 0.0535 | -0.0739 | -0.0546 |

## Against outputs\screen\fits\scr-kg-base-zg6 (reported only): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8127 | 0.8112 | -0.0015 [-0.0026, -0.0004] | WITHIN | 0.0047 | -0.0017 | -0.0046 |
| squad | in-domain | 11873 | 0.9120 | 0.9119 | -0.0001 [-0.0020, +0.0019] | WITHIN | 0.9054 | -0.0001 | -0.0006 |
| musique | zero-shot | 2417 | 0.5138 | 0.4780 | -0.0357 [-0.0416, -0.0302] | LOSS | 0.4734 | -0.0211 | -0.2396 |
| hotpotqa | in-domain | 7405 | 0.9127 | 0.9113 | -0.0014 [-0.0034, +0.0007] | WITHIN | 0.6846 | -0.0008 | +0.0059 |
| 2wiki | in-domain | 12576 | 0.9448 | 0.9431 | -0.0017 [-0.0030, -0.0003] | WITHIN | 0.6079 | -0.0041 | -0.0073 |
| webqsp | zero-shot | 1503 | 0.2398 | 0.2222 | -0.0176 [-0.0346, -0.0015] | LOSS | 0.0535 | -0.0160 | -0.0126 |

## Against outputs\screen\fits\scr-g6b-zg6b (reported only): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8126 | 0.8112 | -0.0014 [-0.0023, -0.0006] | WITHIN | 0.0047 | -0.0018 | -0.0026 |
| squad | in-domain | 11873 | 0.9113 | 0.9119 | +0.0006 [-0.0012, +0.0024] | WITHIN | 0.9054 | +0.0006 | +0.0006 |
| musique | zero-shot | 2417 | 0.5052 | 0.4780 | -0.0272 [-0.0327, -0.0220] | LOSS | 0.4734 | -0.0149 | -0.2077 |
| hotpotqa | in-domain | 7405 | 0.9128 | 0.9113 | -0.0015 [-0.0036, +0.0005] | WITHIN | 0.6846 | -0.0019 | +0.0023 |
| 2wiki | in-domain | 12576 | 0.9424 | 0.9431 | +0.0007 [-0.0006, +0.0020] | WITHIN | 0.6079 | +0.0014 | -0.0033 |
| webqsp | zero-shot | 1503 | 0.2784 | 0.2222 | -0.0562 [-0.0706, -0.0420] | LOSS | 0.0535 | -0.0446 | -0.0452 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
