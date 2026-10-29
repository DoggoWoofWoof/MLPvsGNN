# Screen scr-r2gw-zrckpbg-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-wqk-zrckp-s0-hp (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | zero-shot | 9785 | 0.3927 | 0.4131 | +0.0204 [+0.0145, +0.0264] | GAIN | 0.0047 | +0.0095 | +0.0166 |
| squad | in-domain | 11873 | 0.9117 | 0.9069 | -0.0048 [-0.0070, -0.0025] | WITHIN | 0.9054 | -0.0048 | -0.0061 |
| musique | in-domain | 2417 | 0.5743 | 0.5613 | -0.0130 [-0.0191, -0.0071] | LOSS | 0.4734 | -0.0070 | -0.0434 |
| hotpotqa | zero-shot | 7405 | 0.8525 | 0.8456 | -0.0068 [-0.0100, -0.0034] | WITHIN | 0.6846 | -0.0126 | -0.0142 |
| 2wiki | in-domain | 12576 | 0.9299 | 0.9290 | -0.0010 [-0.0027, +0.0007] | WITHIN | 0.6079 | -0.0007 | -0.0138 |
| webqsp | in-domain | 1503 | 0.6382 | 0.6492 | +0.0110 [+0.0005, +0.0221] | GAIN | 0.0535 | +0.0027 | -0.0153 |

## Against outputs\screen\fits\scr-r2gw-zrckpb-hp (reported only): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | zero-shot | 9785 | 0.4459 | 0.4131 | -0.0328 [-0.0384, -0.0272] | LOSS | 0.0047 | -0.0296 | -0.0259 |
| squad | in-domain | 11873 | 0.9090 | 0.9069 | -0.0020 [-0.0039, +0.0000] | WITHIN | 0.9054 | -0.0020 | -0.0011 |
| musique | in-domain | 2417 | 0.5689 | 0.5613 | -0.0076 [-0.0132, -0.0022] | LOSS | 0.4734 | -0.0050 | -0.0207 |
| hotpotqa | zero-shot | 7405 | 0.8498 | 0.8456 | -0.0042 [-0.0072, -0.0011] | WITHIN | 0.6846 | -0.0081 | +0.0097 |
| 2wiki | in-domain | 12576 | 0.9304 | 0.9290 | -0.0015 [-0.0030, +0.0002] | WITHIN | 0.6079 | -0.0022 | -0.0010 |
| webqsp | in-domain | 1503 | 0.6488 | 0.6492 | +0.0004 [-0.0069, +0.0078] | WITHIN | 0.0535 | -0.0027 | -0.0060 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
