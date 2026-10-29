# Screen scr-r2ew-zrckpbe-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-wqk-zrckp-s0-hp (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | zero-shot | 9785 | 0.3927 | 0.4403 | +0.0476 [+0.0412, +0.0540] | GAIN | 0.0047 | +0.0272 | +0.0472 |
| squad | in-domain | 11873 | 0.9117 | 0.9105 | -0.0013 [-0.0036, +0.0010] | WITHIN | 0.9054 | -0.0013 | -0.0040 |
| musique | in-domain | 2417 | 0.5743 | 0.5639 | -0.0105 [-0.0173, -0.0039] | LOSS | 0.4734 | -0.0124 | -0.0066 |
| hotpotqa | zero-shot | 7405 | 0.8525 | 0.8531 | +0.0007 [-0.0028, +0.0041] | WITHIN | 0.6846 | +0.0020 | -0.0090 |
| 2wiki | in-domain | 12576 | 0.9299 | 0.9307 | +0.0008 [-0.0009, +0.0026] | WITHIN | 0.6079 | +0.0014 | -0.0169 |
| webqsp | in-domain | 1503 | 0.6382 | 0.6425 | +0.0043 [-0.0059, +0.0144] | WITHIN | 0.0535 | +0.0007 | +0.0000 |

## Against outputs\screen\fits\scr-r2gw-zrckpb-hp (reported only): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | zero-shot | 9785 | 0.4459 | 0.4403 | -0.0056 [-0.0114, +0.0002] | WITHIN | 0.0047 | -0.0120 | +0.0048 |
| squad | in-domain | 11873 | 0.9090 | 0.9105 | +0.0015 [-0.0008, +0.0038] | WITHIN | 0.9054 | +0.0015 | +0.0010 |
| musique | in-domain | 2417 | 0.5689 | 0.5639 | -0.0050 [-0.0115, +0.0014] | WITHIN | 0.4734 | -0.0103 | +0.0161 |
| hotpotqa | zero-shot | 7405 | 0.8498 | 0.8531 | +0.0033 [-0.0003, +0.0066] | WITHIN | 0.6846 | +0.0065 | +0.0149 |
| 2wiki | in-domain | 12576 | 0.9304 | 0.9307 | +0.0003 [-0.0015, +0.0021] | WITHIN | 0.6079 | -0.0002 | -0.0041 |
| webqsp | in-domain | 1503 | 0.6488 | 0.6425 | -0.0063 [-0.0173, +0.0050] | WITHIN | 0.0535 | -0.0047 | +0.0093 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
