# Screen scr-p1c2-zrc-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-zrct-hp (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7793 | 0.8002 | +0.0209 [+0.0181, +0.0237] | GAIN | 0.0047 | +0.0184 | +0.0173 |
| squad | in-domain | 11873 | 0.9112 | 0.9134 | +0.0022 [+0.0002, +0.0043] | WITHIN | 0.9054 | +0.0022 | -0.0003 |
| musique | in-domain | 2417 | 0.5645 | 0.5651 | +0.0006 [-0.0061, +0.0068] | WITHIN | 0.4734 | +0.0000 | +0.0170 |
| hotpotqa | zero-shot | 7405 | 0.8451 | 0.8363 | -0.0088 [-0.0132, -0.0047] | LOSS | 0.6846 | -0.0159 | -0.0023 |
| 2wiki | in-domain | 12576 | 0.8731 | 0.9125 | +0.0394 [+0.0363, +0.0425] | GAIN | 0.6079 | +0.0744 | +0.0428 |
| webqsp | zero-shot | 1503 | 0.2865 | 0.3142 | +0.0276 [+0.0133, +0.0419] | GAIN | 0.0535 | +0.0200 | +0.0040 |

## Against outputs\step1\fits\L-hotpotqa (reported only): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6540 | 0.8002 | +0.1462 [+0.1407, +0.1516] | GAIN | 0.0047 | +0.1519 | +0.2726 |
| squad | in-domain | 11873 | 0.9116 | 0.9134 | +0.0018 [-0.0003, +0.0038] | WITHIN | 0.9054 | +0.0018 | +0.0029 |
| musique | in-domain | 2417 | 0.5550 | 0.5651 | +0.0100 [+0.0040, +0.0163] | GAIN | 0.4734 | +0.0157 | +0.0174 |
| hotpotqa | zero-shot | 7405 | 0.8585 | 0.8363 | -0.0223 [-0.0269, -0.0178] | LOSS | 0.6846 | -0.0402 | -0.0023 |
| 2wiki | in-domain | 12576 | 0.8672 | 0.9125 | +0.0453 [+0.0421, +0.0487] | GAIN | 0.6079 | +0.0918 | +0.0340 |
| webqsp | zero-shot | 1503 | 0.1200 | 0.3142 | +0.1942 [+0.1709, +0.2171] | GAIN | 0.0535 | +0.1564 | +0.0512 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
