# Screen scr-p1c2-zsp-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-zspg-hp (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7807 | 0.8008 | +0.0201 [+0.0172, +0.0228] | GAIN | 0.0047 | +0.0175 | +0.0266 |
| squad | in-domain | 11873 | 0.9109 | 0.9111 | +0.0002 [-0.0020, +0.0025] | WITHIN | 0.9054 | +0.0002 | +0.0008 |
| musique | in-domain | 2417 | 0.5647 | 0.5707 | +0.0060 [-0.0007, +0.0126] | WITHIN | 0.4734 | +0.0070 | +0.0132 |
| hotpotqa | zero-shot | 7405 | 0.8435 | 0.8475 | +0.0040 [-0.0001, +0.0080] | WITHIN | 0.6846 | +0.0101 | -0.0070 |
| 2wiki | in-domain | 12576 | 0.8744 | 0.9171 | +0.0426 [+0.0395, +0.0458] | GAIN | 0.6079 | +0.0812 | +0.0332 |
| webqsp | zero-shot | 1503 | 0.2262 | 0.2603 | +0.0341 [+0.0181, +0.0493] | GAIN | 0.0535 | +0.0266 | +0.0180 |

## Against outputs\step1\fits\L-hotpotqa (reported only): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6540 | 0.8008 | +0.1468 [+0.1412, +0.1521] | GAIN | 0.0047 | +0.1516 | +0.2828 |
| squad | in-domain | 11873 | 0.9116 | 0.9111 | -0.0006 [-0.0031, +0.0019] | WITHIN | 0.9054 | -0.0006 | +0.0027 |
| musique | in-domain | 2417 | 0.5550 | 0.5707 | +0.0157 [+0.0094, +0.0222] | GAIN | 0.4734 | +0.0257 | +0.0194 |
| hotpotqa | zero-shot | 7405 | 0.8585 | 0.8475 | -0.0111 [-0.0156, -0.0066] | LOSS | 0.6846 | -0.0184 | -0.0032 |
| 2wiki | in-domain | 12576 | 0.8672 | 0.9171 | +0.0498 [+0.0466, +0.0531] | GAIN | 0.6079 | +0.1031 | +0.0365 |
| webqsp | zero-shot | 1503 | 0.1200 | 0.2603 | +0.1403 [+0.1175, +0.1626] | GAIN | 0.0535 | +0.1078 | +0.0818 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
