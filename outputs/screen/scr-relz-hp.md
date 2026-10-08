# Screen scr-relz-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-rel-hp (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7217 | 0.6975 | -0.0242 [-0.0280, -0.0205] | LOSS | 0.0047 | -0.0217 | -0.0476 |
| squad | in-domain | 11873 | 0.9105 | 0.9135 | +0.0030 [+0.0009, +0.0052] | WITHIN | 0.9054 | +0.0030 | +0.0003 |
| musique | in-domain | 2417 | 0.5603 | 0.5654 | +0.0051 [-0.0014, +0.0117] | WITHIN | 0.4734 | +0.0070 | -0.0083 |
| hotpotqa | zero-shot | 7405 | 0.8453 | 0.8336 | -0.0117 [-0.0155, -0.0080] | LOSS | 0.6846 | -0.0235 | +0.0003 |
| 2wiki | in-domain | 12576 | 0.8688 | 0.8669 | -0.0018 [-0.0041, +0.0005] | WITHIN | 0.6079 | +0.0001 | -0.0044 |
| webqsp | zero-shot | 1503 | 0.1902 | 0.2503 | +0.0601 [+0.0424, +0.0778] | GAIN | 0.0535 | +0.0399 | +0.0778 |

## Against outputs\step1\fits\L-hotpotqa (reported only): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6540 | 0.6975 | +0.0435 [+0.0383, +0.0484] | GAIN | 0.0047 | +0.0405 | +0.0916 |
| squad | in-domain | 11873 | 0.9116 | 0.9135 | +0.0019 [-0.0004, +0.0042] | WITHIN | 0.9054 | +0.0019 | +0.0005 |
| musique | in-domain | 2417 | 0.5550 | 0.5654 | +0.0103 [+0.0040, +0.0172] | GAIN | 0.4734 | +0.0190 | +0.0033 |
| hotpotqa | zero-shot | 7405 | 0.8585 | 0.8336 | -0.0250 [-0.0292, -0.0210] | LOSS | 0.6846 | -0.0493 | +0.0005 |
| 2wiki | in-domain | 12576 | 0.8672 | 0.8669 | -0.0003 [-0.0027, +0.0020] | WITHIN | 0.6079 | +0.0046 | -0.0129 |
| webqsp | zero-shot | 1503 | 0.1200 | 0.2503 | +0.1304 [+0.1096, +0.1501] | GAIN | 0.0535 | +0.0945 | +0.1171 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
