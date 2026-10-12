# Screen scr-u1c-zrc-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-zrct-hp (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7793 | 0.7864 | +0.0071 [+0.0052, +0.0090] | WITHIN | 0.0047 | +0.0060 | -0.0013 |
| squad | in-domain | 11873 | 0.9112 | 0.9107 | -0.0005 [-0.0028, +0.0016] | WITHIN | 0.9054 | -0.0005 | -0.0032 |
| musique | in-domain | 2417 | 0.5645 | 0.5634 | -0.0011 [-0.0081, +0.0056] | WITHIN | 0.4734 | +0.0021 | +0.0116 |
| hotpotqa | zero-shot | 7405 | 0.8451 | 0.8023 | -0.0428 [-0.0473, -0.0384] | LOSS | 0.6846 | -0.0831 | -0.0058 |
| 2wiki | in-domain | 12576 | 0.8731 | 0.8598 | -0.0133 [-0.0165, -0.0100] | LOSS | 0.6079 | -0.0322 | +0.0070 |
| webqsp | zero-shot | 1503 | 0.2865 | 0.2945 | +0.0079 [-0.0051, +0.0205] | WITHIN | 0.0535 | +0.0020 | -0.0080 |

## Against outputs\step1\fits\L-hotpotqa (reported only): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6540 | 0.7864 | +0.1324 [+0.1270, +0.1377] | GAIN | 0.0047 | +0.1395 | +0.2540 |
| squad | in-domain | 11873 | 0.9116 | 0.9107 | -0.0009 [-0.0035, +0.0015] | WITHIN | 0.9054 | -0.0009 | -0.0001 |
| musique | in-domain | 2417 | 0.5550 | 0.5634 | +0.0083 [+0.0021, +0.0149] | GAIN | 0.4734 | +0.0178 | +0.0120 |
| hotpotqa | zero-shot | 7405 | 0.8585 | 0.8023 | -0.0562 [-0.0612, -0.0515] | LOSS | 0.6846 | -0.1074 | -0.0058 |
| 2wiki | in-domain | 12576 | 0.8672 | 0.8598 | -0.0074 [-0.0109, -0.0042] | WITHIN | 0.6079 | -0.0149 | -0.0018 |
| webqsp | zero-shot | 1503 | 0.1200 | 0.2945 | +0.1745 [+0.1511, +0.1967] | GAIN | 0.0535 | +0.1384 | +0.0393 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
