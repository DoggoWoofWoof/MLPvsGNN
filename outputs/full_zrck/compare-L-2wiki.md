# Screen L-2wiki (p@swa, s1eval carves)

## Against outputs\full_zrct\fits\L-2wiki (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7793 | 0.8008 | +0.0215 [+0.0185, +0.0242] | GAIN | 0.0047 | +0.0170 | +0.0265 |
| squad | in-domain | 11873 | 0.9121 | 0.9102 | -0.0019 [-0.0037, +0.0001] | WITHIN | 0.9054 | -0.0019 | +0.0008 |
| musique | in-domain | 2417 | 0.5669 | 0.5708 | +0.0039 [-0.0031, +0.0107] | WITHIN | 0.4734 | -0.0037 | +0.0120 |
| hotpotqa | in-domain | 7405 | 0.9001 | 0.9045 | +0.0045 [+0.0009, +0.0078] | WITHIN | 0.6846 | +0.0042 | -0.0005 |
| 2wiki | zero-shot | 12576 | 0.7937 | 0.8576 | +0.0639 [+0.0607, +0.0669] | GAIN | 0.6079 | +0.0816 | +0.0791 |
| webqsp | zero-shot | 1503 | 0.3042 | 0.3590 | +0.0548 [+0.0387, +0.0706] | GAIN | 0.0535 | +0.0486 | +0.0173 |

## Against outputs\step1\fits\L-2wiki (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6468 | 0.8008 | +0.1540 [+0.1482, +0.1594] | GAIN | 0.0047 | +0.1549 | +0.2872 |
| squad | in-domain | 11873 | 0.9112 | 0.9102 | -0.0010 [-0.0031, +0.0010] | WITHIN | 0.9054 | -0.0010 | -0.0017 |
| musique | in-domain | 2417 | 0.5601 | 0.5708 | +0.0107 [+0.0031, +0.0177] | GAIN | 0.4734 | +0.0083 | +0.0219 |
| hotpotqa | in-domain | 7405 | 0.8977 | 0.9045 | +0.0068 [+0.0033, +0.0102] | WITHIN | 0.6846 | +0.0085 | +0.0100 |
| 2wiki | zero-shot | 12576 | 0.8074 | 0.8576 | +0.0502 [+0.0469, +0.0534] | GAIN | 0.6079 | +0.0563 | +0.0890 |
| webqsp | zero-shot | 1503 | 0.1075 | 0.3590 | +0.2514 [+0.2266, +0.2761] | GAIN | 0.0535 | +0.2049 | +0.0792 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
