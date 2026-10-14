# Screen L-metaqa (p@swa, s1eval carves)

## Against outputs\full_zrct\fits\L-metaqa (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | zero-shot | 9785 | 0.1417 | 0.2036 | +0.0619 [+0.0567, +0.0678] | GAIN | 0.0047 | +0.0461 | +0.0077 |
| squad | in-domain | 11873 | 0.9106 | 0.9108 | +0.0003 [-0.0019, +0.0024] | WITHIN | 0.9054 | +0.0003 | +0.0011 |
| musique | in-domain | 2417 | 0.5701 | 0.5744 | +0.0042 [-0.0033, +0.0115] | WITHIN | 0.4734 | -0.0103 | +0.0232 |
| hotpotqa | in-domain | 7405 | 0.9014 | 0.9076 | +0.0063 [+0.0029, +0.0095] | WITHIN | 0.6846 | +0.0089 | -0.0003 |
| 2wiki | in-domain | 12576 | 0.8705 | 0.9367 | +0.0662 [+0.0628, +0.0695] | GAIN | 0.6079 | +0.1339 | +0.0708 |
| webqsp | zero-shot | 1503 | 0.0997 | 0.1208 | +0.0212 [+0.0119, +0.0304] | GAIN | 0.0535 | +0.0180 | +0.0186 |

## Against outputs\step1\fits\L-metaqa (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | zero-shot | 9785 | 0.0773 | 0.2036 | +0.1262 [+0.1196, +0.1328] | GAIN | 0.0047 | +0.1013 | +0.0086 |
| squad | in-domain | 11873 | 0.9076 | 0.9108 | +0.0032 [+0.0009, +0.0055] | WITHIN | 0.9054 | +0.0032 | +0.0045 |
| musique | in-domain | 2417 | 0.5614 | 0.5744 | +0.0130 [+0.0050, +0.0210] | GAIN | 0.4734 | -0.0025 | +0.0376 |
| hotpotqa | in-domain | 7405 | 0.9046 | 0.9076 | +0.0030 [-0.0005, +0.0065] | WITHIN | 0.6846 | +0.0024 | +0.0032 |
| 2wiki | in-domain | 12576 | 0.8639 | 0.9367 | +0.0727 [+0.0696, +0.0761] | GAIN | 0.6079 | +0.1419 | +0.0844 |
| webqsp | zero-shot | 1503 | 0.0926 | 0.1208 | +0.0282 [+0.0191, +0.0387] | GAIN | 0.0535 | +0.0259 | +0.0160 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
