# Screen L-metaqa (p@swa, s1eval carves)

## Against outputs\full_zrct\fits\L-metaqa (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | zero-shot | 9785 | 0.1417 | 0.2494 | +0.1077 [+0.1013, +0.1141] | GAIN | 0.0047 | +0.0868 | +0.0029 |
| squad | in-domain | 11873 | 0.9106 | 0.9110 | +0.0004 [-0.0019, +0.0025] | WITHIN | 0.9054 | +0.0004 | +0.0036 |
| musique | in-domain | 2417 | 0.5701 | 0.5798 | +0.0097 [+0.0022, +0.0169] | GAIN | 0.4734 | -0.0033 | +0.0240 |
| hotpotqa | in-domain | 7405 | 0.9014 | 0.9086 | +0.0073 [+0.0042, +0.0103] | WITHIN | 0.6846 | +0.0115 | -0.0041 |
| 2wiki | in-domain | 12576 | 0.8705 | 0.9337 | +0.0632 [+0.0599, +0.0665] | GAIN | 0.6079 | +0.1283 | +0.0618 |
| webqsp | zero-shot | 1503 | 0.0997 | 0.1139 | +0.0142 [+0.0046, +0.0235] | GAIN | 0.0535 | +0.0113 | +0.0126 |

## Against outputs\full_zrck\fits\L-metaqa (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | zero-shot | 9785 | 0.2036 | 0.2494 | +0.0459 [+0.0407, +0.0507] | GAIN | 0.0047 | +0.0407 | -0.0048 |
| squad | in-domain | 11873 | 0.9108 | 0.9110 | +0.0002 [-0.0016, +0.0020] | WITHIN | 0.9054 | +0.0002 | +0.0025 |
| musique | in-domain | 2417 | 0.5744 | 0.5798 | +0.0054 [+0.0006, +0.0106] | WITHIN | 0.4734 | +0.0070 | +0.0008 |
| hotpotqa | in-domain | 7405 | 0.9076 | 0.9086 | +0.0010 [-0.0010, +0.0032] | WITHIN | 0.6846 | +0.0026 | -0.0038 |
| 2wiki | in-domain | 12576 | 0.9367 | 0.9337 | -0.0030 [-0.0045, -0.0015] | WITHIN | 0.6079 | -0.0056 | -0.0090 |
| webqsp | zero-shot | 1503 | 0.1208 | 0.1139 | -0.0069 [-0.0131, -0.0006] | WITHIN | 0.0535 | -0.0067 | -0.0060 |

## Against outputs\step1\fits\L-metaqa (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | zero-shot | 9785 | 0.0773 | 0.2494 | +0.1721 [+0.1648, +0.1793] | GAIN | 0.0047 | +0.1420 | +0.0038 |
| squad | in-domain | 11873 | 0.9076 | 0.9110 | +0.0034 [+0.0013, +0.0056] | WITHIN | 0.9054 | +0.0034 | +0.0070 |
| musique | in-domain | 2417 | 0.5614 | 0.5798 | +0.0184 [+0.0104, +0.0262] | GAIN | 0.4734 | +0.0046 | +0.0385 |
| hotpotqa | in-domain | 7405 | 0.9046 | 0.9086 | +0.0041 [+0.0008, +0.0073] | WITHIN | 0.6846 | +0.0050 | -0.0005 |
| 2wiki | in-domain | 12576 | 0.8639 | 0.9337 | +0.0697 [+0.0666, +0.0731] | GAIN | 0.6079 | +0.1362 | +0.0754 |
| webqsp | zero-shot | 1503 | 0.0926 | 0.1139 | +0.0213 [+0.0122, +0.0311] | GAIN | 0.0535 | +0.0193 | +0.0100 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
