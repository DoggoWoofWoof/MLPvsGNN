# Screen L-2wiki (p@swa, s1eval carves)

## Against outputs\full_zret\fits\L-2wiki (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6464 | 0.7642 | +0.1178 [+0.1131, +0.1225] | GAIN | 0.0047 | +0.1173 | +0.2299 |
| squad | in-domain | 11873 | 0.9109 | 0.9109 | +0.0000 [+0.0000, +0.0000] | WITHIN | 0.9054 | +0.0000 | +0.0000 |
| musique | in-domain | 2417 | 0.5656 | 0.5656 | +0.0000 [+0.0000, +0.0000] | WITHIN | 0.4734 | +0.0000 | +0.0000 |
| hotpotqa | in-domain | 7405 | 0.9035 | 0.9035 | +0.0000 [+0.0000, +0.0000] | WITHIN | 0.6846 | +0.0000 | +0.0000 |
| 2wiki | zero-shot | 12576 | 0.7993 | 0.7993 | +0.0000 [+0.0000, +0.0000] | WITHIN | 0.6079 | +0.0000 | +0.0000 |
| webqsp | zero-shot | 1503 | 0.1338 | 0.2108 | +0.0770 [+0.0583, +0.0946] | GAIN | 0.0535 | +0.0605 | +0.0366 |

## Against outputs\full_rmatch\fits\L-2wiki (reported only): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7803 | 0.7642 | -0.0161 [-0.0184, -0.0140] | LOSS | 0.0047 | -0.0210 | -0.0480 |
| squad | in-domain | 11873 | 0.9100 | 0.9109 | +0.0009 [-0.0013, +0.0031] | WITHIN | 0.9054 | +0.0009 | +0.0014 |
| musique | in-domain | 2417 | 0.5576 | 0.5656 | +0.0080 [+0.0019, +0.0143] | GAIN | 0.4734 | +0.0087 | +0.0062 |
| hotpotqa | in-domain | 7405 | 0.9006 | 0.9035 | +0.0029 [+0.0003, +0.0055] | WITHIN | 0.6846 | +0.0063 | -0.0024 |
| 2wiki | zero-shot | 12576 | 0.7937 | 0.7993 | +0.0055 [+0.0028, +0.0082] | WITHIN | 0.6079 | +0.0060 | -0.0014 |
| webqsp | zero-shot | 1503 | 0.2607 | 0.2108 | -0.0500 [-0.0652, -0.0359] | LOSS | 0.0535 | -0.0419 | -0.0200 |

## Against outputs\step1\fits\L-2wiki (reported only): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6468 | 0.7642 | +0.1174 [+0.1124, +0.1224] | GAIN | 0.0047 | +0.1170 | +0.2183 |
| squad | in-domain | 11873 | 0.9112 | 0.9109 | -0.0003 [-0.0025, +0.0018] | WITHIN | 0.9054 | -0.0003 | -0.0010 |
| musique | in-domain | 2417 | 0.5601 | 0.5656 | +0.0055 [-0.0009, +0.0118] | WITHIN | 0.4734 | +0.0050 | +0.0029 |
| hotpotqa | in-domain | 7405 | 0.8977 | 0.9035 | +0.0058 [+0.0031, +0.0086] | WITHIN | 0.6846 | +0.0107 | +0.0046 |
| 2wiki | zero-shot | 12576 | 0.8074 | 0.7993 | -0.0081 [-0.0107, -0.0054] | LOSS | 0.6079 | -0.0177 | -0.0008 |
| webqsp | zero-shot | 1503 | 0.1075 | 0.2108 | +0.1032 [+0.0852, +0.1219] | GAIN | 0.0535 | +0.0838 | +0.0512 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
