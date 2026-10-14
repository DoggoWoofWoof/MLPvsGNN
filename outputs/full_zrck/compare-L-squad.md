# Screen L-squad (p@swa, s1eval carves)

## Against outputs\full_zrct\fits\L-squad (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7799 | 0.8015 | +0.0216 [+0.0186, +0.0243] | GAIN | 0.0047 | +0.0178 | +0.0213 |
| squad | zero-shot | 11873 | 0.8912 | 0.8884 | -0.0028 [-0.0056, +0.0002] | WITHIN | 0.9054 | -0.0028 | -0.0009 |
| musique | in-domain | 2417 | 0.5710 | 0.5771 | +0.0061 [-0.0013, +0.0138] | WITHIN | 0.4734 | -0.0025 | +0.0257 |
| hotpotqa | in-domain | 7405 | 0.9041 | 0.9076 | +0.0034 [+0.0004, +0.0066] | WITHIN | 0.6846 | +0.0049 | -0.0007 |
| 2wiki | in-domain | 12576 | 0.8687 | 0.9311 | +0.0623 [+0.0591, +0.0654] | GAIN | 0.6079 | +0.1225 | +0.0707 |
| webqsp | zero-shot | 1503 | 0.2912 | 0.3633 | +0.0721 [+0.0562, +0.0888] | GAIN | 0.0535 | +0.0599 | +0.0319 |

## Against outputs\step1\fits\L-squad (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6535 | 0.8015 | +0.1481 [+0.1426, +0.1537] | GAIN | 0.0047 | +0.1458 | +0.2735 |
| squad | zero-shot | 11873 | 0.8797 | 0.8884 | +0.0087 [+0.0053, +0.0120] | GAIN | 0.9054 | +0.0087 | +0.0233 |
| musique | in-domain | 2417 | 0.5743 | 0.5771 | +0.0028 [-0.0052, +0.0104] | WITHIN | 0.4734 | -0.0025 | +0.0232 |
| hotpotqa | in-domain | 7405 | 0.9045 | 0.9076 | +0.0031 [-0.0001, +0.0063] | WITHIN | 0.6846 | +0.0050 | +0.0197 |
| 2wiki | in-domain | 12576 | 0.8724 | 0.9311 | +0.0587 [+0.0556, +0.0617] | GAIN | 0.6079 | +0.1161 | +0.0740 |
| webqsp | zero-shot | 1503 | 0.1302 | 0.3633 | +0.2331 [+0.2079, +0.2580] | GAIN | 0.0535 | +0.1876 | +0.0805 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
