# Screen J5 (p@swa, s1eval carves)

## Against outputs\full_zspg\fits\J5 (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7801 | 0.8012 | +0.0211 [+0.0181, +0.0238] | GAIN | 0.0047 | +0.0172 | +0.0208 |
| squad | in-domain | 11873 | 0.9108 | 0.9099 | -0.0009 [-0.0030, +0.0012] | WITHIN | 0.9054 | -0.0009 | -0.0039 |
| musique | in-domain | 2417 | 0.5690 | 0.5800 | +0.0110 [+0.0034, +0.0185] | GAIN | 0.4734 | -0.0021 | +0.0410 |
| hotpotqa | in-domain | 7405 | 0.9055 | 0.9044 | -0.0011 [-0.0047, +0.0024] | WITHIN | 0.6846 | -0.0068 | -0.0155 |
| 2wiki | in-domain | 12576 | 0.8766 | 0.9351 | +0.0586 [+0.0553, +0.0619] | GAIN | 0.6079 | +0.1124 | +0.0804 |
| webqsp | zero-shot | 1503 | 0.2582 | 0.2913 | +0.0331 [+0.0149, +0.0514] | GAIN | 0.0535 | +0.0226 | +0.0333 |

## Against outputs\step1\fits\J5 (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6443 | 0.8012 | +0.1569 [+0.1514, +0.1626] | GAIN | 0.0047 | +0.1587 | +0.2817 |
| squad | in-domain | 11873 | 0.9095 | 0.9099 | +0.0003 [-0.0019, +0.0025] | WITHIN | 0.9054 | +0.0003 | -0.0054 |
| musique | in-domain | 2417 | 0.5609 | 0.5800 | +0.0191 [+0.0112, +0.0270] | GAIN | 0.4734 | +0.0112 | +0.0310 |
| hotpotqa | in-domain | 7405 | 0.9013 | 0.9044 | +0.0031 [-0.0005, +0.0065] | WITHIN | 0.6846 | +0.0026 | -0.0088 |
| 2wiki | in-domain | 12576 | 0.8694 | 0.9351 | +0.0657 [+0.0624, +0.0692] | GAIN | 0.6079 | +0.1316 | +0.0736 |
| webqsp | zero-shot | 1503 | 0.1167 | 0.2913 | +0.1745 [+0.1502, +0.1985] | GAIN | 0.0535 | +0.1311 | +0.1178 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
