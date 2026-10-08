# Screen L-squad (p@swa, s1eval carves)

## Against outputs\full_zrs\fits\L-squad (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7631 | 0.7792 | +0.0162 [+0.0139, +0.0184] | GAIN | 0.0047 | +0.0201 | +0.0445 |
| squad | zero-shot | 11873 | 0.8853 | 0.8903 | +0.0051 [+0.0024, +0.0077] | WITHIN | 0.9054 | +0.0051 | +0.0034 |
| musique | in-domain | 2417 | 0.5650 | 0.5690 | +0.0040 [-0.0017, +0.0097] | WITHIN | 0.4734 | +0.0041 | +0.0017 |
| hotpotqa | in-domain | 7405 | 0.9028 | 0.9042 | +0.0014 [-0.0011, +0.0039] | WITHIN | 0.6846 | +0.0046 | +0.0070 |
| 2wiki | in-domain | 12576 | 0.8676 | 0.8689 | +0.0013 [-0.0008, +0.0034] | WITHIN | 0.6079 | +0.0047 | +0.0031 |
| webqsp | zero-shot | 1503 | 0.2264 | 0.2505 | +0.0242 [+0.0104, +0.0381] | GAIN | 0.0535 | +0.0200 | +0.0140 |

## Against outputs\full_zret\fits\L-squad (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6472 | 0.7792 | +0.1321 [+0.1268, +0.1374] | GAIN | 0.0047 | +0.1370 | +0.2569 |
| squad | zero-shot | 11873 | 0.8853 | 0.8903 | +0.0051 [+0.0024, +0.0077] | WITHIN | 0.9054 | +0.0051 | +0.0034 |
| musique | in-domain | 2417 | 0.5650 | 0.5690 | +0.0040 [-0.0017, +0.0097] | WITHIN | 0.4734 | +0.0041 | +0.0017 |
| hotpotqa | in-domain | 7405 | 0.9028 | 0.9042 | +0.0014 [-0.0011, +0.0039] | WITHIN | 0.6846 | +0.0046 | +0.0070 |
| 2wiki | in-domain | 12576 | 0.8676 | 0.8689 | +0.0013 [-0.0008, +0.0034] | WITHIN | 0.6079 | +0.0047 | +0.0031 |
| webqsp | zero-shot | 1503 | 0.1469 | 0.2505 | +0.1036 [+0.0818, +0.1254] | GAIN | 0.0535 | +0.0805 | +0.0692 |

## Against outputs\step1\fits\L-squad (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6535 | 0.7792 | +0.1258 [+0.1206, +0.1312] | GAIN | 0.0047 | +0.1267 | +0.2529 |
| squad | zero-shot | 11873 | 0.8797 | 0.8903 | +0.0106 [+0.0074, +0.0139] | GAIN | 0.9054 | +0.0106 | +0.0230 |
| musique | in-domain | 2417 | 0.5743 | 0.5690 | -0.0053 [-0.0117, +0.0009] | WITHIN | 0.4734 | -0.0070 | -0.0062 |
| hotpotqa | in-domain | 7405 | 0.9045 | 0.9042 | -0.0003 [-0.0030, +0.0024] | WITHIN | 0.6846 | +0.0018 | +0.0211 |
| 2wiki | in-domain | 12576 | 0.8724 | 0.8689 | -0.0035 [-0.0057, -0.0014] | WITHIN | 0.6079 | -0.0038 | +0.0080 |
| webqsp | zero-shot | 1503 | 0.1302 | 0.2505 | +0.1203 [+0.0995, +0.1413] | GAIN | 0.0535 | +0.0938 | +0.0659 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
