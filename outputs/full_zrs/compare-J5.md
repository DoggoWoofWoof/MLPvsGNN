# Screen J5 (p@swa, s1eval carves)

## Against outputs\full_zret\fits\J5 (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6431 | 0.7636 | +0.1205 [+0.1153, +0.1254] | GAIN | 0.0047 | +0.1195 | +0.2052 |
| squad | in-domain | 11873 | 0.9109 | 0.9109 | +0.0000 [+0.0000, +0.0000] | WITHIN | 0.9054 | +0.0000 | +0.0000 |
| musique | in-domain | 2417 | 0.5640 | 0.5640 | +0.0000 [+0.0000, +0.0000] | WITHIN | 0.4734 | +0.0000 | +0.0000 |
| hotpotqa | in-domain | 7405 | 0.9008 | 0.9008 | +0.0000 [+0.0000, +0.0000] | WITHIN | 0.6846 | +0.0000 | +0.0000 |
| 2wiki | in-domain | 12576 | 0.8659 | 0.8659 | +0.0000 [+0.0000, +0.0000] | WITHIN | 0.6079 | +0.0000 | +0.0000 |
| webqsp | zero-shot | 1503 | 0.1417 | 0.2329 | +0.0912 [+0.0723, +0.1105] | GAIN | 0.0535 | +0.0692 | +0.0532 |

## Against outputs\full_rmatch\fits\J5 (reported only): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7788 | 0.7636 | -0.0152 [-0.0174, -0.0130] | LOSS | 0.0047 | -0.0206 | -0.0625 |
| squad | in-domain | 11873 | 0.9075 | 0.9109 | +0.0034 [+0.0010, +0.0057] | WITHIN | 0.9054 | +0.0034 | +0.0011 |
| musique | in-domain | 2417 | 0.5685 | 0.5640 | -0.0046 [-0.0104, +0.0014] | WITHIN | 0.4734 | -0.0050 | -0.0025 |
| hotpotqa | in-domain | 7405 | 0.9026 | 0.9008 | -0.0018 [-0.0046, +0.0009] | WITHIN | 0.6846 | -0.0043 | +0.0005 |
| 2wiki | in-domain | 12576 | 0.8730 | 0.8659 | -0.0071 [-0.0092, -0.0050] | WITHIN | 0.6079 | -0.0125 | +0.0025 |
| webqsp | zero-shot | 1503 | 0.2795 | 0.2329 | -0.0466 [-0.0630, -0.0311] | LOSS | 0.0535 | -0.0353 | -0.0333 |

## Against outputs\step1\fits\J5 (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6443 | 0.7636 | +0.1193 [+0.1145, +0.1244] | GAIN | 0.0047 | +0.1203 | +0.1914 |
| squad | in-domain | 11873 | 0.9095 | 0.9109 | +0.0013 [-0.0010, +0.0036] | WITHIN | 0.9054 | +0.0013 | -0.0019 |
| musique | in-domain | 2417 | 0.5609 | 0.5640 | +0.0031 [-0.0033, +0.0095] | WITHIN | 0.4734 | +0.0112 | +0.0037 |
| hotpotqa | in-domain | 7405 | 0.9013 | 0.9008 | -0.0005 [-0.0034, +0.0024] | WITHIN | 0.6846 | -0.0014 | +0.0049 |
| 2wiki | in-domain | 12576 | 0.8694 | 0.8659 | -0.0035 [-0.0055, -0.0012] | WITHIN | 0.6079 | -0.0072 | +0.0009 |
| webqsp | zero-shot | 1503 | 0.1167 | 0.2329 | +0.1162 [+0.0969, +0.1353] | GAIN | 0.0535 | +0.0925 | +0.0559 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
