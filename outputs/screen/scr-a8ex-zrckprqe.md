# Screen scr-a8ex-zrckprqe (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-r1-zrckpr (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8012 | 0.7999 | -0.0014 [-0.0022, -0.0005] | WITHIN | 0.0047 | -0.0012 | -0.0055 |
| squad | in-domain | 11873 | 0.9127 | 0.9111 | -0.0017 [-0.0034, +0.0000] | WITHIN | 0.9054 | -0.0017 | +0.0035 |
| musique | zero-shot | 2417 | 0.5197 | 0.5195 | -0.0002 [-0.0047, +0.0046] | WITHIN | 0.4734 | +0.0070 | +0.0041 |
| hotpotqa | in-domain | 7405 | 0.9083 | 0.9080 | -0.0003 [-0.0022, +0.0016] | WITHIN | 0.6846 | -0.0004 | +0.0046 |
| 2wiki | in-domain | 12576 | 0.9361 | 0.9332 | -0.0029 [-0.0043, -0.0016] | WITHIN | 0.6079 | -0.0083 | -0.0044 |
| webqsp | zero-shot | 1503 | 0.3762 | 0.3828 | +0.0066 [-0.0030, +0.0160] | WITHIN | 0.0535 | +0.0033 | -0.0033 |

## Against outputs\step1\fits\L-musique (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6537 | 0.7999 | +0.1461 [+0.1405, +0.1514] | GAIN | 0.0047 | +0.1439 | +0.2776 |
| squad | in-domain | 11873 | 0.9100 | 0.9111 | +0.0011 [-0.0012, +0.0034] | WITHIN | 0.9054 | +0.0011 | +0.0005 |
| musique | zero-shot | 2417 | 0.2696 | 0.5195 | +0.2500 [+0.2382, +0.2622] | GAIN | 0.4734 | +0.1374 | +0.4179 |
| hotpotqa | in-domain | 7405 | 0.9015 | 0.9080 | +0.0065 [+0.0032, +0.0097] | WITHIN | 0.6846 | +0.0089 | -0.0030 |
| 2wiki | in-domain | 12576 | 0.8732 | 0.9332 | +0.0600 [+0.0569, +0.0632] | GAIN | 0.6079 | +0.1225 | +0.0661 |
| webqsp | zero-shot | 1503 | 0.1779 | 0.3828 | +0.2050 [+0.1836, +0.2278] | GAIN | 0.0535 | +0.1690 | +0.0798 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
