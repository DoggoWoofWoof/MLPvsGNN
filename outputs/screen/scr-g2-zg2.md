# Screen scr-g2-zg2 (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-p1c3-zspk (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8009 | 0.8027 | +0.0018 [+0.0009, +0.0028] | WITHIN | 0.0047 | +0.0020 | +0.0110 |
| squad | in-domain | 11873 | 0.9114 | 0.9106 | -0.0008 [-0.0028, +0.0009] | WITHIN | 0.9054 | -0.0008 | +0.0023 |
| musique | zero-shot | 2417 | 0.5012 | 0.5110 | +0.0099 [+0.0046, +0.0153] | GAIN | 0.4734 | +0.0112 | +0.0054 |
| hotpotqa | in-domain | 7405 | 0.9080 | 0.9099 | +0.0018 [-0.0003, +0.0039] | WITHIN | 0.6846 | +0.0022 | +0.0032 |
| 2wiki | in-domain | 12576 | 0.9383 | 0.9348 | -0.0035 [-0.0048, -0.0021] | WITHIN | 0.6079 | -0.0085 | -0.0056 |
| webqsp | zero-shot | 1503 | 0.3229 | 0.3188 | -0.0040 [-0.0128, +0.0046] | WITHIN | 0.0535 | -0.0027 | -0.0053 |

## Against outputs\screen\fits\scr-g2-zg2t (reported only): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8041 | 0.8027 | -0.0013 [-0.0022, -0.0005] | WITHIN | 0.0047 | -0.0017 | -0.0042 |
| squad | in-domain | 11873 | 0.9109 | 0.9106 | -0.0003 [-0.0023, +0.0016] | WITHIN | 0.9054 | -0.0003 | +0.0049 |
| musique | zero-shot | 2417 | 0.5252 | 0.5110 | -0.0141 [-0.0194, -0.0089] | LOSS | 0.4734 | -0.0108 | -0.0079 |
| hotpotqa | in-domain | 7405 | 0.9103 | 0.9099 | -0.0004 [-0.0024, +0.0016] | WITHIN | 0.6846 | +0.0003 | -0.0061 |
| 2wiki | in-domain | 12576 | 0.9382 | 0.9348 | -0.0034 [-0.0048, -0.0019] | WITHIN | 0.6079 | -0.0103 | +0.0096 |
| webqsp | zero-shot | 1503 | 0.3057 | 0.3188 | +0.0131 [+0.0048, +0.0214] | GAIN | 0.0535 | +0.0133 | +0.0140 |

## Against outputs\screen\fits\scr-zspg (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7794 | 0.8027 | +0.0233 [+0.0203, +0.0261] | GAIN | 0.0047 | +0.0198 | +0.0324 |
| squad | in-domain | 11873 | 0.9125 | 0.9106 | -0.0019 [-0.0039, +0.0002] | WITHIN | 0.9054 | -0.0019 | +0.0040 |
| musique | zero-shot | 2417 | 0.5111 | 0.5110 | -0.0001 [-0.0084, +0.0075] | WITHIN | 0.4734 | -0.0095 | +0.0368 |
| hotpotqa | in-domain | 7405 | 0.9065 | 0.9099 | +0.0033 [+0.0002, +0.0064] | WITHIN | 0.6846 | +0.0039 | -0.0049 |
| 2wiki | in-domain | 12576 | 0.8724 | 0.9348 | +0.0624 [+0.0590, +0.0656] | GAIN | 0.6079 | +0.1248 | +0.0689 |
| webqsp | zero-shot | 1503 | 0.3245 | 0.3188 | -0.0057 [-0.0230, +0.0115] | WITHIN | 0.0535 | -0.0086 | +0.0080 |

## Against outputs\step1\fits\L-musique (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6537 | 0.8027 | +0.1490 [+0.1433, +0.1542] | GAIN | 0.0047 | +0.1474 | +0.2859 |
| squad | in-domain | 11873 | 0.9100 | 0.9106 | +0.0007 [-0.0016, +0.0030] | WITHIN | 0.9054 | +0.0007 | +0.0008 |
| musique | zero-shot | 2417 | 0.2696 | 0.5110 | +0.2414 [+0.2295, +0.2530] | GAIN | 0.4734 | +0.1274 | +0.3922 |
| hotpotqa | in-domain | 7405 | 0.9015 | 0.9099 | +0.0084 [+0.0050, +0.0116] | GAIN | 0.6846 | +0.0120 | +0.0082 |
| 2wiki | in-domain | 12576 | 0.8732 | 0.9348 | +0.0616 [+0.0584, +0.0648] | GAIN | 0.6079 | +0.1269 | +0.0666 |
| webqsp | zero-shot | 1503 | 0.1779 | 0.3188 | +0.1410 [+0.1180, +0.1655] | GAIN | 0.0535 | +0.1078 | +0.1224 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
