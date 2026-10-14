# Screen scr-p1c3-zspk (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-zspg (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7794 | 0.8009 | +0.0215 [+0.0185, +0.0243] | GAIN | 0.0047 | +0.0178 | +0.0214 |
| squad | in-domain | 11873 | 0.9125 | 0.9114 | -0.0011 [-0.0032, +0.0012] | WITHIN | 0.9054 | -0.0011 | +0.0017 |
| musique | zero-shot | 2417 | 0.5111 | 0.5012 | -0.0100 [-0.0175, -0.0024] | LOSS | 0.4734 | -0.0207 | +0.0314 |
| hotpotqa | in-domain | 7405 | 0.9065 | 0.9080 | +0.0015 [-0.0016, +0.0047] | WITHIN | 0.6846 | +0.0018 | -0.0081 |
| 2wiki | in-domain | 12576 | 0.8724 | 0.9383 | +0.0659 [+0.0625, +0.0692] | GAIN | 0.6079 | +0.1333 | +0.0745 |
| webqsp | zero-shot | 1503 | 0.3245 | 0.3229 | -0.0017 [-0.0190, +0.0152] | WITHIN | 0.0535 | -0.0060 | +0.0133 |

## Against outputs\screen\fits\scr-p1c2-zsp (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8008 | 0.8009 | +0.0001 [-0.0011, +0.0012] | WITHIN | 0.0047 | -0.0010 | -0.0080 |
| squad | in-domain | 11873 | 0.9119 | 0.9114 | -0.0005 [-0.0026, +0.0016] | WITHIN | 0.9054 | -0.0005 | +0.0009 |
| musique | zero-shot | 2417 | 0.4792 | 0.5012 | +0.0220 [+0.0143, +0.0297] | GAIN | 0.4734 | +0.0257 | +0.1655 |
| hotpotqa | in-domain | 7405 | 0.9062 | 0.9080 | +0.0018 [-0.0008, +0.0044] | WITHIN | 0.6846 | +0.0020 | -0.0139 |
| 2wiki | in-domain | 12576 | 0.9174 | 0.9383 | +0.0209 [+0.0187, +0.0231] | GAIN | 0.6079 | +0.0472 | +0.0345 |
| webqsp | zero-shot | 1503 | 0.3078 | 0.3229 | +0.0150 [+0.0045, +0.0257] | GAIN | 0.0535 | +0.0100 | +0.0266 |

## Against outputs\step1\fits\L-musique (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6537 | 0.8009 | +0.1471 [+0.1415, +0.1525] | GAIN | 0.0047 | +0.1453 | +0.2749 |
| squad | in-domain | 11873 | 0.9100 | 0.9114 | +0.0014 [-0.0008, +0.0038] | WITHIN | 0.9054 | +0.0014 | -0.0015 |
| musique | zero-shot | 2417 | 0.2696 | 0.5012 | +0.2316 [+0.2198, +0.2432] | GAIN | 0.4734 | +0.1163 | +0.3868 |
| hotpotqa | in-domain | 7405 | 0.9015 | 0.9080 | +0.0065 [+0.0033, +0.0099] | WITHIN | 0.6846 | +0.0099 | +0.0050 |
| 2wiki | in-domain | 12576 | 0.8732 | 0.9383 | +0.0650 [+0.0619, +0.0683] | GAIN | 0.6079 | +0.1354 | +0.0723 |
| webqsp | zero-shot | 1503 | 0.1779 | 0.3229 | +0.1450 [+0.1216, +0.1699] | GAIN | 0.0535 | +0.1104 | +0.1277 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
