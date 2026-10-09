# Screen scr-zgf (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-zrm (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7789 | 0.7813 | +0.0023 [+0.0010, +0.0036] | WITHIN | 0.0047 | +0.0018 | +0.0039 |
| squad | in-domain | 11873 | 0.9111 | 0.9075 | -0.0035 [-0.0058, -0.0012] | WITHIN | 0.9054 | -0.0035 | -0.0042 |
| musique | zero-shot | 2417 | 0.5231 | 0.5291 | +0.0060 [+0.0003, +0.0115] | WITHIN | 0.4734 | +0.0074 | -0.0368 |
| hotpotqa | in-domain | 7405 | 0.9040 | 0.9024 | -0.0016 [-0.0042, +0.0010] | WITHIN | 0.6846 | -0.0011 | -0.0007 |
| 2wiki | in-domain | 12576 | 0.8709 | 0.8752 | +0.0043 [+0.0024, +0.0063] | WITHIN | 0.6079 | +0.0110 | -0.0162 |
| webqsp | zero-shot | 1503 | 0.3128 | 0.3118 | -0.0009 [-0.0103, +0.0087] | WITHIN | 0.0535 | -0.0033 | -0.0067 |

## Against outputs\screen\fits\scr-zret (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6496 | 0.7813 | +0.1316 [+0.1263, +0.1369] | GAIN | 0.0047 | +0.1347 | +0.2571 |
| squad | in-domain | 11873 | 0.9106 | 0.9075 | -0.0031 [-0.0056, -0.0008] | WITHIN | 0.9054 | -0.0031 | -0.0019 |
| musique | zero-shot | 2417 | 0.3897 | 0.5291 | +0.1394 [+0.1294, +0.1490] | GAIN | 0.4734 | +0.0890 | +0.1936 |
| hotpotqa | in-domain | 7405 | 0.9061 | 0.9024 | -0.0037 [-0.0063, -0.0011] | WITHIN | 0.6846 | -0.0057 | -0.0053 |
| 2wiki | in-domain | 12576 | 0.8714 | 0.8752 | +0.0038 [+0.0017, +0.0058] | WITHIN | 0.6079 | +0.0099 | -0.0225 |
| webqsp | zero-shot | 1503 | 0.1766 | 0.3118 | +0.1352 [+0.1126, +0.1581] | GAIN | 0.0535 | +0.1058 | +0.1045 |

## Against outputs\step1\fits\L-musique (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6537 | 0.7813 | +0.1275 [+0.1224, +0.1327] | GAIN | 0.0047 | +0.1292 | +0.2560 |
| squad | in-domain | 11873 | 0.9100 | 0.9075 | -0.0024 [-0.0050, +0.0000] | WITHIN | 0.9054 | -0.0024 | -0.0059 |
| musique | zero-shot | 2417 | 0.2696 | 0.5291 | +0.2596 [+0.2480, +0.2706] | GAIN | 0.4734 | +0.1572 | +0.3475 |
| hotpotqa | in-domain | 7405 | 0.9015 | 0.9024 | +0.0009 [-0.0019, +0.0036] | WITHIN | 0.6846 | +0.0019 | -0.0019 |
| 2wiki | in-domain | 12576 | 0.8732 | 0.8752 | +0.0020 [+0.0000, +0.0040] | WITHIN | 0.6079 | +0.0044 | -0.0184 |
| webqsp | zero-shot | 1503 | 0.1779 | 0.3118 | +0.1340 [+0.1120, +0.1560] | GAIN | 0.0535 | +0.1065 | +0.0991 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
