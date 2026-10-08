# Screen scr-zrk (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-zrm (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7789 | 0.7787 | -0.0003 [-0.0013, +0.0007] | WITHIN | 0.0047 | -0.0004 | -0.0018 |
| squad | in-domain | 11873 | 0.9111 | 0.9112 | +0.0002 [-0.0016, +0.0020] | WITHIN | 0.9054 | +0.0002 | +0.0000 |
| musique | zero-shot | 2417 | 0.5231 | 0.5238 | +0.0007 [-0.0040, +0.0054] | WITHIN | 0.4734 | -0.0033 | +0.0037 |
| hotpotqa | in-domain | 7405 | 0.9040 | 0.9045 | +0.0005 [-0.0015, +0.0026] | WITHIN | 0.6846 | +0.0020 | -0.0011 |
| 2wiki | in-domain | 12576 | 0.8709 | 0.8724 | +0.0015 [-0.0004, +0.0032] | WITHIN | 0.6079 | +0.0042 | +0.0014 |
| webqsp | zero-shot | 1503 | 0.3128 | 0.3196 | +0.0069 [-0.0020, +0.0156] | WITHIN | 0.0535 | +0.0020 | -0.0153 |

## Against outputs\screen\fits\scr-zret (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6496 | 0.7787 | +0.1290 [+0.1236, +0.1344] | GAIN | 0.0047 | +0.1324 | +0.2514 |
| squad | in-domain | 11873 | 0.9106 | 0.9112 | +0.0006 [-0.0015, +0.0028] | WITHIN | 0.9054 | +0.0006 | +0.0023 |
| musique | zero-shot | 2417 | 0.3897 | 0.5238 | +0.1341 [+0.1242, +0.1443] | GAIN | 0.4734 | +0.0782 | +0.2342 |
| hotpotqa | in-domain | 7405 | 0.9061 | 0.9045 | -0.0016 [-0.0039, +0.0008] | WITHIN | 0.6846 | -0.0026 | -0.0057 |
| 2wiki | in-domain | 12576 | 0.8714 | 0.8724 | +0.0010 [-0.0010, +0.0029] | WITHIN | 0.6079 | +0.0031 | -0.0049 |
| webqsp | zero-shot | 1503 | 0.1766 | 0.3196 | +0.1430 [+0.1204, +0.1664] | GAIN | 0.0535 | +0.1111 | +0.0958 |

## Against outputs\step1\fits\L-musique (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6537 | 0.7787 | +0.1249 [+0.1199, +0.1302] | GAIN | 0.0047 | +0.1269 | +0.2503 |
| squad | in-domain | 11873 | 0.9100 | 0.9112 | +0.0013 [-0.0008, +0.0036] | WITHIN | 0.9054 | +0.0013 | -0.0017 |
| musique | zero-shot | 2417 | 0.2696 | 0.5238 | +0.2542 [+0.2431, +0.2653] | GAIN | 0.4734 | +0.1465 | +0.3881 |
| hotpotqa | in-domain | 7405 | 0.9015 | 0.9045 | +0.0030 [+0.0005, +0.0054] | WITHIN | 0.6846 | +0.0050 | -0.0023 |
| 2wiki | in-domain | 12576 | 0.8732 | 0.8724 | -0.0008 [-0.0028, +0.0012] | WITHIN | 0.6079 | -0.0024 | -0.0008 |
| webqsp | zero-shot | 1503 | 0.1779 | 0.3196 | +0.1418 [+0.1196, +0.1654] | GAIN | 0.0535 | +0.1118 | +0.0905 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
