# Screen scr-a4he-zspkhe (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-p1c3-zspk (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8009 | 0.8022 | +0.0013 [+0.0004, +0.0022] | WITHIN | 0.0047 | +0.0012 | +0.0004 |
| squad | in-domain | 11873 | 0.9114 | 0.9093 | -0.0021 [-0.0040, -0.0003] | WITHIN | 0.9054 | -0.0021 | -0.0021 |
| musique | zero-shot | 2417 | 0.5012 | 0.3976 | -0.1036 [-0.1133, -0.0937] | LOSS | 0.4734 | -0.0633 | -0.2362 |
| hotpotqa | in-domain | 7405 | 0.9080 | 0.9072 | -0.0008 [-0.0029, +0.0012] | WITHIN | 0.6846 | -0.0046 | +0.0008 |
| 2wiki | in-domain | 12576 | 0.9383 | 0.9367 | -0.0016 [-0.0030, -0.0002] | WITHIN | 0.6079 | -0.0021 | +0.0013 |
| webqsp | zero-shot | 1503 | 0.3229 | 0.3190 | -0.0039 [-0.0130, +0.0051] | WITHIN | 0.0535 | -0.0047 | -0.0100 |

## Against outputs\step1\fits\L-musique (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6537 | 0.8022 | +0.1484 [+0.1427, +0.1538] | GAIN | 0.0047 | +0.1466 | +0.2753 |
| squad | in-domain | 11873 | 0.9100 | 0.9093 | -0.0007 [-0.0030, +0.0017] | WITHIN | 0.9054 | -0.0007 | -0.0036 |
| musique | zero-shot | 2417 | 0.2696 | 0.3976 | +0.1280 [+0.1172, +0.1391] | GAIN | 0.4734 | +0.0530 | +0.1506 |
| hotpotqa | in-domain | 7405 | 0.9015 | 0.9072 | +0.0057 [+0.0022, +0.0090] | WITHIN | 0.6846 | +0.0053 | +0.0058 |
| 2wiki | in-domain | 12576 | 0.8732 | 0.9367 | +0.0635 [+0.0602, +0.0668] | GAIN | 0.6079 | +0.1333 | +0.0736 |
| webqsp | zero-shot | 1503 | 0.1779 | 0.3190 | +0.1411 [+0.1185, +0.1663] | GAIN | 0.0535 | +0.1058 | +0.1178 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
