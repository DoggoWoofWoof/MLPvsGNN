# Screen L-squad (p@swa, s1eval carves)

## Against outputs\full_zrct\fits\L-squad (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7799 | 0.8018 | +0.0219 [+0.0190, +0.0246] | GAIN | 0.0047 | +0.0175 | +0.0246 |
| squad | zero-shot | 11873 | 0.8912 | 0.8910 | -0.0002 [-0.0030, +0.0028] | WITHIN | 0.9054 | -0.0002 | -0.0003 |
| musique | in-domain | 2417 | 0.5710 | 0.5776 | +0.0067 [-0.0010, +0.0143] | WITHIN | 0.4734 | -0.0041 | +0.0128 |
| hotpotqa | in-domain | 7405 | 0.9041 | 0.9086 | +0.0045 [+0.0015, +0.0076] | WITHIN | 0.6846 | +0.0073 | -0.0163 |
| 2wiki | in-domain | 12576 | 0.8687 | 0.9330 | +0.0643 [+0.0611, +0.0675] | GAIN | 0.6079 | +0.1310 | +0.0646 |
| webqsp | zero-shot | 1503 | 0.2912 | 0.3567 | +0.0655 [+0.0485, +0.0822] | GAIN | 0.0535 | +0.0566 | +0.0353 |

## Against outputs\full_zrck\fits\L-squad (reported only): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8015 | 0.8018 | +0.0003 [-0.0009, +0.0014] | WITHIN | 0.0047 | -0.0003 | +0.0034 |
| squad | zero-shot | 11873 | 0.8884 | 0.8910 | +0.0026 [+0.0002, +0.0048] | WITHIN | 0.9054 | +0.0026 | +0.0006 |
| musique | in-domain | 2417 | 0.5771 | 0.5776 | +0.0006 [-0.0054, +0.0064] | WITHIN | 0.4734 | -0.0017 | -0.0128 |
| hotpotqa | in-domain | 7405 | 0.9076 | 0.9086 | +0.0010 [-0.0011, +0.0031] | WITHIN | 0.6846 | +0.0024 | -0.0157 |
| 2wiki | in-domain | 12576 | 0.9311 | 0.9330 | +0.0019 [+0.0004, +0.0036] | WITHIN | 0.6079 | +0.0085 | -0.0060 |
| webqsp | zero-shot | 1503 | 0.3633 | 0.3567 | -0.0066 [-0.0175, +0.0038] | WITHIN | 0.0535 | -0.0033 | +0.0033 |

## Against outputs\step1\fits\L-squad (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6535 | 0.8018 | +0.1483 [+0.1430, +0.1539] | GAIN | 0.0047 | +0.1455 | +0.2769 |
| squad | zero-shot | 11873 | 0.8797 | 0.8910 | +0.0113 [+0.0079, +0.0147] | GAIN | 0.9054 | +0.0113 | +0.0239 |
| musique | in-domain | 2417 | 0.5743 | 0.5776 | +0.0033 [-0.0042, +0.0113] | WITHIN | 0.4734 | -0.0041 | +0.0103 |
| hotpotqa | in-domain | 7405 | 0.9045 | 0.9086 | +0.0041 [+0.0009, +0.0074] | WITHIN | 0.6846 | +0.0074 | +0.0041 |
| 2wiki | in-domain | 12576 | 0.8724 | 0.9330 | +0.0607 [+0.0575, +0.0636] | GAIN | 0.6079 | +0.1246 | +0.0679 |
| webqsp | zero-shot | 1503 | 0.1302 | 0.3567 | +0.2264 [+0.2009, +0.2507] | GAIN | 0.0535 | +0.1843 | +0.0838 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
