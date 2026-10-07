# Screen scr-pools (p@swa, s1eval carves)

## Against outputs\step1\fits\L-musique (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6537 | 0.6564 | +0.0027 [-0.0005, +0.0059] | WITHIN | 0.0047 | +0.0006 | +0.0108 |
| squad | in-domain | 11873 | 0.9100 | 0.9106 | +0.0007 [-0.0014, +0.0028] | WITHIN | 0.9054 | +0.0007 | -0.0009 |
| musique | zero-shot | 2417 | 0.2696 | 0.2899 | +0.0203 [+0.0122, +0.0284] | GAIN | 0.4734 | +0.0021 | +0.0050 |
| hotpotqa | in-domain | 7405 | 0.9015 | 0.9035 | +0.0020 [-0.0003, +0.0044] | WITHIN | 0.6846 | +0.0012 | +0.0086 |
| 2wiki | in-domain | 12576 | 0.8732 | 0.8728 | -0.0005 [-0.0024, +0.0014] | WITHIN | 0.6079 | -0.0001 | +0.0027 |
| webqsp | zero-shot | 1503 | 0.1779 | 0.1578 | -0.0200 [-0.0320, -0.0084] | LOSS | 0.0535 | -0.0146 | -0.0126 |

## Against outputs\step4c\screen\L-musique (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6456 | 0.6564 | +0.0108 [+0.0078, +0.0138] | GAIN | 0.0047 | +0.0063 | +0.0249 |
| squad | in-domain | 11873 | 0.9100 | 0.9106 | +0.0007 [-0.0014, +0.0028] | WITHIN | 0.9054 | +0.0007 | -0.0009 |
| musique | zero-shot | 2417 | 0.2882 | 0.2899 | +0.0017 [-0.0053, +0.0090] | WITHIN | 0.4734 | -0.0012 | -0.0025 |
| hotpotqa | in-domain | 7405 | 0.9014 | 0.9035 | +0.0022 [-0.0002, +0.0045] | WITHIN | 0.6846 | +0.0014 | +0.0096 |
| 2wiki | in-domain | 12576 | 0.8730 | 0.8728 | -0.0003 [-0.0022, +0.0016] | WITHIN | 0.6079 | +0.0005 | +0.0025 |
| webqsp | zero-shot | 1503 | 0.1661 | 0.1578 | -0.0083 [-0.0179, +0.0012] | WITHIN | 0.0535 | -0.0067 | -0.0080 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
