# Screen scr-a6ex-zspxl-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-g5-zspx-hp (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8039 | 0.8043 | +0.0003 [-0.0007, +0.0013] | WITHIN | 0.0047 | -0.0008 | +0.0014 |
| squad | in-domain | 11873 | 0.9129 | 0.9150 | +0.0021 [+0.0001, +0.0041] | WITHIN | 0.9054 | +0.0021 | +0.0036 |
| musique | in-domain | 2417 | 0.5901 | 0.5903 | +0.0002 [-0.0052, +0.0052] | WITHIN | 0.4734 | -0.0021 | -0.0132 |
| hotpotqa | zero-shot | 7405 | 0.8547 | 0.8550 | +0.0003 [-0.0026, +0.0034] | WITHIN | 0.6846 | -0.0012 | +0.0009 |
| 2wiki | in-domain | 12576 | 0.9312 | 0.9351 | +0.0038 [+0.0023, +0.0053] | WITHIN | 0.6079 | +0.0114 | -0.0012 |
| webqsp | zero-shot | 1503 | 0.2914 | 0.2836 | -0.0078 [-0.0178, +0.0013] | WITHIN | 0.0535 | -0.0073 | -0.0013 |

## Against outputs\step1\fits\L-hotpotqa (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6540 | 0.8043 | +0.1503 [+0.1445, +0.1556] | GAIN | 0.0047 | +0.1540 | +0.2939 |
| squad | in-domain | 11873 | 0.9116 | 0.9150 | +0.0034 [+0.0011, +0.0057] | WITHIN | 0.9054 | +0.0034 | +0.0055 |
| musique | in-domain | 2417 | 0.5550 | 0.5903 | +0.0353 [+0.0277, +0.0432] | GAIN | 0.4734 | +0.0401 | +0.0418 |
| hotpotqa | zero-shot | 7405 | 0.8585 | 0.8550 | -0.0035 [-0.0085, +0.0013] | WITHIN | 0.6846 | -0.0078 | -0.0228 |
| 2wiki | in-domain | 12576 | 0.8672 | 0.9351 | +0.0678 [+0.0645, +0.0715] | GAIN | 0.6079 | +0.1425 | +0.0736 |
| webqsp | zero-shot | 1503 | 0.1200 | 0.2836 | +0.1636 [+0.1385, +0.1870] | GAIN | 0.0535 | +0.1198 | +0.0898 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
