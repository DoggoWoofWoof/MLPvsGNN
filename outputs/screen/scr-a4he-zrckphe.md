# Screen scr-a4he-zrckphe (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-p1c3p-zrckp (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8019 | 0.8017 | -0.0002 [-0.0012, +0.0007] | WITHIN | 0.0047 | -0.0011 | +0.0026 |
| squad | in-domain | 11873 | 0.9116 | 0.9108 | -0.0008 [-0.0025, +0.0010] | WITHIN | 0.9054 | -0.0008 | -0.0019 |
| musique | zero-shot | 2417 | 0.5121 | 0.4416 | -0.0706 [-0.0784, -0.0626] | LOSS | 0.4734 | -0.0401 | -0.1324 |
| hotpotqa | in-domain | 7405 | 0.9065 | 0.9030 | -0.0036 [-0.0058, -0.0012] | WITHIN | 0.6846 | -0.0035 | -0.0054 |
| 2wiki | in-domain | 12576 | 0.9347 | 0.9275 | -0.0072 [-0.0088, -0.0054] | WITHIN | 0.6079 | -0.0187 | -0.0001 |
| webqsp | zero-shot | 1503 | 0.3627 | 0.3809 | +0.0181 [+0.0077, +0.0293] | GAIN | 0.0535 | +0.0080 | +0.0047 |

## Against outputs\step1\fits\L-musique (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6537 | 0.8017 | +0.1480 [+0.1423, +0.1532] | GAIN | 0.0047 | +0.1443 | +0.2837 |
| squad | in-domain | 11873 | 0.9100 | 0.9108 | +0.0008 [-0.0013, +0.0030] | WITHIN | 0.9054 | +0.0008 | -0.0030 |
| musique | zero-shot | 2417 | 0.2696 | 0.4416 | +0.1720 [+0.1601, +0.1843] | GAIN | 0.4734 | +0.0807 | +0.2809 |
| hotpotqa | in-domain | 7405 | 0.9015 | 0.9030 | +0.0015 [-0.0021, +0.0050] | WITHIN | 0.6846 | +0.0007 | -0.0128 |
| 2wiki | in-domain | 12576 | 0.8732 | 0.9275 | +0.0543 [+0.0511, +0.0576] | GAIN | 0.6079 | +0.1076 | +0.0726 |
| webqsp | zero-shot | 1503 | 0.1779 | 0.3809 | +0.2030 [+0.1803, +0.2268] | GAIN | 0.0535 | +0.1663 | +0.0772 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
