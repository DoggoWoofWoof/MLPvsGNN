# Screen scr-hubwalk-hp (p@swa, s1eval carves)

## Against outputs\step1\fits\L-hotpotqa (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6540 | 0.6467 | -0.0073 [-0.0106, -0.0042] | WITHIN | 0.0047 | -0.0057 | -0.0085 |
| squad | in-domain | 11873 | 0.9116 | 0.9119 | +0.0003 [-0.0018, +0.0024] | WITHIN | 0.9054 | +0.0003 | -0.0008 |
| musique | in-domain | 2417 | 0.5550 | 0.5585 | +0.0035 [-0.0024, +0.0101] | WITHIN | 0.4734 | +0.0087 | -0.0012 |
| hotpotqa | zero-shot | 7405 | 0.8585 | 0.8431 | -0.0155 [-0.0191, -0.0117] | LOSS | 0.6846 | -0.0307 | +0.0042 |
| 2wiki | in-domain | 12576 | 0.8672 | 0.8646 | -0.0026 [-0.0049, -0.0003] | WITHIN | 0.6079 | -0.0088 | +0.0017 |
| webqsp | zero-shot | 1503 | 0.1200 | 0.1147 | -0.0053 [-0.0149, +0.0043] | WITHIN | 0.0535 | -0.0053 | +0.0033 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
