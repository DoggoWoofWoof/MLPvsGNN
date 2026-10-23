# Screen scr-u2hf-base-zspk-s2-lmetaqa (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-u2hf-base-zspk-lmetaqa (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | zero-shot | 9785 | 0.2135 | 0.2141 | +0.0006 [-0.0036, +0.0047] | WITHIN | 0.0047 | +0.0043 | +0.0086 |
| squad | in-domain | 11873 | 0.9114 | 0.9111 | -0.0003 [-0.0024, +0.0018] | WITHIN | 0.9054 | -0.0003 | -0.0052 |
| musique | in-domain | 2417 | 0.5831 | 0.5817 | -0.0014 [-0.0069, +0.0040] | WITHIN | 0.4734 | -0.0037 | +0.0029 |
| hotpotqa | in-domain | 7405 | 0.9108 | 0.9094 | -0.0014 [-0.0036, +0.0006] | WITHIN | 0.6846 | -0.0019 | -0.0085 |
| 2wiki | in-domain | 12576 | 0.9373 | 0.9380 | +0.0008 [-0.0006, +0.0021] | WITHIN | 0.6079 | +0.0017 | -0.0085 |
| webqsp | zero-shot | 1503 | 0.1102 | 0.1142 | +0.0040 [-0.0032, +0.0112] | WITHIN | 0.0535 | +0.0027 | +0.0040 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
