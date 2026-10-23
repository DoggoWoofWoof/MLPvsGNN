# Screen scr-u2hf-zspkw-lmetaqa (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-u2hf-base-zspk-lmetaqa (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | zero-shot | 9785 | 0.2135 | 0.2310 | +0.0175 [+0.0132, +0.0218] | GAIN | 0.0047 | +0.0189 | +0.0009 |
| squad | in-domain | 11873 | 0.9114 | 0.9100 | -0.0013 [-0.0032, +0.0007] | WITHIN | 0.9054 | -0.0013 | -0.0023 |
| musique | in-domain | 2417 | 0.5831 | 0.5869 | +0.0038 [-0.0014, +0.0094] | WITHIN | 0.4734 | +0.0050 | +0.0074 |
| hotpotqa | in-domain | 7405 | 0.9108 | 0.9108 | +0.0000 [-0.0022, +0.0023] | WITHIN | 0.6846 | +0.0000 | -0.0039 |
| 2wiki | in-domain | 12576 | 0.9373 | 0.9361 | -0.0012 [-0.0027, +0.0003] | WITHIN | 0.6079 | -0.0044 | -0.0035 |
| webqsp | zero-shot | 1503 | 0.1102 | 0.1167 | +0.0065 [-0.0016, +0.0146] | WITHIN | 0.0535 | +0.0047 | +0.0040 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
