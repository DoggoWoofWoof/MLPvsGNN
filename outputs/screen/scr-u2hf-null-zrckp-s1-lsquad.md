# Screen scr-u2hf-base-zrckp-s1-lsquad (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-u2hf-base-zrckp-lsquad (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8025 | 0.8034 | +0.0009 [-0.0002, +0.0021] | WITHIN | 0.0047 | +0.0016 | +0.0042 |
| squad | zero-shot | 11873 | 0.8940 | 0.8955 | +0.0014 [-0.0011, +0.0040] | WITHIN | 0.9054 | +0.0014 | -0.0057 |
| musique | in-domain | 2417 | 0.5803 | 0.5804 | +0.0001 [-0.0056, +0.0057] | WITHIN | 0.4734 | +0.0017 | -0.0128 |
| hotpotqa | in-domain | 7405 | 0.9065 | 0.9097 | +0.0032 [+0.0010, +0.0053] | WITHIN | 0.6846 | +0.0059 | +0.0009 |
| 2wiki | in-domain | 12576 | 0.9324 | 0.9348 | +0.0024 [+0.0009, +0.0040] | WITHIN | 0.6079 | +0.0080 | -0.0021 |
| webqsp | zero-shot | 1503 | 0.3588 | 0.3554 | -0.0034 [-0.0140, +0.0073] | WITHIN | 0.0535 | +0.0000 | -0.0100 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
