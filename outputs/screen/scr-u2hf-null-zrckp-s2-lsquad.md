# Screen scr-u2hf-base-zrckp-s2-lsquad (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-u2hf-base-zrckp-lsquad (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8025 | 0.8042 | +0.0017 [+0.0006, +0.0028] | WITHIN | 0.0047 | +0.0017 | +0.0101 |
| squad | zero-shot | 11873 | 0.8940 | 0.8954 | +0.0013 [-0.0011, +0.0039] | WITHIN | 0.9054 | +0.0013 | -0.0020 |
| musique | in-domain | 2417 | 0.5803 | 0.5774 | -0.0029 [-0.0084, +0.0024] | WITHIN | 0.4734 | -0.0046 | -0.0137 |
| hotpotqa | in-domain | 7405 | 0.9065 | 0.9099 | +0.0034 [+0.0010, +0.0055] | WITHIN | 0.6846 | +0.0068 | -0.0072 |
| 2wiki | in-domain | 12576 | 0.9324 | 0.9349 | +0.0025 [+0.0010, +0.0041] | WITHIN | 0.6079 | +0.0078 | +0.0037 |
| webqsp | zero-shot | 1503 | 0.3588 | 0.3416 | -0.0171 [-0.0283, -0.0064] | LOSS | 0.0535 | -0.0180 | -0.0166 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
