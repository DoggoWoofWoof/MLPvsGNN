# Screen scr-u2hf-zrckpw-lsquad (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-u2hf-base-zrckp-lsquad (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8025 | 0.8001 | -0.0024 [-0.0034, -0.0013] | WITHIN | 0.0047 | -0.0026 | -0.0061 |
| squad | zero-shot | 11873 | 0.8940 | 0.8958 | +0.0018 [-0.0007, +0.0045] | WITHIN | 0.9054 | +0.0018 | -0.0050 |
| musique | in-domain | 2417 | 0.5803 | 0.5791 | -0.0012 [-0.0073, +0.0047] | WITHIN | 0.4734 | -0.0025 | -0.0219 |
| hotpotqa | in-domain | 7405 | 0.9065 | 0.9065 | +0.0000 [-0.0022, +0.0022] | WITHIN | 0.6846 | -0.0008 | -0.0016 |
| 2wiki | in-domain | 12576 | 0.9324 | 0.9327 | +0.0003 [-0.0012, +0.0019] | WITHIN | 0.6079 | -0.0003 | +0.0026 |
| webqsp | zero-shot | 1503 | 0.3588 | 0.3566 | -0.0021 [-0.0119, +0.0079] | WITHIN | 0.0535 | -0.0040 | -0.0060 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
