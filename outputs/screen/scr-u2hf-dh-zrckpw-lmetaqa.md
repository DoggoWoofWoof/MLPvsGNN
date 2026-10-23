# Screen scr-u2hf-zrckpw-lmetaqa (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-u2hf-base-zrckp-lmetaqa (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | zero-shot | 9785 | 0.2463 | 0.2435 | -0.0028 [-0.0070, +0.0014] | WITHIN | 0.0047 | -0.0029 | +0.0013 |
| squad | in-domain | 11873 | 0.9095 | 0.9122 | +0.0026 [+0.0007, +0.0046] | WITHIN | 0.9054 | +0.0026 | +0.0001 |
| musique | in-domain | 2417 | 0.5757 | 0.5755 | -0.0002 [-0.0055, +0.0050] | WITHIN | 0.4734 | -0.0021 | +0.0029 |
| hotpotqa | in-domain | 7405 | 0.9063 | 0.9063 | +0.0000 [-0.0020, +0.0023] | WITHIN | 0.6846 | +0.0000 | -0.0016 |
| 2wiki | in-domain | 12576 | 0.9286 | 0.9345 | +0.0059 [+0.0043, +0.0076] | WITHIN | 0.6079 | +0.0152 | -0.0001 |
| webqsp | zero-shot | 1503 | 0.1152 | 0.1240 | +0.0087 [+0.0014, +0.0166] | GAIN | 0.0535 | +0.0067 | +0.0060 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
