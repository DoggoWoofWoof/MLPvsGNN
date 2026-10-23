# Screen scr-u2hf-zrckpw-l2wiki (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-u2hf-base-zrckp-l2wiki (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8029 | 0.8034 | +0.0005 [-0.0002, +0.0013] | WITHIN | 0.0047 | +0.0007 | -0.0004 |
| squad | in-domain | 11873 | 0.9099 | 0.9100 | +0.0002 [-0.0017, +0.0021] | WITHIN | 0.9054 | +0.0002 | -0.0003 |
| musique | in-domain | 2417 | 0.5678 | 0.5687 | +0.0009 [-0.0043, +0.0060] | WITHIN | 0.4734 | -0.0066 | +0.0066 |
| hotpotqa | in-domain | 7405 | 0.9014 | 0.9005 | -0.0008 [-0.0030, +0.0012] | WITHIN | 0.6846 | -0.0030 | +0.0030 |
| 2wiki | zero-shot | 12576 | 0.8519 | 0.8496 | -0.0024 [-0.0045, -0.0003] | WITHIN | 0.6079 | -0.0044 | -0.0026 |
| webqsp | zero-shot | 1503 | 0.3580 | 0.3615 | +0.0035 [-0.0046, +0.0121] | WITHIN | 0.0535 | +0.0080 | -0.0027 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
