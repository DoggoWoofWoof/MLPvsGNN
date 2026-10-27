# Screen scr-a7p-zrckps-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-p1gb-zrckp-hp (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7997 | 0.8041 | +0.0044 [+0.0033, +0.0056] | WITHIN | 0.0047 | +0.0039 | +0.0149 |
| squad | in-domain | 11873 | 0.9122 | 0.9139 | +0.0017 [-0.0003, +0.0035] | WITHIN | 0.9054 | +0.0017 | +0.0012 |
| musique | in-domain | 2417 | 0.5695 | 0.5698 | +0.0003 [-0.0050, +0.0060] | WITHIN | 0.4734 | -0.0025 | -0.0070 |
| hotpotqa | zero-shot | 7405 | 0.8546 | 0.8623 | +0.0076 [+0.0045, +0.0107] | GAIN | 0.6846 | +0.0157 | -0.0032 |
| 2wiki | in-domain | 12576 | 0.9340 | 0.9333 | -0.0007 [-0.0022, +0.0009] | WITHIN | 0.6079 | -0.0036 | -0.0061 |
| webqsp | zero-shot | 1503 | 0.3364 | 0.3637 | +0.0273 [+0.0147, +0.0402] | GAIN | 0.0535 | +0.0180 | +0.0193 |

## Against outputs\step1\fits\L-hotpotqa (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6540 | 0.8041 | +0.1501 [+0.1446, +0.1555] | GAIN | 0.0047 | +0.1537 | +0.2923 |
| squad | in-domain | 11873 | 0.9116 | 0.9139 | +0.0023 [+0.0000, +0.0045] | WITHIN | 0.9054 | +0.0023 | +0.0022 |
| musique | in-domain | 2417 | 0.5550 | 0.5698 | +0.0148 [+0.0068, +0.0230] | GAIN | 0.4734 | +0.0165 | +0.0248 |
| hotpotqa | zero-shot | 7405 | 0.8585 | 0.8623 | +0.0037 [-0.0010, +0.0080] | WITHIN | 0.6846 | +0.0055 | -0.0412 |
| 2wiki | in-domain | 12576 | 0.8672 | 0.9333 | +0.0661 [+0.0630, +0.0694] | GAIN | 0.6079 | +0.1316 | +0.0576 |
| webqsp | zero-shot | 1503 | 0.1200 | 0.3637 | +0.2437 [+0.2196, +0.2685] | GAIN | 0.0535 | +0.1910 | +0.0852 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
