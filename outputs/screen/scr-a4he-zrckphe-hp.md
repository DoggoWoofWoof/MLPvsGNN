# Screen scr-a4he-zrckphe-hp (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-p1c3p-zrckp-hp (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8044 | 0.8032 | -0.0013 [-0.0020, -0.0005] | WITHIN | 0.0047 | -0.0021 | -0.0029 |
| squad | in-domain | 11873 | 0.9127 | 0.9122 | -0.0005 [-0.0024, +0.0014] | WITHIN | 0.9054 | -0.0005 | +0.0000 |
| musique | in-domain | 2417 | 0.5745 | 0.5760 | +0.0016 [-0.0032, +0.0062] | WITHIN | 0.4734 | +0.0046 | +0.0037 |
| hotpotqa | zero-shot | 7405 | 0.8513 | 0.8571 | +0.0058 [+0.0029, +0.0087] | WITHIN | 0.6846 | +0.0130 | -0.0112 |
| 2wiki | in-domain | 12576 | 0.9350 | 0.9326 | -0.0024 [-0.0039, -0.0007] | WITHIN | 0.6079 | -0.0045 | -0.0046 |
| webqsp | zero-shot | 1503 | 0.3475 | 0.3464 | -0.0010 [-0.0114, +0.0102] | WITHIN | 0.0535 | -0.0060 | -0.0020 |

## Against outputs\step1\fits\L-hotpotqa (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6540 | 0.8032 | +0.1492 [+0.1437, +0.1546] | GAIN | 0.0047 | +0.1518 | +0.2855 |
| squad | in-domain | 11873 | 0.9116 | 0.9122 | +0.0006 [-0.0017, +0.0029] | WITHIN | 0.9054 | +0.0006 | +0.0024 |
| musique | in-domain | 2417 | 0.5550 | 0.5760 | +0.0210 [+0.0131, +0.0288] | GAIN | 0.4734 | +0.0228 | +0.0368 |
| hotpotqa | zero-shot | 7405 | 0.8585 | 0.8571 | -0.0014 [-0.0061, +0.0034] | WITHIN | 0.6846 | -0.0022 | -0.0456 |
| 2wiki | in-domain | 12576 | 0.8672 | 0.9326 | +0.0654 [+0.0620, +0.0690] | GAIN | 0.6079 | +0.1365 | +0.0600 |
| webqsp | zero-shot | 1503 | 0.1200 | 0.3464 | +0.2264 [+0.2021, +0.2510] | GAIN | 0.0535 | +0.1790 | +0.0665 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
