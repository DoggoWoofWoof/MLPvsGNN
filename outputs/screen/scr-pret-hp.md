# Screen scr-pret-hp (p@swa, s1eval carves)

## Against outputs\step1\fits\L-hotpotqa (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6540 | 0.6476 | -0.0064 [-0.0095, -0.0035] | WITHIN | 0.0047 | -0.0027 | -0.0114 |
| squad | in-domain | 11873 | 0.9116 | 0.9103 | -0.0013 [-0.0039, +0.0012] | WITHIN | 0.9054 | -0.0013 | -0.0098 |
| musique | in-domain | 2417 | 0.5550 | 0.5513 | -0.0038 [-0.0097, +0.0022] | WITHIN | 0.4734 | +0.0033 | +0.0046 |
| hotpotqa | zero-shot | 7405 | 0.8585 | 0.8332 | -0.0253 [-0.0295, -0.0213] | LOSS | 0.6846 | -0.0493 | -0.0020 |
| 2wiki | in-domain | 12576 | 0.8672 | 0.8709 | +0.0037 [+0.0015, +0.0059] | WITHIN | 0.6079 | +0.0143 | -0.0080 |
| webqsp | zero-shot | 1503 | 0.1200 | 0.1158 | -0.0042 [-0.0153, +0.0068] | WITHIN | 0.0535 | -0.0033 | -0.0100 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
