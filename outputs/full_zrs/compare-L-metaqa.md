# Screen L-metaqa (p@swa, s1eval carves)

## Against outputs\full_zret\fits\L-metaqa (decides the verdict): **NO_GAIN**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | zero-shot | 9785 | 0.1417 | 0.1417 | +0.0000 [+0.0000, +0.0000] | WITHIN | 0.0047 | +0.0000 | +0.0000 |
| squad | in-domain | 11873 | 0.9106 | 0.9106 | +0.0000 [+0.0000, +0.0000] | WITHIN | 0.9054 | +0.0000 | +0.0000 |
| musique | in-domain | 2417 | 0.5701 | 0.5701 | +0.0000 [+0.0000, +0.0000] | WITHIN | 0.4734 | +0.0000 | +0.0000 |
| hotpotqa | in-domain | 7405 | 0.9014 | 0.9014 | +0.0000 [+0.0000, +0.0000] | WITHIN | 0.6846 | +0.0000 | +0.0000 |
| 2wiki | in-domain | 12576 | 0.8705 | 0.8705 | +0.0000 [+0.0000, +0.0000] | WITHIN | 0.6079 | +0.0000 | +0.0000 |
| webqsp | zero-shot | 1503 | 0.0997 | 0.0997 | +0.0000 [+0.0000, +0.0000] | WITHIN | 0.0535 | +0.0000 | +0.0000 |

## Against outputs\full_rmatch\fits\L-metaqa (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | zero-shot | 9785 | 0.0773 | 0.1417 | +0.0644 [+0.0594, +0.0696] | GAIN | 0.0047 | +0.0552 | +0.0009 |
| squad | in-domain | 11873 | 0.9076 | 0.9106 | +0.0029 [+0.0008, +0.0051] | WITHIN | 0.9054 | +0.0029 | +0.0034 |
| musique | in-domain | 2417 | 0.5614 | 0.5701 | +0.0087 [+0.0023, +0.0151] | GAIN | 0.4734 | +0.0079 | +0.0145 |
| hotpotqa | in-domain | 7405 | 0.9046 | 0.9014 | -0.0032 [-0.0059, -0.0006] | WITHIN | 0.6846 | -0.0065 | +0.0035 |
| 2wiki | in-domain | 12576 | 0.8639 | 0.8705 | +0.0066 [+0.0044, +0.0085] | WITHIN | 0.6079 | +0.0080 | +0.0136 |
| webqsp | zero-shot | 1503 | 0.0926 | 0.0997 | +0.0071 [-0.0007, +0.0149] | WITHIN | 0.0535 | +0.0080 | -0.0027 |

## Against outputs\step1\fits\L-metaqa (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | zero-shot | 9785 | 0.0773 | 0.1417 | +0.0644 [+0.0594, +0.0696] | GAIN | 0.0047 | +0.0552 | +0.0009 |
| squad | in-domain | 11873 | 0.9076 | 0.9106 | +0.0029 [+0.0008, +0.0051] | WITHIN | 0.9054 | +0.0029 | +0.0034 |
| musique | in-domain | 2417 | 0.5614 | 0.5701 | +0.0087 [+0.0023, +0.0151] | GAIN | 0.4734 | +0.0079 | +0.0145 |
| hotpotqa | in-domain | 7405 | 0.9046 | 0.9014 | -0.0032 [-0.0059, -0.0006] | WITHIN | 0.6846 | -0.0065 | +0.0035 |
| 2wiki | in-domain | 12576 | 0.8639 | 0.8705 | +0.0066 [+0.0044, +0.0085] | WITHIN | 0.6079 | +0.0080 | +0.0136 |
| webqsp | zero-shot | 1503 | 0.0926 | 0.0997 | +0.0071 [-0.0007, +0.0149] | WITHIN | 0.0535 | +0.0080 | -0.0027 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
