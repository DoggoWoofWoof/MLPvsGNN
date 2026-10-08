# Screen scr-relz (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-rel (decides the verdict): **MIXED**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7253 | 0.7058 | -0.0196 [-0.0229, -0.0164] | LOSS | 0.0047 | -0.0152 | -0.0323 |
| squad | in-domain | 11873 | 0.9091 | 0.9142 | +0.0051 [+0.0028, +0.0073] | WITHIN | 0.9054 | +0.0051 | +0.0012 |
| musique | zero-shot | 2417 | 0.3564 | 0.5127 | +0.1563 [+0.1450, +0.1672] | GAIN | 0.4734 | +0.1150 | +0.2966 |
| hotpotqa | in-domain | 7405 | 0.9020 | 0.9038 | +0.0018 [-0.0009, +0.0045] | WITHIN | 0.6846 | +0.0030 | -0.0103 |
| 2wiki | in-domain | 12576 | 0.8720 | 0.8725 | +0.0005 [-0.0015, +0.0025] | WITHIN | 0.6079 | +0.0002 | +0.0019 |
| webqsp | zero-shot | 1503 | 0.2414 | 0.2541 | +0.0127 [-0.0036, +0.0292] | WITHIN | 0.0535 | +0.0060 | +0.0259 |

## Against outputs\step1\fits\L-musique (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6537 | 0.7058 | +0.0520 [+0.0472, +0.0563] | GAIN | 0.0047 | +0.0457 | +0.1113 |
| squad | in-domain | 11873 | 0.9100 | 0.9142 | +0.0042 [+0.0019, +0.0067] | WITHIN | 0.9054 | +0.0042 | -0.0008 |
| musique | zero-shot | 2417 | 0.2696 | 0.5127 | +0.2431 [+0.2310, +0.2544] | GAIN | 0.4734 | +0.1523 | +0.3438 |
| hotpotqa | in-domain | 7405 | 0.9015 | 0.9038 | +0.0024 [-0.0003, +0.0049] | WITHIN | 0.6846 | +0.0041 | -0.0070 |
| 2wiki | in-domain | 12576 | 0.8732 | 0.8725 | -0.0007 [-0.0028, +0.0014] | WITHIN | 0.6079 | -0.0045 | -0.0008 |
| webqsp | zero-shot | 1503 | 0.1779 | 0.2541 | +0.0763 [+0.0584, +0.0961] | GAIN | 0.0535 | +0.0539 | +0.0805 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
