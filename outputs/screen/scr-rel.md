# Screen scr-rel (p@swa, s1eval carves)

## Against outputs\step1\fits\L-musique (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6537 | 0.7253 | +0.0716 [+0.0672, +0.0761] | GAIN | 0.0047 | +0.0609 | +0.1436 |
| squad | in-domain | 11873 | 0.9100 | 0.9091 | -0.0008 [-0.0029, +0.0013] | WITHIN | 0.9054 | -0.0008 | -0.0019 |
| musique | zero-shot | 2417 | 0.2696 | 0.3564 | +0.0868 [+0.0773, +0.0965] | GAIN | 0.4734 | +0.0372 | +0.0472 |
| hotpotqa | in-domain | 7405 | 0.9015 | 0.9020 | +0.0005 [-0.0019, +0.0028] | WITHIN | 0.6846 | +0.0011 | +0.0032 |
| 2wiki | in-domain | 12576 | 0.8732 | 0.8720 | -0.0012 [-0.0029, +0.0006] | WITHIN | 0.6079 | -0.0047 | -0.0027 |
| webqsp | zero-shot | 1503 | 0.1779 | 0.2414 | +0.0635 [+0.0477, +0.0794] | GAIN | 0.0535 | +0.0479 | +0.0546 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
