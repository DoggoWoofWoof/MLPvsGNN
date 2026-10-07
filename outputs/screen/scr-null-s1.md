# Screen scr-null-s1 (p@swa, s1eval carves)

## Against outputs\step1\fits\L-musique (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6537 | 0.6545 | +0.0008 [-0.0020, +0.0036] | WITHIN | 0.0047 | -0.0006 | +0.0040 |
| squad | in-domain | 11873 | 0.9100 | 0.9100 | +0.0001 [-0.0019, +0.0022] | WITHIN | 0.9054 | +0.0001 | -0.0017 |
| musique | zero-shot | 2417 | 0.2696 | 0.3199 | +0.0503 [+0.0422, +0.0587] | GAIN | 0.4734 | +0.0244 | +0.0203 |
| hotpotqa | in-domain | 7405 | 0.9015 | 0.9028 | +0.0014 [-0.0007, +0.0034] | WITHIN | 0.6846 | +0.0018 | +0.0135 |
| 2wiki | in-domain | 12576 | 0.8732 | 0.8731 | -0.0002 [-0.0019, +0.0015] | WITHIN | 0.6079 | -0.0015 | +0.0029 |
| webqsp | zero-shot | 1503 | 0.1779 | 0.1708 | -0.0070 [-0.0195, +0.0049] | WITHIN | 0.0535 | +0.0007 | -0.0020 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
