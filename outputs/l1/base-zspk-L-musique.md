# Screen scr-p1c3-zspk (p@swa, s1eval carves)

## Against outputs\step1\fits\L-musique (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6537 | 0.8009 | +0.1471 [+0.1415, +0.1525] | GAIN | 0.0047 | +0.1453 | +0.2749 |
| squad | in-domain | 11873 | 0.9100 | 0.9114 | +0.0014 [-0.0008, +0.0038] | WITHIN | 0.9054 | +0.0014 | -0.0015 |
| musique | zero-shot | 2417 | 0.2696 | 0.5012 | +0.2316 [+0.2198, +0.2432] | GAIN | 0.4734 | +0.1163 | +0.3868 |
| hotpotqa | in-domain | 7405 | 0.9015 | 0.9080 | +0.0065 [+0.0033, +0.0099] | WITHIN | 0.6846 | +0.0099 | +0.0050 |
| 2wiki | in-domain | 12576 | 0.8732 | 0.9383 | +0.0650 [+0.0619, +0.0683] | GAIN | 0.6079 | +0.1354 | +0.0723 |
| webqsp | zero-shot | 1503 | 0.1779 | 0.3229 | +0.1450 [+0.1216, +0.1699] | GAIN | 0.0535 | +0.1104 | +0.1277 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
