# Screen L-squad (p@swa, s1eval carves)

## Against outputs\full_zspg\fits\L-squad (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.7784 | 0.8039 | +0.0255 [+0.0226, +0.0284] | GAIN | 0.0047 | +0.0211 | +0.0374 |
| squad | zero-shot | 11873 | 0.8914 | 0.8930 | +0.0015 [-0.0013, +0.0045] | WITHIN | 0.9054 | +0.0015 | +0.0084 |
| musique | in-domain | 2417 | 0.5765 | 0.5840 | +0.0075 [+0.0000, +0.0153] | WITHIN | 0.4734 | -0.0062 | +0.0248 |
| hotpotqa | in-domain | 7405 | 0.9054 | 0.9082 | +0.0028 [-0.0003, +0.0060] | WITHIN | 0.6846 | +0.0023 | +0.0001 |
| 2wiki | in-domain | 12576 | 0.8733 | 0.9266 | +0.0533 [+0.0501, +0.0566] | GAIN | 0.6079 | +0.0922 | +0.0736 |
| webqsp | zero-shot | 1503 | 0.2771 | 0.3034 | +0.0263 [+0.0076, +0.0442] | GAIN | 0.0535 | +0.0186 | +0.0299 |

## Against outputs\step1\fits\L-squad (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6535 | 0.8039 | +0.1504 [+0.1449, +0.1561] | GAIN | 0.0047 | +0.1480 | +0.2868 |
| squad | zero-shot | 11873 | 0.8797 | 0.8930 | +0.0132 [+0.0099, +0.0166] | GAIN | 0.9054 | +0.0132 | +0.0222 |
| musique | in-domain | 2417 | 0.5743 | 0.5840 | +0.0097 [+0.0026, +0.0174] | GAIN | 0.4734 | -0.0017 | +0.0232 |
| hotpotqa | in-domain | 7405 | 0.9045 | 0.9082 | +0.0037 [+0.0003, +0.0071] | WITHIN | 0.6846 | +0.0053 | +0.0173 |
| 2wiki | in-domain | 12576 | 0.8724 | 0.9266 | +0.0542 [+0.0511, +0.0575] | GAIN | 0.6079 | +0.1004 | +0.0789 |
| webqsp | zero-shot | 1503 | 0.1302 | 0.3034 | +0.1731 [+0.1477, +0.1977] | GAIN | 0.0535 | +0.1324 | +0.1124 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
