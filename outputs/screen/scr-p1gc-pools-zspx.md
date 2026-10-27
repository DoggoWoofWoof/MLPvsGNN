# Screen scr-p1gc-zspx (p@swa, s1eval carves)

## Against outputs\screen\fits\scr-g5-zspx (decides the verdict): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.8030 | 0.7992 | -0.0038 [-0.0055, -0.0021] | WITHIN | 0.0047 | -0.0026 | -0.0097 |
| squad | in-domain | 11873 | 0.9092 | 0.9131 | +0.0039 [+0.0019, +0.0060] | WITHIN | 0.9054 | +0.0039 | +0.0017 |
| musique | zero-shot | 2417 | 0.5073 | 0.5206 | +0.0133 [+0.0067, +0.0197] | GAIN | 0.4734 | +0.0058 | +0.0174 |
| hotpotqa | in-domain | 7405 | 0.9074 | 0.9105 | +0.0031 [+0.0006, +0.0057] | WITHIN | 0.6846 | +0.0045 | -0.0022 |
| 2wiki | in-domain | 12576 | 0.9351 | 0.9289 | -0.0062 [-0.0078, -0.0046] | WITHIN | 0.6079 | -0.0148 | +0.0026 |
| webqsp | zero-shot | 1503 | 0.3204 | 0.3513 | +0.0309 [+0.0170, +0.0444] | GAIN | 0.0535 | +0.0299 | -0.0080 |

## Against outputs\step1\fits\L-musique (reported only): **PROMISING**

| dataset | read | questions | base R@5 | new R@5 | delta R@5 [95% CI] | call | rrf R@5 | delta FC@5 | delta hit@1 |
|---|---|---:|---:|---:|---|---|---:|---:|---:|
| metaqa | in-domain | 9785 | 0.6537 | 0.7992 | +0.1455 [+0.1400, +0.1507] | GAIN | 0.0047 | +0.1448 | +0.2797 |
| squad | in-domain | 11873 | 0.9100 | 0.9131 | +0.0031 [+0.0010, +0.0052] | WITHIN | 0.9054 | +0.0031 | -0.0002 |
| musique | zero-shot | 2417 | 0.2696 | 0.5206 | +0.2510 [+0.2392, +0.2631] | GAIN | 0.4734 | +0.1332 | +0.3678 |
| hotpotqa | in-domain | 7405 | 0.9015 | 0.9105 | +0.0090 [+0.0059, +0.0123] | GAIN | 0.6846 | +0.0117 | +0.0109 |
| 2wiki | in-domain | 12576 | 0.8732 | 0.9289 | +0.0557 [+0.0526, +0.0589] | GAIN | 0.6079 | +0.1125 | +0.0676 |
| webqsp | zero-shot | 1503 | 0.1779 | 0.3513 | +0.1734 [+0.1499, +0.1978] | GAIN | 0.0535 | +0.1371 | +0.1218 |

Rule (docs/SCREENS.md): a read's call is GAIN when its R@5 difference is at least 0.0075 and its 95% question-bootstrap interval lies above 0, LOSS when the mirror holds, WITHIN otherwise; PROMISING = a GAIN and no LOSS, MIXED = both, NO_GAIN = no GAIN. rrf is the zero-shot floor (plain retrieval).
