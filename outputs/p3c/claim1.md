# P3c: Claim 1 for the lean pair (zrckp of zspk), with intervals

**Call: CLAIM1_NOT_ALL.** Development reads on s1eval (candidate p@swa, 2,000 question resamples, 95% intervals). The rule is docs/P3C_CLAIM1_INTERVALS.md.

## The in-domain clause (J5, R@5): the MLP keeps at least 0.90 of the GNN's gain over RRF

| dataset | RRF | GNN | MLP | GNN's gain [95%] | MLP's share [95%] | call |
| --- | ---: | ---: | ---: | --- | --- | --- |
| metaqa | 0.0047 | 0.8012 | 0.7989 | +0.7964 [+0.7896, +0.8027] | 0.997 [0.996, 0.998] | HOLDS |
| squad | 0.9054 | 0.9099 | 0.9112 | +0.0045 [+0.0012, +0.0077] | 1.302 [0.859, 2.824] | NOT_SHOWN |
| musique | 0.4734 | 0.5800 | 0.5728 | +0.1065 [+0.0965, +0.1160] | 0.933 [0.881, 0.983] | NOT_SHOWN |
| hotpotqa | 0.6846 | 0.9044 | 0.9065 | +0.2198 [+0.2137, +0.2259] | 1.010 [0.999, 1.020] | HOLDS |
| 2wiki | 0.6079 | 0.9351 | 0.9304 | +0.3272 [+0.3226, +0.3319] | 0.986 [0.981, 0.991] | HOLDS |

## The zero-shot clause (R@5): not below the GNN by more than the in-domain gap

| dataset | read | zero-shot gap (GNN - MLP) | allowed (in-domain gap, at least 0) | excess [95%] | call |
| --- | --- | ---: | ---: | --- | --- |
| metaqa | L-metaqa | -0.0366 | 0.0023 | -0.0389 [-0.0447, -0.0332] | HOLDS |
| squad | L-squad | +0.0019 | 0.0000 | +0.0019 [-0.0006, +0.0043] | HOLDS |
| musique | L-musique | -0.0110 | 0.0072 | -0.0181 [-0.0266, -0.0097] | HOLDS |
| hotpotqa | L-hotpotqa | -0.0014 | 0.0000 | -0.0014 [-0.0045, +0.0016] | HOLDS |
| 2wiki | L-2wiki | +0.0059 | 0.0047 | +0.0012 [-0.0015, +0.0039] | HOLDS |
| webqsp | J5 | -0.0698 | 0.0000 | -0.0698 [-0.0870, -0.0523] | HOLDS |

## Beside, deciding nothing

**The shares on every read** (R@5 / FC@5 / hit@1; n/a where the GNN's point gain is at most 0.002):

| split | dataset | read | R@5 share | FC@5 share | hit@1 share |
| --- | --- | --- | --- | --- | --- |
| J5 | metaqa | in-domain | 0.997 [0.996, 0.998] | 0.998 [0.996, 1.001] | 0.986 [0.983, 0.990] |
| J5 | squad | in-domain | 1.302 [0.859, 2.824] | 1.302 [0.859, 2.824] | 1.577 [1.262, 2.126] |
| J5 | musique | in-domain | 0.933 [0.881, 0.983] | 0.990 [0.918, 1.065] | 0.936 [0.822, 1.054] |
| J5 | hotpotqa | in-domain | 1.010 [0.999, 1.020] | 1.011 [1.001, 1.021] | 0.844 [0.480, 1.227] |
| J5 | 2wiki | in-domain | 0.986 [0.981, 0.991] | 0.984 [0.977, 0.990] | 0.915 [0.893, 0.936] |
| J5 | webqsp | zero-shot | 1.294 [1.211, 1.385] | 1.351 [1.243, 1.472] | 0.744 [0.600, 0.894] |
| L-metaqa | metaqa | zero-shot | 1.176 [1.146, 1.205] | 1.193 [1.155, 1.232] | 0.769 [0.549, 1.042] |
| L-metaqa | squad | in-domain | 1.650 [-inf, 8.200] | 1.650 [-inf, 8.200] | 1.603 [1.276, 2.107] |
| L-metaqa | musique | in-domain | 1.052 [0.991, 1.113] | 1.010 [0.932, 1.093] | 1.062 [0.936, 1.209] |
| L-metaqa | hotpotqa | in-domain | 0.999 [0.987, 1.010] | 0.998 [0.987, 1.008] | 0.639 [0.482, 0.773] |
| L-metaqa | 2wiki | in-domain | 0.999 [0.994, 1.004] | 1.002 [0.995, 1.009] | 0.992 [0.959, 1.025] |
| L-metaqa | webqsp | zero-shot | 0.963 [0.838, 1.102] | 0.976 [0.831, 1.143] | 0.718 [0.383, 1.046] |
| L-squad | metaqa | in-domain | 0.997 [0.996, 0.999] | 0.996 [0.994, 0.999] | 0.989 [0.986, 0.993] |
| L-squad | squad | zero-shot | n/a | n/a | n/a |
| L-squad | musique | in-domain | 0.943 [0.891, 0.996] | 0.981 [0.910, 1.056] | 0.875 [0.775, 0.976] |
| L-squad | hotpotqa | in-domain | 1.002 [0.992, 1.012] | 1.005 [0.996, 1.015] | 0.572 [0.392, 0.713] |
| L-squad | 2wiki | in-domain | 1.020 [1.015, 1.025] | 1.043 [1.035, 1.051] | 0.888 [0.862, 0.914] |
| L-squad | webqsp | zero-shot | 1.213 [1.137, 1.297] | 1.261 [1.160, 1.369] | 0.769 [0.630, 0.935] |
| L-musique | metaqa | in-domain | 1.001 [1.000, 1.003] | 1.000 [0.998, 1.003] | 1.007 [1.003, 1.011] |
| L-musique | squad | in-domain | 1.042 [0.690, 1.542] | 1.042 [0.690, 1.542] | 1.026 [0.845, 1.252] |
| L-musique | musique | zero-shot | 1.395 [1.130, 1.884] | 1.098 [0.892, 1.400] | 1.719 [1.264, 2.854] |
| L-musique | hotpotqa | in-domain | 0.993 [0.985, 1.003] | 0.987 [0.979, 0.996] | 0.462 [0.168, 0.680] |
| L-musique | 2wiki | in-domain | 0.989 [0.985, 0.994] | 0.985 [0.979, 0.990] | 1.004 [0.981, 1.028] |
| L-musique | webqsp | zero-shot | 1.148 [1.083, 1.219] | 1.226 [1.140, 1.321] | 0.634 [0.513, 0.750] |
| L-hotpotqa | metaqa | in-domain | 1.000 [0.999, 1.002] | 1.000 [0.998, 1.002] | 0.999 [0.995, 1.001] |
| L-hotpotqa | squad | in-domain | 0.906 [0.654, 1.149] | 0.906 [0.654, 1.149] | 1.005 [0.854, 1.192] |
| L-hotpotqa | musique | in-domain | 0.991 [0.938, 1.043] | 0.953 [0.883, 1.024] | 0.848 [0.754, 0.941] |
| L-hotpotqa | hotpotqa | zero-shot | 1.009 [0.990, 1.027] | 1.006 [0.988, 1.025] | n/a |
| L-hotpotqa | 2wiki | in-domain | 1.005 [1.000, 1.010] | 1.012 [1.005, 1.019] | 0.914 [0.892, 0.935] |
| L-hotpotqa | webqsp | zero-shot | 1.371 [1.274, 1.482] | 1.441 [1.316, 1.584] | 0.783 [0.616, 0.980] |
| L-2wiki | metaqa | in-domain | 0.998 [0.996, 0.999] | 0.996 [0.994, 0.999] | 0.993 [0.990, 0.997] |
| L-2wiki | squad | in-domain | 1.127 [0.792, 1.812] | 1.127 [0.792, 1.812] | 1.454 [1.206, 1.837] |
| L-2wiki | musique | in-domain | 0.973 [0.922, 1.023] | 1.007 [0.940, 1.081] | 0.876 [0.786, 0.971] |
| L-2wiki | hotpotqa | in-domain | 1.005 [0.994, 1.015] | 1.003 [0.992, 1.012] | 0.983 [0.840, 1.140] |
| L-2wiki | 2wiki | zero-shot | 0.977 [0.968, 0.985] | 0.976 [0.966, 0.987] | 0.971 [0.943, 1.001] |
| L-2wiki | webqsp | zero-shot | 1.287 [1.207, 1.377] | 1.332 [1.232, 1.444] | 0.778 [0.636, 0.937] |

**webqsp on the leave-out fits** (zero-shot on every split; allowed gap 0):

| split | zero-shot gap | excess [95%] | would-be call |
| --- | ---: | --- | --- |
| L-metaqa | +0.0023 | +0.0023 [-0.0060, +0.0105] | HOLDS |
| L-squad | -0.0533 | -0.0533 [-0.0707, -0.0355] | HOLDS |
| L-musique | -0.0399 | -0.0399 [-0.0566, -0.0231] | HOLDS |
| L-hotpotqa | -0.0795 | -0.0795 [-0.0973, -0.0611] | HOLDS |
| L-2wiki | -0.0703 | -0.0703 [-0.0873, -0.0529] | HOLDS |
