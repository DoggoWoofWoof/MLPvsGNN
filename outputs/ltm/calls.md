# LTM: zrckpn (zrckp + LTW's next-hop columns (NEXT: each row against the first stage's top five rows, cross-fitted)) against DOB's CPU zrckp, on PKO's pools, on the host's CPU

## Screen: ADOPT_FOR_FULL_RUN

counts: {"GAIN": 2, "LOSS": 0, "WITHIN": 8}
DEP's sixteen-read rule beside: screen ADOPT_FOR_FULL_RUN, full run None

| split | read | base s0, s1, s2 | zrckpn s0 | delta | floor | call |
|---|---|---|---:|---:|---:|---|
| WQ-musique | squad | 0.9127, 0.9122, 0.9138 | 0.9138 | +0.0011 | 0.0075 | WITHIN |
| WQ-musique | hotpotqa (C1R) | 0.8411, 0.8424, 0.8388 | 0.8460 | +0.0049 | 0.0075 | WITHIN |
| WQ-musique | 2wiki | 0.9070, 0.9079, 0.9049 | 0.9155 | +0.0085 | 0.0075 | GAIN |
| WQ-musique | webqsp | 0.6522, 0.6583, 0.6566 | 0.6457 | -0.0065 | 0.0106 | WITHIN |
| WQ-hotpotqa | squad | 0.9138, 0.9129, 0.9133 | 0.9132 | -0.0006 | 0.0075 | WITHIN |
| WQ-hotpotqa | musique (C1R) | 0.5749, 0.5718, 0.5752 | 0.5806 | +0.0057 | 0.0075 | WITHIN |
| WQ-hotpotqa | 2wiki | 0.8987, 0.9082, 0.9063 | 0.9135 | +0.0148 | 0.0172 | WITHIN |
| WQ-hotpotqa | webqsp | 0.6446, 0.6524, 0.6514 | 0.6469 | +0.0023 | 0.0146 | WITHIN |
| WQ-hotpotqa | musique-3hop | 0.5241, 0.5237, 0.5232 | 0.5351 | +0.0110 | 0.0075 | GAIN |
| WQ-hotpotqa | musique-4hop | 0.3475, 0.3556, 0.3531 | 0.3574 | +0.0099 | 0.0138 | WITHIN |
| WQ-musique | musique | 0.5152, 0.5086, 0.5134 | 0.5190 | +0.0038 | 0.0097 | WITHIN (beside) |
| WQ-musique | metaqa | 0.3406, 0.3670, 0.3860 | 0.4126 | +0.0720 | 0.0743 | WITHIN (beside) |
| WQ-hotpotqa | hotpotqa | 0.8103, 0.8099, 0.8101 | 0.8193 | +0.0090 | 0.0075 | GAIN (beside) |
| WQ-hotpotqa | metaqa | 0.3847, 0.4037, 0.3797 | 0.3668 | -0.0179 | 0.0278 | WITHIN (beside) |
| WQ-musique | musique-3hop | 0.4434, 0.4434, 0.4390 | 0.4544 | +0.0110 | 0.0075 | GAIN (beside) |
| WQ-musique | musique-4hop | 0.3142, 0.2969, 0.3154 | 0.3086 | -0.0056 | 0.0245 | WITHIN (beside) |

Beside, deciding nothing: the columns per carve (ltmcols' records):

| split | carve | training | rows next to an anchor | A ranked | A T2 bil | pair | rrf |
|---|---|---|---:|---:|---:|---:|---:|
| WQ-musique | webqsp/fit | True | 0.050 | 88 | 0.023 | 0.182 | 0.216 |
| WQ-musique | squad/fit | True | 0.473 | 0 | - | - | - |
| WQ-musique | hotpotqa/fit | True | 0.115 | 2222 | 0.793 | 0.460 | 0.390 |
| WQ-musique | 2wiki/fit | True | 0.089 | 3464 | 0.915 | 0.213 | 0.177 |
| WQ-musique | metaqa/s1eval | False | 0.008 | 19 | 0.105 | 0.368 | 0.737 |
| WQ-musique | squad/s1eval | False | 0.430 | 0 | - | - | - |
| WQ-musique | musique/s1eval | False | 0.047 | 923 | 0.454 | 0.360 | 0.298 |
| WQ-musique | hotpotqa/s1eval | False | 0.114 | 2799 | 0.809 | 0.446 | 0.399 |
| WQ-musique | 2wiki/s1eval | False | 0.093 | 7619 | 0.865 | 0.179 | 0.213 |
| WQ-musique | webqsp/s1eval | False | 0.048 | 99 | 0.061 | 0.182 | 0.192 |
| WQ-hotpotqa | webqsp/fit | True | 0.050 | 88 | 0.023 | 0.182 | 0.216 |
| WQ-hotpotqa | squad/fit | True | 0.473 | 0 | - | - | - |
| WQ-hotpotqa | musique/s1fit | True | 0.045 | 1262 | 0.542 | 0.405 | 0.345 |
| WQ-hotpotqa | 2wiki/fit | True | 0.089 | 3464 | 0.915 | 0.213 | 0.177 |
| WQ-hotpotqa | metaqa/s1eval | False | 0.008 | 19 | 0.053 | 0.368 | 0.737 |
| WQ-hotpotqa | squad/s1eval | False | 0.430 | 0 | - | - | - |
| WQ-hotpotqa | musique/s1eval | False | 0.047 | 923 | 0.493 | 0.360 | 0.298 |
| WQ-hotpotqa | hotpotqa/s1eval | False | 0.114 | 2799 | 0.679 | 0.446 | 0.399 |
| WQ-hotpotqa | 2wiki/s1eval | False | 0.093 | 7619 | 0.855 | 0.179 | 0.213 |
| WQ-hotpotqa | webqsp/s1eval | False | 0.048 | 99 | 0.061 | 0.182 | 0.192 |

| fit | train s | read s |
|---|---:|---:|
| scr-ltm-zrckpn-s0 | 2714.4 | 3517.7 |
| scr-ltm-zrckpn-s0-hp | 5190.2 | 3745.2 |
