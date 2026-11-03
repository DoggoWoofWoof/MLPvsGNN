# LTS: zrckps (zrckp + LTS's next-hop columns (NEXT on the span residual: each anchor's question residual off the span of the first stage's top five rows, cross-fitted)) against LTM's zrckpn, on PKO's pools, on the host's CPU

## Screen: NO_GAIN

counts: {"GAIN": 0, "LOSS": 0, "WITHIN": 10}

| split | read | zrckpn s0, s1, s2 | zrckps s0 | delta | floor | call |
|---|---|---|---:|---:|---:|---|
| WQ-musique | squad | 0.9138, 0.9125, 0.9149 | 0.9133 | -0.0005 | 0.0075 | WITHIN |
| WQ-musique | hotpotqa | 0.8460, 0.8469, 0.8491 | 0.8448 | -0.0012 | 0.0075 | WITHIN |
| WQ-musique | 2wiki | 0.9155, 0.9159, 0.9133 | 0.9137 | -0.0018 | 0.0075 | WITHIN |
| WQ-musique | webqsp | 0.6457, 0.6498, 0.6552 | 0.6407 | -0.0050 | 0.0146 | WITHIN |
| WQ-hotpotqa | squad | 0.9132, 0.9120, 0.9132 | 0.9133 | +0.0001 | 0.0075 | WITHIN |
| WQ-hotpotqa | musique | 0.5806, 0.5803, 0.5777 | 0.5759 | -0.0047 | 0.0075 | WITHIN |
| WQ-hotpotqa | 2wiki | 0.9135, 0.9166, 0.9135 | 0.9102 | -0.0033 | 0.0075 | WITHIN |
| WQ-hotpotqa | webqsp | 0.6469, 0.6597, 0.6512 | 0.6520 | +0.0051 | 0.0191 | WITHIN |
| WQ-hotpotqa | musique-3hop | 0.5351, 0.5333, 0.5263 | 0.5237 | -0.0114 | 0.0127 | WITHIN |
| WQ-hotpotqa | musique-4hop | 0.3574, 0.3562, 0.3562 | 0.3543 | -0.0031 | 0.0075 | WITHIN |
| WQ-musique | musique | 0.5190, 0.5227, 0.5381 | 0.5262 | +0.0072 | 0.0275 | WITHIN (beside) |
| WQ-musique | metaqa | 0.4126, 0.4256, 0.4167 | 0.4230 | +0.0104 | 0.0193 | WITHIN (beside) |
| WQ-hotpotqa | hotpotqa | 0.8193, 0.8182, 0.8136 | 0.8188 | -0.0005 | 0.0082 | WITHIN (beside) |
| WQ-hotpotqa | metaqa | 0.3668, 0.4261, 0.4131 | 0.4372 | +0.0704 | 0.1064 | WITHIN (beside) |
| WQ-musique | musique-3hop | 0.4544, 0.4596, 0.4702 | 0.4596 | +0.0053 | 0.0235 | WITHIN (beside) |
| WQ-musique | musique-4hop | 0.3086, 0.3148, 0.3352 | 0.3148 | +0.0062 | 0.0385 | WITHIN (beside) |

next (section 7): LTM section 8 (c): the term over every pool row against each anchor, linked or not

Beside, deciding nothing: the A lists' T2 under bil, LTM's scorer and LTS's (carve records):

| split | carve | A ranked | LTM | LTS | LTS - LTM | musique 2/3/4-hop, LTM / LTS |
|---|---|---:|---:|---:|---:|---|
| WQ-musique | webqsp/fit | 88 | 0.023 | 0.045 | +0.023 |  |
| WQ-musique | squad/fit | 0 | - | - | - |  |
| WQ-musique | hotpotqa/fit | 2222 | 0.793 | 0.792 | -0.000 |  |
| WQ-musique | 2wiki/fit | 3464 | 0.915 | 0.918 | +0.003 |  |
| WQ-musique | metaqa/s1eval | 19 | 0.105 | 0.158 | +0.053 |  |
| WQ-musique | squad/s1eval | 0 | - | - | - |  |
| WQ-musique | musique/s1eval | 923 | 0.454 | 0.450 | -0.004 | 0.509 / 0.498, 0.371 / 0.388, 0.371 / 0.355 |
| WQ-musique | hotpotqa/s1eval | 2799 | 0.809 | 0.802 | -0.007 |  |
| WQ-musique | 2wiki/s1eval | 7619 | 0.865 | 0.862 | -0.003 |  |
| WQ-musique | webqsp/s1eval | 99 | 0.061 | 0.040 | -0.020 |  |
| WQ-hotpotqa | webqsp/fit | 88 | 0.023 | 0.023 | +0.000 |  |
| WQ-hotpotqa | squad/fit | 0 | - | - | - |  |
| WQ-hotpotqa | musique/s1fit | 1262 | 0.542 | 0.533 | -0.009 | 0.596 / 0.590, 0.457 / 0.440, 0.368 / 0.368 |
| WQ-hotpotqa | 2wiki/fit | 3464 | 0.915 | 0.911 | -0.004 |  |
| WQ-hotpotqa | metaqa/s1eval | 19 | 0.053 | 0.105 | +0.053 |  |
| WQ-hotpotqa | squad/s1eval | 0 | - | - | - |  |
| WQ-hotpotqa | musique/s1eval | 923 | 0.493 | 0.490 | -0.003 | 0.549 / 0.536, 0.420 / 0.449, 0.387 / 0.363 |
| WQ-hotpotqa | hotpotqa/s1eval | 2799 | 0.679 | 0.674 | -0.005 |  |
| WQ-hotpotqa | 2wiki/s1eval | 7619 | 0.855 | 0.858 | +0.003 |  |
| WQ-hotpotqa | webqsp/s1eval | 99 | 0.061 | 0.061 | +0.000 |  |

LTC's zrckpc (screen NO_GAIN) - zrckpn, as filed: WQ-musique squad s0 +0.0004; WQ-musique hotpotqa s0 -0.0029; WQ-musique 2wiki s0 -0.0042; WQ-musique webqsp s0 -0.0024; WQ-hotpotqa squad s0 -0.0003; WQ-hotpotqa musique s0 -0.0055; WQ-hotpotqa 2wiki s0 -0.0032; WQ-hotpotqa webqsp s0 -0.0076; WQ-hotpotqa musique-3hop s0 -0.0105; WQ-hotpotqa musique-4hop s0 -0.0099; WQ-musique musique s0 +0.0133; WQ-musique metaqa s0 -0.0298; WQ-hotpotqa hotpotqa s0 -0.0024; WQ-hotpotqa metaqa s0 +0.0397; WQ-musique musique-3hop s0 +0.0105; WQ-musique musique-4hop s0 +0.0068

| fit | train s | read s |
|---|---:|---:|
| scr-lts-zrckps-s0 | 3295.3 | 3789.1 |
| scr-lts-zrckps-s0-hp | 6170.9 | 4174.4 |
