# zn: the lean MLP without message passing, against zrc and zsp

R@5 on the six s1eval carves (p@swa, seed 0). Differences are zn minus the model, with the 95% question bootstrap. share = (zn - rrf) / (model - rrf), not computed where the model's gain over rrf is under 0.01. (docs/ZN_LEAN_NO_MP.md)

| split | dataset | read | rrf | zn | zrc | zsp | zn - zrc [95% CI] | zn - zsp [95% CI] | share of zsp's gain | share of zrc's gain |
|---|---|---|---:|---:|---:|---:|---|---|---:|---:|
| L-musique | metaqa | in-domain | 0.0047 | 0.2212 | 0.7782 | 0.7794 | -0.5570 [-0.5659, -0.5484] | -0.5582 [-0.5670, -0.5495] | 0.279 | 0.280 |
| L-musique | squad | in-domain | 0.9054 | 0.9121 | 0.9105 | 0.9125 | +0.0016 [-0.0008, +0.0040] | -0.0004 [-0.0029, +0.0020] | n/a | n/a |
| L-musique | musique | zero-shot | 0.4734 | 0.4783 | 0.5235 | 0.5111 | -0.0453 [-0.0548, -0.0358] | -0.0329 [-0.0433, -0.0226] | 0.130 | 0.098 |
| L-musique | hotpotqa | in-domain | 0.6846 | 0.8018 | 0.9041 | 0.9065 | -0.1022 [-0.1075, -0.0968] | -0.1047 [-0.1101, -0.0993] | 0.528 | 0.534 |
| L-musique | 2wiki | in-domain | 0.6079 | 0.7996 | 0.8733 | 0.8724 | -0.0737 [-0.0772, -0.0698] | -0.0728 [-0.0766, -0.0690] | 0.725 | 0.722 |
| L-musique | webqsp | zero-shot | 0.0535 | 0.0943 | 0.3384 | 0.3245 | -0.2441 [-0.2696, -0.2203] | -0.2302 [-0.2552, -0.2061] | 0.151 | 0.143 |
| L-hotpotqa | metaqa | in-domain | 0.0047 | 0.2192 | 0.7793 | 0.7807 | -0.5600 [-0.5687, -0.5515] | -0.5615 [-0.5700, -0.5529] | 0.276 | 0.277 |
| L-hotpotqa | squad | in-domain | 0.9054 | 0.9093 | 0.9112 | 0.9109 | -0.0019 [-0.0045, +0.0007] | -0.0016 [-0.0041, +0.0009] | n/a | n/a |
| L-hotpotqa | musique | in-domain | 0.4734 | 0.5184 | 0.5645 | 0.5647 | -0.0461 [-0.0546, -0.0380] | -0.0463 [-0.0546, -0.0386] | 0.493 | 0.494 |
| L-hotpotqa | hotpotqa | zero-shot | 0.6846 | 0.7398 | 0.8451 | 0.8435 | -0.1053 [-0.1111, -0.0997] | -0.1036 [-0.1095, -0.0982] | 0.347 | 0.344 |
| L-hotpotqa | 2wiki | in-domain | 0.6079 | 0.7970 | 0.8731 | 0.8744 | -0.0762 [-0.0799, -0.0725] | -0.0775 [-0.0813, -0.0738] | 0.710 | 0.713 |
| L-hotpotqa | webqsp | zero-shot | 0.0535 | 0.0973 | 0.2865 | 0.2262 | -0.1892 [-0.2117, -0.1668] | -0.1289 [-0.1503, -0.1078] | 0.254 | 0.188 |
