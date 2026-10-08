# The seed null's floors over every full-run split

Step 1's base arm trained with seeds 1 and 2 on each of the six splits, each compared with step 1's seed-0 p@swa of its split on the six s1eval carves (R@5). floor = max(0.0075, 2 x sqrt((D1^2 + D2^2) / 2)), twice the seed-only spread (docs/SCREENS.md, section 2). Bold: a floor above 0.0075.

| split | dataset | read | seed-0 R@5 | D1 (seed 1) | D2 (seed 2) | spread | floor |
|---|---|---|---:|---:|---:|---:|---:|
| J5 | metaqa | in-domain | 0.6443 | +0.0097 | +0.0049 | 0.0077 | **0.0154** |
| J5 | squad | in-domain | 0.9095 | -0.0024 | -0.0021 | 0.0023 | 0.0075 |
| J5 | musique | in-domain | 0.5609 | +0.0052 | +0.0099 | 0.0079 | **0.0158** |
| J5 | hotpotqa | in-domain | 0.9013 | +0.0002 | -0.0010 | 0.0007 | 0.0075 |
| J5 | 2wiki | in-domain | 0.8694 | +0.0028 | +0.0003 | 0.0020 | 0.0075 |
| J5 | webqsp | zero-shot | 0.1167 | +0.0205 | -0.0040 | 0.0148 | **0.0295** |
| L-metaqa | metaqa | zero-shot | 0.0773 | +0.0071 | -0.0095 | 0.0084 | **0.0168** |
| L-metaqa | squad | in-domain | 0.9076 | +0.0029 | +0.0017 | 0.0024 | 0.0075 |
| L-metaqa | musique | in-domain | 0.5614 | -0.0027 | -0.0010 | 0.0020 | 0.0075 |
| L-metaqa | hotpotqa | in-domain | 0.9046 | -0.0007 | -0.0011 | 0.0009 | 0.0075 |
| L-metaqa | 2wiki | in-domain | 0.8639 | +0.0110 | +0.0056 | 0.0087 | **0.0175** |
| L-metaqa | webqsp | zero-shot | 0.0926 | +0.0020 | -0.0033 | 0.0027 | 0.0075 |
| L-squad | metaqa | in-domain | 0.6535 | -0.0037 | -0.0010 | 0.0027 | 0.0075 |
| L-squad | squad | zero-shot | 0.8797 | +0.0024 | +0.0021 | 0.0023 | 0.0075 |
| L-squad | musique | in-domain | 0.5743 | -0.0088 | -0.0144 | 0.0119 | **0.0239** |
| L-squad | hotpotqa | in-domain | 0.9045 | -0.0010 | -0.0032 | 0.0024 | 0.0075 |
| L-squad | 2wiki | in-domain | 0.8724 | +0.0002 | -0.0083 | 0.0059 | **0.0117** |
| L-squad | webqsp | zero-shot | 0.1302 | +0.0067 | +0.0171 | 0.0130 | **0.0260** |
| L-musique | metaqa | in-domain | 0.6537 | +0.0008 | -0.0023 | 0.0017 | 0.0075 |
| L-musique | squad | in-domain | 0.9100 | +0.0001 | +0.0013 | 0.0009 | 0.0075 |
| L-musique | musique | zero-shot | 0.2696 | +0.0503 | -0.0079 | 0.0360 | **0.0720** |
| L-musique | hotpotqa | in-domain | 0.9015 | +0.0014 | +0.0016 | 0.0015 | 0.0075 |
| L-musique | 2wiki | in-domain | 0.8732 | -0.0002 | -0.0007 | 0.0005 | 0.0075 |
| L-musique | webqsp | zero-shot | 0.1779 | -0.0070 | -0.0018 | 0.0051 | **0.0102** |
| L-hotpotqa | metaqa | in-domain | 0.6540 | -0.0092 | -0.0051 | 0.0074 | **0.0149** |
| L-hotpotqa | squad | in-domain | 0.9116 | +0.0002 | +0.0007 | 0.0005 | 0.0075 |
| L-hotpotqa | musique | in-domain | 0.5550 | +0.0015 | +0.0023 | 0.0019 | 0.0075 |
| L-hotpotqa | hotpotqa | zero-shot | 0.8585 | -0.0055 | -0.0130 | 0.0100 | **0.0200** |
| L-hotpotqa | 2wiki | in-domain | 0.8672 | -0.0020 | +0.0011 | 0.0016 | 0.0075 |
| L-hotpotqa | webqsp | zero-shot | 0.1200 | +0.0102 | +0.0129 | 0.0116 | **0.0233** |
| L-2wiki | metaqa | in-domain | 0.6468 | +0.0033 | +0.0037 | 0.0035 | 0.0075 |
| L-2wiki | squad | in-domain | 0.9112 | -0.0053 | -0.0046 | 0.0050 | **0.0099** |
| L-2wiki | musique | in-domain | 0.5601 | +0.0002 | +0.0012 | 0.0009 | 0.0075 |
| L-2wiki | hotpotqa | in-domain | 0.8977 | +0.0063 | -0.0001 | 0.0045 | **0.0089** |
| L-2wiki | 2wiki | zero-shot | 0.8074 | -0.0030 | -0.0073 | 0.0056 | **0.0112** |
| L-2wiki | webqsp | zero-shot | 0.1075 | +0.0243 | +0.0230 | 0.0237 | **0.0473** |

Floors above 0.0075: 17 of 36 (J5 metaqa, J5 musique, J5 webqsp, L-metaqa metaqa, L-metaqa 2wiki, L-squad musique, L-squad 2wiki, L-squad webqsp, L-musique musique, L-musique webqsp, L-hotpotqa metaqa, L-hotpotqa hotpotqa, L-hotpotqa webqsp, L-2wiki squad, L-2wiki hotpotqa, L-2wiki 2wiki, L-2wiki webqsp).
