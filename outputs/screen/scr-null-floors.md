# The seed null's floors

Step 1's base arm trained with seeds 1 and 2, each compared with step 1's seed-0 p@swa of its split on the six s1eval carves (R@5). floor = max(0.0075, 2 x sqrt((D1^2 + D2^2) / 2)), twice the seed-only spread (docs/SCREENS.md, section 2). Bold: a floor above 0.0075.

| split | dataset | read | seed-0 R@5 | D1 (seed 1) | D2 (seed 2) | spread | floor |
|---|---|---|---:|---:|---:|---:|---:|
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

Floors above 0.0075: 5 of 12 (L-musique musique, L-musique webqsp, L-hotpotqa metaqa, L-hotpotqa hotpotqa, L-hotpotqa webqsp).
