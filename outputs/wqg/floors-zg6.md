WQG (docs/WQG_ZG6_NULL_UNDER_WQ.md): zg6 at seeds 1 and 2 (WQG's fits) against the seed-0 fit of each split (scr-wqk-zg6-s0, scr-wqk-zg6-s0-hp) under WQ on the T4. The floors below are zg6's, for the calls on zg6-based arms.

# WQ's floors on zg6

zg6 fitted on each WQ split with seeds 0, 1 and 2; seeds 1 and 2 each compared with seed 0 on the six s1eval carves (p@swa, R@5). floor = max(0.0075, 2 sqrt((D1^2 + D2^2) / 2)). Bold: above 0.0075.

| split | dataset | read | seed 0 R@5 | D1 | D2 | spread | floor |
|---|---|---|---:|---:|---:|---:|---:|
| WQ-musique | metaqa | zero-shot | 0.4393 | -0.0154 | -0.0428 | 0.0322 | **0.0643** |
| WQ-musique | squad | in-domain | 0.9097 | +0.0016 | +0.0027 | 0.0022 | 0.0075 |
| WQ-musique | musique | zero-shot | 0.5018 | +0.0150 | +0.0093 | 0.0125 | **0.0250** |
| WQ-musique | hotpotqa | in-domain | 0.9109 | +0.0015 | +0.0016 | 0.0016 | 0.0075 |
| WQ-musique | 2wiki | in-domain | 0.9390 | +0.0012 | +0.0024 | 0.0019 | 0.0075 |
| WQ-musique | webqsp | in-domain | 0.6423 | -0.0012 | -0.0093 | 0.0066 | **0.0133** |
| WQ-hotpotqa | metaqa | zero-shot | 0.4167 | -0.0037 | +0.0025 | 0.0032 | 0.0075 |
| WQ-hotpotqa | squad | in-domain | 0.9114 | -0.0004 | +0.0003 | 0.0004 | 0.0075 |
| WQ-hotpotqa | musique | in-domain | 0.5961 | -0.0065 | -0.0017 | 0.0048 | **0.0095** |
| WQ-hotpotqa | hotpotqa | zero-shot | 0.8183 | +0.0078 | +0.0012 | 0.0056 | **0.0112** |
| WQ-hotpotqa | 2wiki | in-domain | 0.9414 | -0.0017 | +0.0007 | 0.0013 | 0.0075 |
| WQ-hotpotqa | webqsp | in-domain | 0.6421 | -0.0081 | -0.0110 | 0.0097 | **0.0193** |

Floors above 0.0075: 6 of 12 (WQ-musique metaqa, WQ-musique musique, WQ-musique webqsp, WQ-hotpotqa musique, WQ-hotpotqa hotpotqa, WQ-hotpotqa webqsp).
