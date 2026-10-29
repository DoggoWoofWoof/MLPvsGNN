WQG (docs/WQG_ZG6_NULL_UNDER_WQ.md): zg6r at seeds 1 and 2 (WQG's fits) against the seed-0 fit of each split (scr-g6rw-zg6r, scr-g6rw-zg6r-hp) under WQ on the T4. The floors below are zg6r's, for the calls on zg6r-based arms.

# WQ's floors on zg6r

zg6r fitted on each WQ split with seeds 0, 1 and 2; seeds 1 and 2 each compared with seed 0 on the six s1eval carves (p@swa, R@5). floor = max(0.0075, 2 sqrt((D1^2 + D2^2) / 2)). Bold: above 0.0075.

| split | dataset | read | seed 0 R@5 | D1 | D2 | spread | floor |
|---|---|---|---:|---:|---:|---:|---:|
| WQ-musique | metaqa | zero-shot | 0.4105 | +0.0139 | -0.0020 | 0.0099 | **0.0199** |
| WQ-musique | squad | in-domain | 0.9105 | +0.0003 | +0.0023 | 0.0016 | 0.0075 |
| WQ-musique | musique | zero-shot | 0.4106 | +0.0502 | +0.0512 | 0.0507 | **0.1014** |
| WQ-musique | hotpotqa | in-domain | 0.9134 | -0.0024 | -0.0016 | 0.0020 | 0.0075 |
| WQ-musique | 2wiki | in-domain | 0.9369 | +0.0038 | +0.0014 | 0.0029 | 0.0075 |
| WQ-musique | webqsp | in-domain | 0.6501 | -0.0074 | -0.0026 | 0.0055 | **0.0111** |
| WQ-hotpotqa | metaqa | zero-shot | 0.3834 | +0.0521 | +0.0322 | 0.0433 | **0.0866** |
| WQ-hotpotqa | squad | in-domain | 0.9111 | +0.0019 | +0.0013 | 0.0016 | 0.0075 |
| WQ-hotpotqa | musique | in-domain | 0.5941 | -0.0021 | +0.0037 | 0.0030 | 0.0075 |
| WQ-hotpotqa | hotpotqa | zero-shot | 0.8529 | -0.0041 | -0.0044 | 0.0043 | **0.0085** |
| WQ-hotpotqa | 2wiki | in-domain | 0.9420 | -0.0002 | +0.0012 | 0.0009 | 0.0075 |
| WQ-hotpotqa | webqsp | in-domain | 0.6536 | -0.0053 | -0.0135 | 0.0103 | **0.0205** |

Floors above 0.0075: 6 of 12 (WQ-musique metaqa, WQ-musique musique, WQ-musique webqsp, WQ-hotpotqa metaqa, WQ-hotpotqa hotpotqa, WQ-hotpotqa webqsp).
