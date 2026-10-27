# WQ's floors on zrckp

zrckp fitted on each WQ split with seeds 0, 1 and 2; seeds 1 and 2 each compared with seed 0 on the six s1eval carves (p@swa, R@5). floor = max(0.0075, 2 sqrt((D1^2 + D2^2) / 2)). Bold: above 0.0075.

| split | dataset | read | seed 0 R@5 | D1 | D2 | spread | floor |
|---|---|---|---:|---:|---:|---:|---:|
| WQ-musique | metaqa | zero-shot | 0.4393 | -0.0204 | -0.0389 | 0.0311 | **0.0621** |
| WQ-musique | squad | in-domain | 0.9106 | -0.0014 | -0.0002 | 0.0010 | 0.0075 |
| WQ-musique | musique | zero-shot | 0.4972 | -0.0335 | -0.0219 | 0.0283 | **0.0566** |
| WQ-musique | hotpotqa | in-domain | 0.9051 | -0.0053 | -0.0031 | 0.0043 | **0.0087** |
| WQ-musique | 2wiki | in-domain | 0.9315 | -0.0035 | -0.0056 | 0.0047 | **0.0093** |
| WQ-musique | webqsp | in-domain | 0.6433 | -0.0012 | +0.0023 | 0.0018 | 0.0075 |
| WQ-hotpotqa | metaqa | zero-shot | 0.3927 | +0.0267 | +0.0220 | 0.0245 | **0.0489** |
| WQ-hotpotqa | squad | in-domain | 0.9117 | -0.0003 | -0.0001 | 0.0002 | 0.0075 |
| WQ-hotpotqa | musique | in-domain | 0.5743 | +0.0024 | +0.0043 | 0.0035 | 0.0075 |
| WQ-hotpotqa | hotpotqa | zero-shot | 0.8525 | +0.0022 | +0.0059 | 0.0045 | **0.0089** |
| WQ-hotpotqa | 2wiki | in-domain | 0.9299 | -0.0015 | +0.0013 | 0.0014 | 0.0075 |
| WQ-hotpotqa | webqsp | in-domain | 0.6382 | +0.0035 | +0.0011 | 0.0026 | 0.0075 |

Floors above 0.0075: 6 of 12 (WQ-musique metaqa, WQ-musique musique, WQ-musique hotpotqa, WQ-musique 2wiki, WQ-hotpotqa metaqa, WQ-hotpotqa hotpotqa).
