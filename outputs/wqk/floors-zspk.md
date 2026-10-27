# WQ's floors on zspk

zspk fitted on each WQ split with seeds 0, 1 and 2; seeds 1 and 2 each compared with seed 0 on the six s1eval carves (p@swa, R@5). floor = max(0.0075, 2 sqrt((D1^2 + D2^2) / 2)). Bold: above 0.0075.

| split | dataset | read | seed 0 R@5 | D1 | D2 | spread | floor |
|---|---|---|---:|---:|---:|---:|---:|
| WQ-musique | metaqa | zero-shot | 0.3984 | -0.0303 | -0.0068 | 0.0220 | **0.0439** |
| WQ-musique | squad | in-domain | 0.9110 | -0.0006 | -0.0007 | 0.0007 | 0.0075 |
| WQ-musique | musique | zero-shot | 0.4590 | -0.0273 | +0.0272 | 0.0272 | **0.0545** |
| WQ-musique | hotpotqa | in-domain | 0.9058 | -0.0015 | +0.0006 | 0.0011 | 0.0075 |
| WQ-musique | 2wiki | in-domain | 0.9345 | -0.0008 | -0.0013 | 0.0011 | 0.0075 |
| WQ-musique | webqsp | in-domain | 0.6334 | -0.0123 | -0.0007 | 0.0087 | **0.0174** |
| WQ-hotpotqa | metaqa | zero-shot | 0.3384 | +0.0215 | +0.0034 | 0.0154 | **0.0308** |
| WQ-hotpotqa | squad | in-domain | 0.9114 | -0.0007 | +0.0000 | 0.0005 | 0.0075 |
| WQ-hotpotqa | musique | in-domain | 0.5794 | -0.0028 | +0.0057 | 0.0045 | **0.0090** |
| WQ-hotpotqa | hotpotqa | zero-shot | 0.8496 | -0.0012 | -0.0036 | 0.0027 | 0.0075 |
| WQ-hotpotqa | 2wiki | in-domain | 0.9330 | +0.0015 | +0.0023 | 0.0019 | 0.0075 |
| WQ-hotpotqa | webqsp | in-domain | 0.6153 | +0.0020 | +0.0060 | 0.0045 | **0.0089** |

Floors above 0.0075: 6 of 12 (WQ-musique metaqa, WQ-musique musique, WQ-musique webqsp, WQ-hotpotqa metaqa, WQ-hotpotqa musique, WQ-hotpotqa webqsp).
