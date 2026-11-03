# DOB: zrckp with one input block dropped, against zrckp, on PKO's pools, on the host's CPU

## CALLED

| drop | block | call | GAIN reads | LOSS reads | mean delta |
|---|---|---|---|---|---:|
| xdcos | dense_cos | MIXED | WQ-musique musique, WQ-hotpotqa metaqa | WQ-musique hotpotqa, WQ-musique 2wiki | +0.0082 |
| xtopo | topo_STRUCT | HURTS | WQ-musique webqsp |  | +0.0092 |
| xdepth | depth_STRUCT | HURTS | WQ-hotpotqa metaqa |  | +0.0111 |
| xseed | SEED | HURTS | WQ-hotpotqa metaqa |  | +0.0056 |
| xwalk | WALK | HURTS | WQ-musique webqsp |  | +0.0049 |
| xdists | DISTS | IDLE |  |  | +0.0022 |
| xwalkf | WALKF | HURTS | WQ-musique musique |  | +0.0019 |

Seed-0 deltas against the CPU base (R@5); floor = max(0.0075, 2 x the base's spread):

| split | dataset | base s0, s1, s2 | floor | xdcos | xtopo | xdepth | xseed | xwalk | xdists | xwalkf |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| WQ-musique | metaqa | 0.3406, 0.3670, 0.3860 | 0.0743 | +0.0642 | +0.0738 | +0.0618 | +0.0478 | +0.0353 | +0.0193 | -0.0176 |
| WQ-musique | squad | 0.9127, 0.9122, 0.9138 | 0.0075 | +0.0007 | +0.0008 | +0.0009 | -0.0015 | +0.0005 | +0.0003 | -0.0003 |
| WQ-musique | musique | 0.5152, 0.5086, 0.5134 | 0.0097 | +0.0132* | +0.0009 | +0.0055 | -0.0073 | +0.0001 | -0.0032 | +0.0186* |
| WQ-musique | hotpotqa | 0.8411, 0.8424, 0.8388 | 0.0075 | -0.0105* | -0.0006 | +0.0028 | +0.0006 | +0.0033 | +0.0007 | +0.0007 |
| WQ-musique | 2wiki | 0.9070, 0.9079, 0.9049 | 0.0075 | -0.0080* | +0.0069 | +0.0027 | +0.0019 | +0.0034 | +0.0019 | -0.0063 |
| WQ-musique | webqsp | 0.6522, 0.6583, 0.6566 | 0.0106 | -0.0082 | +0.0134* | +0.0061 | +0.0000 | +0.0182* | +0.0086 | +0.0078 |
| WQ-hotpotqa | metaqa | 0.3847, 0.4037, 0.3797 | 0.0278 | +0.0508* | -0.0101 | +0.0334* | +0.0369* | -0.0163 | -0.0011 | +0.0139 |
| WQ-hotpotqa | squad | 0.9138, 0.9129, 0.9133 | 0.0075 | -0.0006 | -0.0009 | +0.0005 | -0.0023 | -0.0007 | -0.0013 | +0.0005 |
| WQ-hotpotqa | musique | 0.5749, 0.5718, 0.5752 | 0.0075 | -0.0063 | +0.0035 | +0.0000 | -0.0036 | -0.0032 | -0.0030 | -0.0045 |
| WQ-hotpotqa | hotpotqa | 0.8103, 0.8099, 0.8101 | 0.0075 | -0.0038 | +0.0025 | +0.0044 | -0.0040 | +0.0035 | -0.0010 | -0.0063 |
| WQ-hotpotqa | 2wiki | 0.8987, 0.9082, 0.9063 | 0.0172 | +0.0065 | +0.0127 | +0.0075 | +0.0023 | +0.0059 | +0.0035 | +0.0080 |
| WQ-hotpotqa | webqsp | 0.6446, 0.6524, 0.6514 | 0.0146 | +0.0001 | +0.0074 | +0.0075 | -0.0031 | +0.0089 | +0.0023 | +0.0084 |

\* a GAIN or a LOSS at the read's floor.

Beside, deciding nothing: the CPU base against ML1's zrckp on the GPU (seed means):

| split | dataset | CPU base | ML1 zrckp (GPU) |
|---|---|---:|---:|
| WQ-musique | metaqa | 0.3645 | 0.4008 |
| WQ-musique | squad | 0.9129 | 0.9133 |
| WQ-musique | musique | 0.5124 | 0.5178 |
| WQ-musique | hotpotqa | 0.8408 | 0.8419 |
| WQ-musique | 2wiki | 0.9066 | 0.9031 |
| WQ-musique | webqsp | 0.6557 | 0.6476 |
| WQ-hotpotqa | metaqa | 0.3894 | 0.3931 |
| WQ-hotpotqa | squad | 0.9133 | 0.9139 |
| WQ-hotpotqa | musique | 0.5740 | 0.5778 |
| WQ-hotpotqa | hotpotqa | 0.8101 | 0.8100 |
| WQ-hotpotqa | 2wiki | 0.9044 | 0.9037 |
| WQ-hotpotqa | webqsp | 0.6495 | 0.6600 |

Amendment A2: xrank withdrawn before any xrank fit trained (zrckp's model z-scores every input against the retrieved rows, which the rank block's rrf column marks); seven drops, 20 fits, 18 compares.
