# C1R: Claim 1 under WQ on PKO's pools, zrckpn against zg6r

**CLAIM1_NOT_ALL** (bar 0.9, 2000 resamples, seed-0 fits)

| split | dataset | read | MLP | GNN | RRF | GNN gain [lo, hi] | share [lo, hi] | clause |
|---|---|---|---:|---:|---:|---|---|---|
| WQ-musique | metaqa | zero-shot | 0.4126 | 0.3615 | 0.0047 | +0.3568 [+0.3478, +0.3658] | 1.143 [1.123, 1.165] | HOLDS |
| WQ-musique | squad | in-domain | 0.9138 | 0.9142 | 0.9054 | +0.0088 [+0.0055, +0.0120] | 0.962 [0.755, 1.244] | NOT_SHOWN |
| WQ-musique | musique | zero-shot | 0.5190 | 0.5086 | 0.4734 | +0.0352 [+0.0244, +0.0455] | 1.295 [1.091, 1.597] | HOLDS |
| WQ-musique | hotpotqa | in-domain | 0.8460 | 0.8593 | 0.6846 | +0.1747 [+0.1682, +0.1809] | 0.924 [0.907, 0.942] | HOLDS |
| WQ-musique | 2wiki | in-domain | 0.9155 | 0.9269 | 0.6079 | +0.3190 [+0.3141, +0.3236] | 0.964 [0.958, 0.971] | HOLDS |
| WQ-musique | webqsp | in-domain | 0.6457 | 0.6632 | 0.0535 | +0.6097 [+0.5849, +0.6314] | 0.971 [0.950, 0.994] | HOLDS |
| WQ-hotpotqa | metaqa | zero-shot | 0.3668 | 0.3519 | 0.0047 | +0.3472 [+0.3383, +0.3556] | 1.043 [1.020, 1.067] | HOLDS |
| WQ-hotpotqa | squad | in-domain | 0.9132 | 0.9139 | 0.9054 | +0.0085 [+0.0051, +0.0120] | 0.921 [0.686, 1.227] | NOT_SHOWN |
| WQ-hotpotqa | musique | in-domain | 0.5806 | 0.5912 | 0.4734 | +0.1177 [+0.1076, +0.1274] | 0.911 [0.858, 0.966] | NOT_SHOWN |
| WQ-hotpotqa | hotpotqa | zero-shot | 0.8193 | 0.8004 | 0.6846 | +0.1158 [+0.1090, +0.1223] | 1.163 [1.124, 1.206] | HOLDS |
| WQ-hotpotqa | 2wiki | in-domain | 0.9135 | 0.9296 | 0.6079 | +0.3217 [+0.3169, +0.3264] | 0.950 [0.943, 0.957] | HOLDS |
| WQ-hotpotqa | webqsp | in-domain | 0.6469 | 0.6658 | 0.0535 | +0.6123 [+0.5885, +0.6352] | 0.969 [0.946, 0.994] | HOLDS |

Zero-shot clauses (R@5): excess = zero-shot gap minus the allowed in-domain gap (floored at 0).

| read | in-domain split | zs gap | allowed | excess [lo, hi] | call |
|---|---|---:|---:|---|---|
| WQ-musique musique | WQ-hotpotqa | -0.0104 | 0.0105 | -0.0209 [-0.0303, -0.0117] | HOLDS |
| WQ-hotpotqa hotpotqa | WQ-musique | -0.0189 | 0.0132 | -0.0321 [-0.0375, -0.0270] | HOLDS |
| WQ-musique metaqa | none (allowed 0) | -0.0511 | 0.0000 | -0.0511 [-0.0581, -0.0441] | HOLDS |
| WQ-hotpotqa metaqa | none (allowed 0) | -0.0148 | 0.0000 | -0.0148 [-0.0231, -0.0070] | HOLDS |
