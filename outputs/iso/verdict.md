# ISO: zg6r-m (zg6r from its own encoder start) against zg6r (from zspk's), on PKO's pools

## KEEP

counts: {"GAIN": 1, "LOSS": 1, "UNCONFIRMED_GAIN": 0, "WITHIN": 10, "ISO_LOSS": 1}

| split | dataset | d0 | d1 | d2 | mean | floor | ML1 call | ISO loss |
|---|---|---:|---:|---:|---:|---:|---|---|
| WQ-musique | metaqa | -0.0446 | +0.0057 | +0.0166 | -0.0074 | 0.0621 | WITHIN | no |
| WQ-musique | squad | -0.0022 | +0.0002 | -0.0009 | -0.0010 | 0.0075 | WITHIN | no |
| WQ-musique | musique | +0.0202 | +0.0249 | +0.0241 | +0.0231 | 0.0110 | GAIN | no |
| WQ-musique | hotpotqa | -0.0081 | -0.0075 | -0.0005 | -0.0054 | 0.0075 | LOSS | yes |
| WQ-musique | 2wiki | -0.0017 | -0.0027 | +0.0067 | +0.0008 | 0.0075 | WITHIN | no |
| WQ-musique | webqsp | +0.0142 | +0.0074 | +0.0241 | +0.0152 | 0.0142 | WITHIN | no |
| WQ-hotpotqa | metaqa | -0.0628 | -0.0786 | -0.0325 | -0.0580 | 0.0637 | WITHIN | no |
| WQ-hotpotqa | squad | -0.0006 | -0.0007 | +0.0003 | -0.0003 | 0.0075 | WITHIN | no |
| WQ-hotpotqa | musique | +0.0056 | -0.0089 | -0.0038 | -0.0024 | 0.0126 | WITHIN | no |
| WQ-hotpotqa | hotpotqa | -0.0117 | -0.0127 | -0.0019 | -0.0088 | 0.0200 | WITHIN | no |
| WQ-hotpotqa | 2wiki | -0.0031 | -0.0056 | +0.0002 | -0.0028 | 0.0075 | WITHIN | no |
| WQ-hotpotqa | webqsp | +0.0214 | +0.0261 | +0.0539 | +0.0338 | 0.0292 | WITHIN | no |

Beside, deciding nothing (seed means; kept = (zg6r-m - zspk) / (zg6r - zspk)):

| split | dataset | zspk | zg6r | zg6r-m | kept |
|---|---|---:|---:|---:|---:|
| WQ-musique | metaqa | 0.3574 | 0.3518 | 0.3443 |  |
| WQ-musique | squad | 0.9139 | 0.9150 | 0.9140 |  |
| WQ-musique | musique | 0.4934 | 0.5086 | 0.5317 | 2.511 |
| WQ-musique | hotpotqa | 0.8440 | 0.8584 | 0.8530 | 0.626 |
| WQ-musique | 2wiki | 0.9105 | 0.9260 | 0.9268 | 1.049 |
| WQ-musique | webqsp | 0.6403 | 0.6604 | 0.6756 | 1.76 |
| WQ-hotpotqa | metaqa | 0.3199 | 0.3691 | 0.3111 | -0.179 |
| WQ-hotpotqa | squad | 0.9133 | 0.9135 | 0.9132 |  |
| WQ-hotpotqa | musique | 0.5788 | 0.5946 | 0.5923 | 0.851 |
| WQ-hotpotqa | hotpotqa | 0.8091 | 0.7929 | 0.7841 |  |
| WQ-hotpotqa | 2wiki | 0.9143 | 0.9287 | 0.9259 | 0.804 |
| WQ-hotpotqa | webqsp | 0.6345 | 0.6564 | 0.6902 | 2.541 |
