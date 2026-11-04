# DOB amendment A3: xtopo (topo_STRUCT dropped) at three seeds against the CPU base

- Own (ML1's rule, twelve reads): **NOT_ADOPTED**
- WWW (the six in-domain reads): **NOT_ADOPTED**

own reads (R@5 deltas at seeds 0, 1, 2):

| split | read | d0 | d1 | d2 | floor | call |
|---|---|---:|---:|---:|---:|---|
| WQ-musique | metaqa | +0.0738 | -0.0044 | -0.0591 | 0.1144 | WITHIN |
| WQ-musique | squad | +0.0008 | -0.0011 | -0.0023 | 0.0075 | WITHIN |
| WQ-musique | musique | +0.0009 | +0.0212 | +0.0196 | 0.0228 | WITHIN |
| WQ-musique | hotpotqa | -0.0006 | -0.0009 | +0.0031 | 0.0075 | WITHIN |
| WQ-musique | 2wiki | +0.0069 | -0.0042 | +0.0061 | 0.0108 | WITHIN |
| WQ-musique | webqsp | +0.0134 | -0.0004 | +0.0016 | 0.0131 | UNCONFIRMED_GAIN |
| WQ-hotpotqa | metaqa | -0.0101 | +0.0019 | +0.0377 | 0.0564 | WITHIN |
| WQ-hotpotqa | squad | -0.0009 | -0.0007 | -0.0016 | 0.0075 | WITHIN |
| WQ-hotpotqa | musique | +0.0035 | +0.0006 | +0.0002 | 0.0075 | WITHIN |
| WQ-hotpotqa | hotpotqa | +0.0025 | -0.0021 | +0.0014 | 0.0075 | WITHIN |
| WQ-hotpotqa | 2wiki | +0.0127 | -0.0045 | +0.0035 | 0.0145 | WITHIN |
| WQ-hotpotqa | webqsp | +0.0074 | -0.0094 | +0.0064 | 0.0149 | WITHIN |

WWW reads (R@5 deltas at seeds 0, 1, 2):

| split | read | d0 | d1 | d2 | floor | call |
|---|---|---:|---:|---:|---:|---|
| WQ-musique | hotpotqa | -0.0006 | -0.0009 | +0.0031 | 0.0075 | WITHIN |
| WQ-musique | 2wiki | +0.0069 | -0.0042 | +0.0061 | 0.0108 | WITHIN |
| WQ-hotpotqa | musique | +0.0035 | +0.0006 | +0.0002 | 0.0075 | WITHIN |
| WQ-hotpotqa | 2wiki | +0.0127 | -0.0045 | +0.0035 | 0.0145 | WITHIN |
| WQ-hotpotqa | musique-3hop | +0.0154 | -0.0070 | +0.0018 | 0.0270 | WITHIN |
| WQ-hotpotqa | musique-4hop | +0.0056 | -0.0093 | +0.0006 | 0.0119 | WITHIN |

