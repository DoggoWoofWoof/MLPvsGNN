# GNW: zg6r with the pair scores as message weights against zg6r (ML1), on PKO's pools

## Screen: NOT_ADOPTED

counts: {"GAIN": 0, "LOSS": 1, "WITHIN": 9}
DEP's sixteen-read rule beside: screen NOT_ADOPTED, full run None
next form: the pair scores as fixed message weights close for the GNN; next form: learned attention to the top five, the pair scores as its bias

| split | read | base s0, s1, s2 | GNW s0 | delta | floor | call |
|---|---|---|---:|---:|---:|---|
| WQ-musique | squad | 0.9142, 0.9145, 0.9163 | 0.9133 | -0.0009 | 0.0075 | WITHIN |
| WQ-musique | hotpotqa | 0.8593, 0.8596, 0.8562 | 0.8550 | -0.0043 | 0.0075 | WITHIN |
| WQ-musique | 2wiki | 0.9269, 0.9297, 0.9215 | 0.9276 | +0.0007 | 0.0086 | WITHIN |
| WQ-musique | webqsp | 0.6632, 0.6586, 0.6593 | 0.6508 | -0.0124 | 0.0085 | LOSS |
| WQ-hotpotqa | squad | 0.9139, 0.9140, 0.9127 | 0.9120 | -0.0019 | 0.0075 | WITHIN |
| WQ-hotpotqa | musique | 0.5912, 0.5988, 0.5939 | 0.5981 | +0.0069 | 0.0114 | WITHIN |
| WQ-hotpotqa | 2wiki | 0.9296, 0.9290, 0.9276 | 0.9277 | -0.0019 | 0.0075 | WITHIN |
| WQ-hotpotqa | webqsp | 0.6658, 0.6512, 0.6523 | 0.6527 | -0.0131 | 0.0281 | WITHIN |
| WQ-hotpotqa | musique-3hop | 0.5333, 0.5443, 0.5430 | 0.5373 | +0.0039 | 0.0207 | WITHIN |
| WQ-hotpotqa | musique-4hop | 0.3765, 0.3833, 0.3772 | 0.3784 | +0.0019 | 0.0096 | WITHIN |
| WQ-musique | musique | 0.5086, 0.5039, 0.5134 | 0.5251 | +0.0165 | 0.0095 | GAIN (beside) |
| WQ-musique | metaqa | 0.3615, 0.3477, 0.3461 | 0.3630 | +0.0015 | 0.0292 | WITHIN (beside) |
| WQ-hotpotqa | hotpotqa | 0.8004, 0.7943, 0.7840 | 0.7984 | -0.0020 | 0.0247 | WITHIN (beside) |
| WQ-hotpotqa | metaqa | 0.3519, 0.3903, 0.3651 | 0.3525 | +0.0006 | 0.0574 | WITHIN (beside) |
| WQ-musique | musique-3hop | 0.4329, 0.4272, 0.4404 | 0.4526 | +0.0197 | 0.0133 | GAIN (beside) |
| WQ-musique | musique-4hop | 0.3130, 0.3167, 0.3179 | 0.3377 | +0.0247 | 0.0087 | GAIN (beside) |

GNP beside: screen NO_GAIN, full run None

GNX beside: screen NO_GAIN, full run None

GNA beside: screen NO_GAIN, full run None
