# GNR: zg6r with GNR's NEXT input (anchors: the base GNN's own top five) against zg6r (ML1), on PKO's pools

## Screen: NO_GAIN

counts: {"GAIN": 0, "LOSS": 0, "WITHIN": 10}
DEP's sixteen-read rule beside: screen NO_GAIN, full run None
next form: the in-sample anchors close; next form, in its own file: the first pass cross-fitted (two half fits of the base per split), so that the training carves' anchors are out of sample as the s1eval carves' are

| split | read | base s0, s1, s2 | GNR s0 | delta | GNX delta (rrf's anchors) | floor | call |
|---|---|---|---:|---:|---:|---:|---|
| WQ-musique | squad | 0.9142, 0.9145, 0.9163 | 0.9147 | +0.0005 | +0.0003 | 0.0075 | WITHIN |
| WQ-musique | hotpotqa | 0.8593, 0.8596, 0.8562 | 0.8571 | -0.0022 | +0.0004 | 0.0075 | WITHIN |
| WQ-musique | 2wiki | 0.9269, 0.9297, 0.9215 | 0.9263 | -0.0006 | +0.0015 | 0.0086 | WITHIN |
| WQ-musique | webqsp | 0.6632, 0.6586, 0.6593 | 0.6617 | -0.0015 | +0.0005 | 0.0085 | WITHIN |
| WQ-hotpotqa | squad | 0.9139, 0.9140, 0.9127 | 0.9147 | +0.0008 | +0.0009 | 0.0075 | WITHIN |
| WQ-hotpotqa | musique | 0.5912, 0.5988, 0.5939 | 0.5941 | +0.0029 | +0.0002 | 0.0114 | WITHIN |
| WQ-hotpotqa | 2wiki | 0.9296, 0.9290, 0.9276 | 0.9262 | -0.0034 | -0.0026 | 0.0075 | WITHIN |
| WQ-hotpotqa | webqsp | 0.6658, 0.6512, 0.6523 | 0.6626 | -0.0032 | -0.0022 | 0.0281 | WITHIN |
| WQ-hotpotqa | musique-3hop | 0.5333, 0.5443, 0.5430 | 0.5382 | +0.0048 | +0.0035 | 0.0207 | WITHIN |
| WQ-hotpotqa | musique-4hop | 0.3765, 0.3833, 0.3772 | 0.3778 | +0.0012 | -0.0062 | 0.0096 | WITHIN |
| WQ-musique | musique | 0.5086, 0.5039, 0.5134 | 0.5108 | +0.0022 | +0.0147 | 0.0095 | WITHIN (beside) |
| WQ-musique | metaqa | 0.3615, 0.3477, 0.3461 | 0.3600 | -0.0015 | +0.0245 | 0.0292 | WITHIN (beside) |
| WQ-hotpotqa | hotpotqa | 0.8004, 0.7943, 0.7840 | 0.8070 | +0.0066 | +0.0004 | 0.0247 | WITHIN (beside) |
| WQ-hotpotqa | metaqa | 0.3519, 0.3903, 0.3651 | 0.3476 | -0.0043 | +0.0110 | 0.0574 | WITHIN (beside) |
| WQ-musique | musique-3hop | 0.4329, 0.4272, 0.4404 | 0.4316 | -0.0013 | +0.0127 | 0.0133 | WITHIN (beside) |
| WQ-musique | musique-4hop | 0.3130, 0.3167, 0.3179 | 0.3198 | +0.0068 | +0.0173 | 0.0087 | WITHIN (beside) |

GNX beside (LTM's columns, rrf's anchors): screen NO_GAIN, full run None
First passes: WQ-musique check True, WQ-hotpotqa check True; per carve in outputs/gnr/report.md
