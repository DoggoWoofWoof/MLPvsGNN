# GNS: zg6r with learned attention to the base GNN's own top five (a first pass) against zg6r (ML1), on PKO's pools

## Screen: NO_GAIN

counts: {"GAIN": 0, "LOSS": 0, "WITHIN": 10}
DEP's sixteen-read rule beside: screen ADOPT_FOR_FULL_RUN, full run None
next form: attention to the in-sample first-pass places closes for the GNN; next form: GNR's first pass cross-fitted (two half fits of the base per split), so that the training carves' places are out of sample as the s1eval carves' are, for GNR's columns and GNS's

| split | read | base s0, s1, s2 | GNS s0 | delta | GNT delta (rrf's places) | floor | call |
|---|---|---|---:|---:|---:|---:|---|
| WQ-musique | squad | 0.9142, 0.9145, 0.9163 | 0.9127 | -0.0015 | -0.0023 | 0.0075 | WITHIN |
| WQ-musique | hotpotqa | 0.8593, 0.8596, 0.8562 | 0.8548 | -0.0045 | -0.0014 | 0.0075 | WITHIN |
| WQ-musique | 2wiki | 0.9269, 0.9297, 0.9215 | 0.9268 | -0.0001 | +0.0024 | 0.0086 | WITHIN |
| WQ-musique | webqsp | 0.6632, 0.6586, 0.6593 | 0.6619 | -0.0013 | -0.0108 | 0.0085 | WITHIN |
| WQ-hotpotqa | squad | 0.9139, 0.9140, 0.9127 | 0.9138 | -0.0001 | -0.0028 | 0.0075 | WITHIN |
| WQ-hotpotqa | musique | 0.5912, 0.5988, 0.5939 | 0.5946 | +0.0034 | +0.0031 | 0.0114 | WITHIN |
| WQ-hotpotqa | 2wiki | 0.9296, 0.9290, 0.9276 | 0.9232 | -0.0064 | -0.0102 | 0.0075 | WITHIN |
| WQ-hotpotqa | webqsp | 0.6658, 0.6512, 0.6523 | 0.6508 | -0.0150 | -0.0201 | 0.0281 | WITHIN |
| WQ-hotpotqa | musique-3hop | 0.5333, 0.5443, 0.5430 | 0.5443 | +0.0110 | +0.0048 | 0.0207 | WITHIN |
| WQ-hotpotqa | musique-4hop | 0.3765, 0.3833, 0.3772 | 0.3716 | -0.0049 | +0.0012 | 0.0096 | WITHIN |
| WQ-musique | musique | 0.5086, 0.5039, 0.5134 | 0.5193 | +0.0107 | - | 0.0095 | GAIN (beside) |
| WQ-musique | metaqa | 0.3615, 0.3477, 0.3461 | 0.3858 | +0.0243 | - | 0.0292 | WITHIN (beside) |
| WQ-hotpotqa | hotpotqa | 0.8004, 0.7943, 0.7840 | 0.8022 | +0.0018 | - | 0.0247 | WITHIN (beside) |
| WQ-hotpotqa | metaqa | 0.3519, 0.3903, 0.3651 | 0.3388 | -0.0131 | - | 0.0574 | WITHIN (beside) |
| WQ-musique | musique-3hop | 0.4329, 0.4272, 0.4404 | 0.4351 | +0.0022 | - | 0.0133 | WITHIN (beside) |
| WQ-musique | musique-4hop | 0.3130, 0.3167, 0.3179 | 0.3210 | +0.0080 | - | 0.0087 | WITHIN (beside) |

GNT beside: screen NOT_ADOPTED, full run None

GNW beside: screen NOT_ADOPTED, full run None

GNR beside: screen NO_GAIN, full run None

GNX beside: screen NO_GAIN, full run None

First passes: WQ-musique check True, WQ-hotpotqa check True; columns: WQ-musique check True, places = rrf's top five on 0.000-0.209 of the questions; WQ-hotpotqa check True, places = rrf's top five on 0.000-0.176 of the questions
