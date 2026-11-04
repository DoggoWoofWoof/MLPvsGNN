# LTX: zg6r with the cross-fitted next-hop term against zg6r (ML1), on PKO's pools

## Screen: NO_GAIN

counts: {"GAIN": 0, "LOSS": 0, "WITHIN": 10}
DEP's sixteen-read rule beside: screen ADOPT_FOR_FULL_RUN, full run None
next form: the cross-fitted next-hop term closes for the GNN; next form: LTG's cumulative residual (docs/LTG_NEXT_HOP_GNN.md section 6), in its own file

| split | read | base s0, s1, s2 | LTX s0 | delta | floor | call |
|---|---|---|---:|---:|---:|---|
| WQ-musique | squad | 0.9142, 0.9145, 0.9163 | 0.9158 | +0.0016 | 0.0075 | WITHIN |
| WQ-musique | hotpotqa | 0.8593, 0.8596, 0.8562 | 0.8631 | +0.0038 | 0.0075 | WITHIN |
| WQ-musique | 2wiki | 0.9269, 0.9297, 0.9215 | 0.9319 | +0.0050 | 0.0086 | WITHIN |
| WQ-musique | webqsp | 0.6632, 0.6586, 0.6593 | 0.6713 | +0.0081 | 0.0085 | WITHIN |
| WQ-hotpotqa | squad | 0.9139, 0.9140, 0.9127 | 0.9132 | -0.0007 | 0.0075 | WITHIN |
| WQ-hotpotqa | musique | 0.5912, 0.5988, 0.5939 | 0.5991 | +0.0079 | 0.0114 | WITHIN |
| WQ-hotpotqa | 2wiki | 0.9296, 0.9290, 0.9276 | 0.9309 | +0.0013 | 0.0075 | WITHIN |
| WQ-hotpotqa | webqsp | 0.6658, 0.6512, 0.6523 | 0.6598 | -0.0060 | 0.0281 | WITHIN |
| WQ-hotpotqa | musique-3hop | 0.5333, 0.5443, 0.5430 | 0.5404 | +0.0070 | 0.0207 | WITHIN |
| WQ-hotpotqa | musique-4hop | 0.3765, 0.3833, 0.3772 | 0.3846 | +0.0080 | 0.0096 | WITHIN |
| WQ-musique | musique | 0.5086, 0.5039, 0.5134 | 0.5163 | +0.0077 | 0.0095 | WITHIN (beside) |
| WQ-musique | metaqa | 0.3615, 0.3477, 0.3461 | 0.3605 | -0.0010 | 0.0292 | WITHIN (beside) |
| WQ-hotpotqa | hotpotqa | 0.8004, 0.7943, 0.7840 | 0.8023 | +0.0019 | 0.0247 | WITHIN (beside) |
| WQ-hotpotqa | metaqa | 0.3519, 0.3903, 0.3651 | 0.3919 | +0.0400 | 0.0574 | WITHIN (beside) |
| WQ-musique | musique-3hop | 0.4329, 0.4272, 0.4404 | 0.4386 | +0.0057 | 0.0133 | WITHIN (beside) |
| WQ-musique | musique-4hop | 0.3130, 0.3167, 0.3179 | 0.3235 | +0.0105 | 0.0087 | GAIN (beside) |

next-hop loss (train, last window): scr-ltx-xf-s0 18396 steps, {"step": 18396, "main": 1.428743, "next_hop": 0.727046, "senders": 7.02}; scr-ltx-xf-s0-hp 17155 steps, {"step": 17155, "main": 1.645148, "next_hop": 0.62528, "senders": 6.206}

LTG beside: screen NOT_ADOPTED, full run None

GNX beside: screen NO_GAIN, full run None

GNW beside: screen NOT_ADOPTED, full run None
