# GNC: zg6r with GNC's NEXT input (anchors: the base GNN's own top five, cross-fitted) against zg6r (ML1), on PKO's pools

## Screen: INCOMPLETE

counts: {}
DEP's sixteen-read rule beside: screen INCOMPLETE, full run None
next form: what is missing is fixed in an amendment, before any number is read

| split | read | base s0, s1, s2 | GNC s0 | delta | GNR delta (in-sample anchors) | GNX delta (rrf's anchors) | floor | call |
|---|---|---|---:|---:|---:|---:|---:|---|
| WQ-musique | squad | 0.9142, 0.9145, 0.9163 | 0.9148 | +0.0006 | +0.0005 | +0.0003 | 0.0075 | WITHIN |
| WQ-musique | hotpotqa | 0.8593, 0.8596, 0.8562 | 0.8586 | -0.0007 | -0.0022 | +0.0004 | 0.0075 | WITHIN |
| WQ-musique | 2wiki | 0.9269, 0.9297, 0.9215 | 0.9269 | +0.0000 | -0.0006 | +0.0015 | 0.0086 | WITHIN |
| WQ-musique | webqsp | 0.6632, 0.6586, 0.6593 | 0.6508 | -0.0124 | -0.0015 | +0.0005 | 0.0085 | LOSS |
| WQ-musique | musique | 0.5086, 0.5039, 0.5134 | 0.5134 | +0.0048 | +0.0022 | +0.0147 | 0.0095 | WITHIN (beside) |
| WQ-musique | metaqa | 0.3615, 0.3477, 0.3461 | 0.3874 | +0.0259 | -0.0015 | +0.0245 | 0.0292 | WITHIN (beside) |
| WQ-musique | musique-3hop | 0.4329, 0.4272, 0.4404 | 0.4390 | +0.0061 | -0.0013 | +0.0127 | 0.0133 | WITHIN (beside) |
| WQ-musique | musique-4hop | 0.3130, 0.3167, 0.3179 | 0.3154 | +0.0025 | +0.0068 | +0.0173 | 0.0087 | WITHIN (beside) |

GNR beside (in-sample anchors): screen NO_GAIN, full run None

GNX beside (LTM's columns, rrf's anchors): screen NO_GAIN, full run NOT_ADOPTED
First passes: WQ-musique check True, WQ-hotpotqa check True; per carve in outputs/gnc/report.md
