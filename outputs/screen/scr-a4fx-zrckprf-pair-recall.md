A4fx (docs/A4FX_FAMILY_DROPOUT_ADOPTED.md): zrckprf (zrckpr trained with A4f's family dropout and the R1 cell (0.2 per question on each block, the KT term, the chain term and R1's term)) against zrckpr.

# scr-a4fx-zrckprf-pair re-called under the seed null: **NO_GAIN** (filed: NO_GAIN)

R@5 of zrckprf's p@swa minus zrckpr's p@swa of the same split (zrckpr's card fits), on the six s1eval carves (95% question bootstrap), called with each read's floor: max(0.0075, twice the seed-only spread of step 1's seeds 1 and 2 against seed 0) (docs/SCREENS.md, section 2, and docs/A4FX_FAMILY_DROPOUT_ADOPTED.md). D1, D2: the null's seed-only differences on the read. Bold: a changed call.

| split | dataset | read | zrckpr R@5 | zrckprf R@5 (A4fx's family dropout) | delta R@5 [95% CI] | D1, D2 | floor | filed | re-call |
|---|---|---|---:|---:|---|---|---:|---|---|
| L-musique | metaqa | in-domain | 0.8012 | 0.7994 | -0.0018 [-0.0032, -0.0004] | +0.0008, -0.0023 | 0.0075 | WITHIN | WITHIN |
| L-musique | squad | in-domain | 0.9127 | 0.9097 | -0.0030 [-0.0051, -0.0010] | +0.0001, +0.0013 | 0.0075 | WITHIN | WITHIN |
| L-musique | musique | zero-shot | 0.5197 | 0.3534 | -0.1663 [-0.1766, -0.1560] | +0.0503, -0.0079 | 0.0720 | LOSS | LOSS |
| L-musique | hotpotqa | in-domain | 0.9083 | 0.9038 | -0.0045 [-0.0069, -0.0022] | +0.0014, +0.0016 | 0.0075 | WITHIN | WITHIN |
| L-musique | 2wiki | in-domain | 0.9361 | 0.9310 | -0.0051 [-0.0067, -0.0035] | -0.0002, -0.0007 | 0.0075 | WITHIN | WITHIN |
| L-musique | webqsp | zero-shot | 0.3762 | 0.3542 | -0.0220 [-0.0345, -0.0100] | -0.0070, -0.0018 | 0.0102 | LOSS | LOSS |
| L-hotpotqa | metaqa | in-domain | 0.8040 | 0.7968 | -0.0073 [-0.0088, -0.0059] | -0.0092, -0.0051 | 0.0149 | WITHIN | WITHIN |
| L-hotpotqa | squad | in-domain | 0.9114 | 0.9109 | -0.0005 [-0.0029, +0.0018] | +0.0002, +0.0007 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | musique | in-domain | 0.5779 | 0.5702 | -0.0077 [-0.0134, -0.0020] | +0.0015, +0.0023 | 0.0075 | LOSS | LOSS |
| L-hotpotqa | hotpotqa | zero-shot | 0.8558 | 0.8612 | +0.0054 [+0.0022, +0.0086] | -0.0055, -0.0130 | 0.0200 | WITHIN | WITHIN |
| L-hotpotqa | 2wiki | in-domain | 0.9331 | 0.9281 | -0.0050 [-0.0068, -0.0032] | -0.0020, +0.0011 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | webqsp | zero-shot | 0.3411 | 0.3531 | +0.0120 [-0.0015, +0.0250] | +0.0102, +0.0129 | 0.0233 | WITHIN | WITHIN |

Re-call: reads with a GAIN: 0 of 12; with a LOSS: 3. PROMISING needs at least one GAIN and no LOSS.
Each fit, filed -> re-called: L-musique NO_GAIN -> NO_GAIN; L-hotpotqa NO_GAIN -> NO_GAIN.
Changed calls: none.
Within 0.00015 of the floor (the call stands): none.
