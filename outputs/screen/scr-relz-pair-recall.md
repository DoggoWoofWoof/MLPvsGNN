# scr-relz-pair re-called under the seed null: **MIXED** (filed: MIXED)

R@5 of relz's p@swa minus rel's p@swa of the same split (rel's screen fits), on the six s1eval carves (95% question bootstrap), called with each read's floor: max(0.0075, twice the seed-only spread of step 1's seeds 1 and 2 against seed 0) (docs/SCREENS.md, section 2 and the tenth round). D1, D2: the null's seed-only differences on the read. Bold: a changed call.

| split | dataset | read | rel R@5 | relz R@5 | delta R@5 [95% CI] | D1, D2 | floor | filed | re-call |
|---|---|---|---:|---:|---|---|---:|---|---|
| L-musique | metaqa | in-domain | 0.7253 | 0.7058 | -0.0196 [-0.0229, -0.0164] | +0.0008, -0.0023 | 0.0075 | LOSS | LOSS |
| L-musique | squad | in-domain | 0.9091 | 0.9142 | +0.0051 [+0.0028, +0.0073] | +0.0001, +0.0013 | 0.0075 | WITHIN | WITHIN |
| L-musique | musique | zero-shot | 0.3564 | 0.5127 | +0.1563 [+0.1450, +0.1672] | +0.0503, -0.0079 | 0.0720 | GAIN | GAIN |
| L-musique | hotpotqa | in-domain | 0.9020 | 0.9038 | +0.0018 [-0.0009, +0.0045] | +0.0014, +0.0016 | 0.0075 | WITHIN | WITHIN |
| L-musique | 2wiki | in-domain | 0.8720 | 0.8725 | +0.0005 [-0.0015, +0.0025] | -0.0002, -0.0007 | 0.0075 | WITHIN | WITHIN |
| L-musique | webqsp | zero-shot | 0.2414 | 0.2541 | +0.0127 [-0.0036, +0.0292] | -0.0070, -0.0018 | 0.0102 | WITHIN | WITHIN |
| L-hotpotqa | metaqa | in-domain | 0.7217 | 0.6975 | -0.0242 [-0.0280, -0.0205] | -0.0092, -0.0051 | 0.0149 | LOSS | LOSS |
| L-hotpotqa | squad | in-domain | 0.9105 | 0.9135 | +0.0030 [+0.0009, +0.0052] | +0.0002, +0.0007 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | musique | in-domain | 0.5603 | 0.5654 | +0.0051 [-0.0014, +0.0117] | +0.0015, +0.0023 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | hotpotqa | zero-shot | 0.8453 | 0.8336 | -0.0117 [-0.0155, -0.0080] | -0.0055, -0.0130 | 0.0200 | LOSS | **WITHIN** |
| L-hotpotqa | 2wiki | in-domain | 0.8688 | 0.8669 | -0.0018 [-0.0041, +0.0005] | -0.0020, +0.0011 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | webqsp | zero-shot | 0.1902 | 0.2503 | +0.0601 [+0.0424, +0.0778] | +0.0102, +0.0129 | 0.0233 | GAIN | GAIN |

Re-call: reads with a GAIN: 2 of 12; with a LOSS: 2. PROMISING needs at least one GAIN and no LOSS.
Each fit, filed -> re-called: L-musique MIXED -> MIXED; L-hotpotqa MIXED -> MIXED.
Changed calls: L-hotpotqa hotpotqa LOSS -> WITHIN.
Within 0.00015 of the floor (the call stands): none.
