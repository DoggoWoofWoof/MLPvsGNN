U1f (docs/U1F_RETRAIN_ON_GU1024.md; U1c's screen on GU_1024): zsp on GU_1024 and its step 4h pools against zsp on today's.

# scr-u1f-zsp-pair re-called under the seed null: **NO_GAIN** (filed: NO_GAIN)

R@5 of zsp's p@swa minus zsp's p@swa of the same split (zsp's card fits on today's graph and pools), on the six s1eval carves (95% question bootstrap), called with each read's floor: max(0.0075, twice the seed-only spread of step 1's seeds 1 and 2 against seed 0) (docs/SCREENS.md, section 2, and docs/U1C_RETRAIN_ON_U.md). D1, D2: the null's seed-only differences on the read. Bold: a changed call.

| split | dataset | read | zsp R@5 (today's) | zsp R@5 (U1f) | delta R@5 [95% CI] | D1, D2 | floor | filed | re-call |
|---|---|---|---:|---:|---|---|---:|---|---|
| L-musique | metaqa | in-domain | 0.7794 | 0.7834 | +0.0040 [+0.0019, +0.0061] | +0.0008, -0.0023 | 0.0075 | WITHIN | WITHIN |
| L-musique | squad | in-domain | 0.9125 | 0.9132 | +0.0008 [-0.0014, +0.0027] | +0.0001, +0.0013 | 0.0075 | WITHIN | WITHIN |
| L-musique | musique | zero-shot | 0.5111 | 0.4978 | -0.0133 [-0.0214, -0.0050] | +0.0503, -0.0079 | 0.0720 | LOSS | **WITHIN** |
| L-musique | hotpotqa | in-domain | 0.9065 | 0.8417 | -0.0648 [-0.0693, -0.0602] | +0.0014, +0.0016 | 0.0075 | LOSS | LOSS |
| L-musique | 2wiki | in-domain | 0.8724 | 0.8624 | -0.0100 [-0.0132, -0.0067] | -0.0002, -0.0007 | 0.0075 | LOSS | LOSS |
| L-musique | webqsp | zero-shot | 0.3245 | 0.2819 | -0.0427 [-0.0590, -0.0258] | -0.0070, -0.0018 | 0.0102 | LOSS | LOSS |
| L-hotpotqa | metaqa | in-domain | 0.7807 | 0.7873 | +0.0065 [+0.0046, +0.0085] | -0.0092, -0.0051 | 0.0149 | WITHIN | WITHIN |
| L-hotpotqa | squad | in-domain | 0.9109 | 0.9135 | +0.0026 [+0.0006, +0.0048] | +0.0002, +0.0007 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | musique | in-domain | 0.5647 | 0.5685 | +0.0038 [-0.0023, +0.0099] | +0.0015, +0.0023 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | hotpotqa | zero-shot | 0.8435 | 0.8082 | -0.0353 [-0.0401, -0.0308] | -0.0055, -0.0130 | 0.0200 | LOSS | LOSS |
| L-hotpotqa | 2wiki | in-domain | 0.8744 | 0.8617 | -0.0127 [-0.0161, -0.0092] | -0.0020, +0.0011 | 0.0075 | LOSS | LOSS |
| L-hotpotqa | webqsp | zero-shot | 0.2262 | 0.2414 | +0.0152 [-0.0002, +0.0301] | +0.0102, +0.0129 | 0.0233 | WITHIN | WITHIN |

Re-call: reads with a GAIN: 0 of 12; with a LOSS: 5. PROMISING needs at least one GAIN and no LOSS.
Each fit, filed -> re-called: L-musique NO_GAIN -> NO_GAIN; L-hotpotqa NO_GAIN -> NO_GAIN.
Changed calls: L-musique musique LOSS -> WITHIN.
Within 0.00015 of the floor (the call stands): none.
