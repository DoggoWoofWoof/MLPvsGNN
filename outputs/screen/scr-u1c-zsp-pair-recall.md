U1c (docs/U1C_RETRAIN_ON_U.md): zsp on U1d's graph and step 4h's pools against zsp on today's.

# scr-u1c-zsp-pair re-called under the seed null: **NO_GAIN** (filed: NO_GAIN)

R@5 of zsp's p@swa minus zsp's p@swa of the same split (zsp's card fits on today's graph and pools), on the six s1eval carves (95% question bootstrap), called with each read's floor: max(0.0075, twice the seed-only spread of step 1's seeds 1 and 2 against seed 0) (docs/SCREENS.md, section 2, and docs/U1C_RETRAIN_ON_U.md). D1, D2: the null's seed-only differences on the read. Bold: a changed call.

| split | dataset | read | zsp R@5 (today's) | zsp R@5 (U1c) | delta R@5 [95% CI] | D1, D2 | floor | filed | re-call |
|---|---|---|---:|---:|---|---|---:|---|---|
| L-musique | metaqa | in-domain | 0.7794 | 0.7835 | +0.0041 [+0.0021, +0.0061] | +0.0008, -0.0023 | 0.0075 | WITHIN | WITHIN |
| L-musique | squad | in-domain | 0.9125 | 0.9132 | +0.0008 [-0.0014, +0.0029] | +0.0001, +0.0013 | 0.0075 | WITHIN | WITHIN |
| L-musique | musique | zero-shot | 0.5111 | 0.4922 | -0.0189 [-0.0280, -0.0103] | +0.0503, -0.0079 | 0.0720 | LOSS | **WITHIN** |
| L-musique | hotpotqa | in-domain | 0.9065 | 0.8431 | -0.0634 [-0.0677, -0.0589] | +0.0014, +0.0016 | 0.0075 | LOSS | LOSS |
| L-musique | 2wiki | in-domain | 0.8724 | 0.8612 | -0.0112 [-0.0145, -0.0079] | -0.0002, -0.0007 | 0.0075 | LOSS | LOSS |
| L-musique | webqsp | zero-shot | 0.3245 | 0.2683 | -0.0562 [-0.0719, -0.0406] | -0.0070, -0.0018 | 0.0102 | LOSS | LOSS |
| L-hotpotqa | metaqa | in-domain | 0.7807 | 0.7863 | +0.0056 [+0.0036, +0.0076] | -0.0092, -0.0051 | 0.0149 | WITHIN | WITHIN |
| L-hotpotqa | squad | in-domain | 0.9109 | 0.9121 | +0.0012 [-0.0010, +0.0034] | +0.0002, +0.0007 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | musique | in-domain | 0.5647 | 0.5613 | -0.0034 [-0.0103, +0.0032] | +0.0015, +0.0023 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | hotpotqa | zero-shot | 0.8435 | 0.8038 | -0.0397 [-0.0446, -0.0350] | -0.0055, -0.0130 | 0.0200 | LOSS | LOSS |
| L-hotpotqa | 2wiki | in-domain | 0.8744 | 0.8605 | -0.0139 [-0.0172, -0.0105] | -0.0020, +0.0011 | 0.0075 | LOSS | LOSS |
| L-hotpotqa | webqsp | zero-shot | 0.2262 | 0.2364 | +0.0102 [-0.0041, +0.0241] | +0.0102, +0.0129 | 0.0233 | WITHIN | WITHIN |

Re-call: reads with a GAIN: 0 of 12; with a LOSS: 5. PROMISING needs at least one GAIN and no LOSS.
Each fit, filed -> re-called: L-musique NO_GAIN -> NO_GAIN; L-hotpotqa NO_GAIN -> NO_GAIN.
Changed calls: L-musique musique LOSS -> WITHIN.
Within 0.00015 of the floor (the call stands): none.
