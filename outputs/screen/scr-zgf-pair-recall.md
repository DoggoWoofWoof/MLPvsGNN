# scr-zgf-pair re-called under the seed null: **NO_GAIN** (filed: NO_GAIN)

R@5 of zgf's p@swa minus zrm's p@swa of the same split (zrm's screen fits), on the six s1eval carves (95% question bootstrap), called with each read's floor: max(0.0075, twice the seed-only spread of step 1's seeds 1 and 2 against seed 0) (docs/SCREENS.md, section 2 and the twenty-sixth round). D1, D2: the null's seed-only differences on the read. Bold: a changed call.

| split | dataset | read | zrm R@5 | zgf R@5 | delta R@5 [95% CI] | D1, D2 | floor | filed | re-call |
|---|---|---|---:|---:|---|---|---:|---|---|
| L-musique | metaqa | in-domain | 0.7789 | 0.7813 | +0.0023 [+0.0010, +0.0036] | +0.0008, -0.0023 | 0.0075 | WITHIN | WITHIN |
| L-musique | squad | in-domain | 0.9111 | 0.9075 | -0.0035 [-0.0058, -0.0012] | +0.0001, +0.0013 | 0.0075 | WITHIN | WITHIN |
| L-musique | musique | zero-shot | 0.5231 | 0.5291 | +0.0060 [+0.0003, +0.0115] | +0.0503, -0.0079 | 0.0720 | WITHIN | WITHIN |
| L-musique | hotpotqa | in-domain | 0.9040 | 0.9024 | -0.0016 [-0.0042, +0.0010] | +0.0014, +0.0016 | 0.0075 | WITHIN | WITHIN |
| L-musique | 2wiki | in-domain | 0.8709 | 0.8752 | +0.0043 [+0.0024, +0.0063] | -0.0002, -0.0007 | 0.0075 | WITHIN | WITHIN |
| L-musique | webqsp | zero-shot | 0.3128 | 0.3118 | -0.0009 [-0.0103, +0.0087] | -0.0070, -0.0018 | 0.0102 | WITHIN | WITHIN |
| L-hotpotqa | metaqa | in-domain | 0.7801 | 0.7815 | +0.0015 [+0.0002, +0.0028] | -0.0092, -0.0051 | 0.0149 | WITHIN | WITHIN |
| L-hotpotqa | squad | in-domain | 0.9116 | 0.8972 | -0.0143 [-0.0175, -0.0112] | +0.0002, +0.0007 | 0.0075 | LOSS | LOSS |
| L-hotpotqa | musique | in-domain | 0.5600 | 0.5001 | -0.0599 [-0.0683, -0.0511] | +0.0015, +0.0023 | 0.0075 | LOSS | LOSS |
| L-hotpotqa | hotpotqa | zero-shot | 0.8457 | 0.8020 | -0.0437 [-0.0481, -0.0392] | -0.0055, -0.0130 | 0.0200 | LOSS | LOSS |
| L-hotpotqa | 2wiki | in-domain | 0.8681 | 0.8626 | -0.0055 [-0.0081, -0.0028] | -0.0020, +0.0011 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | webqsp | zero-shot | 0.2416 | 0.2407 | -0.0008 [-0.0122, +0.0108] | +0.0102, +0.0129 | 0.0233 | WITHIN | WITHIN |

Re-call: reads with a GAIN: 0 of 12; with a LOSS: 3. PROMISING needs at least one GAIN and no LOSS.
Each fit, filed -> re-called: L-musique NO_GAIN -> NO_GAIN; L-hotpotqa NO_GAIN -> NO_GAIN.
Changed calls: none.
Within 0.00015 of the floor (the call stands): none.
