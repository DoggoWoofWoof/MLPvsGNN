A3 (docs/A3_VSHARE_LEAN.md): zrckpv (zrckp trained with A3's share augmentation of zret's reference) against zrckp.

# scr-a3-zrckpv-pair re-called under the seed null: **NO_GAIN** (filed: MIXED)

R@5 of zrckpv's p@swa minus zrckp's p@swa of the same split (zrckp's card fits), on the six s1eval carves (95% question bootstrap), called with each read's floor: max(0.0075, twice the seed-only spread of step 1's seeds 1 and 2 against seed 0) (docs/SCREENS.md, section 2, and docs/A3_VSHARE_LEAN.md). D1, D2: the null's seed-only differences on the read. Bold: a changed call.

| split | dataset | read | zrckp R@5 | zrckpv R@5 (A3's share augmentation) | delta R@5 [95% CI] | D1, D2 | floor | filed | re-call |
|---|---|---|---:|---:|---|---|---:|---|---|
| L-musique | metaqa | in-domain | 0.8019 | 0.8013 | -0.0006 [-0.0015, +0.0002] | +0.0008, -0.0023 | 0.0075 | WITHIN | WITHIN |
| L-musique | squad | in-domain | 0.9116 | 0.9109 | -0.0008 [-0.0024, +0.0010] | +0.0001, +0.0013 | 0.0075 | WITHIN | WITHIN |
| L-musique | musique | zero-shot | 0.5121 | 0.5217 | +0.0095 [+0.0047, +0.0141] | +0.0503, -0.0079 | 0.0720 | GAIN | **WITHIN** |
| L-musique | hotpotqa | in-domain | 0.9065 | 0.9058 | -0.0007 [-0.0028, +0.0014] | +0.0014, +0.0016 | 0.0075 | WITHIN | WITHIN |
| L-musique | 2wiki | in-domain | 0.9347 | 0.9361 | +0.0014 [+0.0000, +0.0028] | -0.0002, -0.0007 | 0.0075 | WITHIN | WITHIN |
| L-musique | webqsp | zero-shot | 0.3627 | 0.3657 | +0.0030 [-0.0063, +0.0129] | -0.0070, -0.0018 | 0.0102 | WITHIN | WITHIN |
| L-hotpotqa | metaqa | in-domain | 0.8044 | 0.8035 | -0.0009 [-0.0017, -0.0001] | -0.0092, -0.0051 | 0.0149 | WITHIN | WITHIN |
| L-hotpotqa | squad | in-domain | 0.9127 | 0.9118 | -0.0009 [-0.0028, +0.0008] | +0.0002, +0.0007 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | musique | in-domain | 0.5745 | 0.5757 | +0.0013 [-0.0037, +0.0062] | +0.0015, +0.0023 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | hotpotqa | zero-shot | 0.8513 | 0.8478 | -0.0035 [-0.0065, -0.0005] | -0.0055, -0.0130 | 0.0200 | WITHIN | WITHIN |
| L-hotpotqa | 2wiki | in-domain | 0.9350 | 0.9350 | -0.0000 [-0.0015, +0.0016] | -0.0020, +0.0011 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | webqsp | zero-shot | 0.3475 | 0.3347 | -0.0128 [-0.0229, -0.0026] | +0.0102, +0.0129 | 0.0233 | LOSS | **WITHIN** |

Re-call: reads with a GAIN: 0 of 12; with a LOSS: 0. PROMISING needs at least one GAIN and no LOSS.
Each fit, filed -> re-called: L-musique PROMISING -> NO_GAIN; L-hotpotqa NO_GAIN -> NO_GAIN.
Changed calls: L-musique musique GAIN -> WITHIN; L-hotpotqa webqsp LOSS -> WITHIN.
Within 0.00015 of the floor (the call stands): none.
