A3vp (docs/A3VP_P1GB_POOLS.md): zrckpy (zrckp trained with A3v's two-sided share augmentation of zret's reference) against zrckp, both on P1g-b's pools.

# scr-a3vp-zrckpy-pair re-called under the seed null: **NO_GAIN** (filed: NO_GAIN)

R@5 of zrckpy's p@swa minus zrckp's p@swa of the same split (zrckp's card fits on P1g-b's pools), on the six s1eval carves (95% question bootstrap), called with each read's floor: max(0.0075, twice the seed-only spread of step 1's seeds 1 and 2 against seed 0) (docs/SCREENS.md, section 2, and docs/A3VP_P1GB_POOLS.md). D1, D2: the null's seed-only differences on the read. Bold: a changed call.

| split | dataset | read | zrckp R@5 (P1g-b's) | zrckpy R@5 (A3v's rule, P1g-b's) | delta R@5 [95% CI] | D1, D2 | floor | filed | re-call |
|---|---|---|---:|---:|---|---|---:|---|---|
| L-musique | metaqa | in-domain | 0.7993 | 0.8004 | +0.0012 [+0.0001, +0.0022] | +0.0008, -0.0023 | 0.0075 | WITHIN | WITHIN |
| L-musique | squad | in-domain | 0.9127 | 0.9116 | -0.0010 [-0.0028, +0.0008] | +0.0001, +0.0013 | 0.0075 | WITHIN | WITHIN |
| L-musique | musique | zero-shot | 0.5013 | 0.5067 | +0.0054 [+0.0010, +0.0098] | +0.0503, -0.0079 | 0.0720 | WITHIN | WITHIN |
| L-musique | hotpotqa | in-domain | 0.9035 | 0.9042 | +0.0007 [-0.0016, +0.0030] | +0.0014, +0.0016 | 0.0075 | WITHIN | WITHIN |
| L-musique | 2wiki | in-domain | 0.9377 | 0.9308 | -0.0069 [-0.0084, -0.0054] | -0.0002, -0.0007 | 0.0075 | WITHIN | WITHIN |
| L-musique | webqsp | zero-shot | 0.3768 | 0.3655 | -0.0113 [-0.0206, -0.0014] | -0.0070, -0.0018 | 0.0102 | LOSS | LOSS |
| L-hotpotqa | metaqa | in-domain | 0.7997 | 0.8026 | +0.0030 [+0.0021, +0.0039] | -0.0092, -0.0051 | 0.0149 | WITHIN | WITHIN |
| L-hotpotqa | squad | in-domain | 0.9122 | 0.9125 | +0.0003 [-0.0016, +0.0022] | +0.0002, +0.0007 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | musique | in-domain | 0.5695 | 0.5637 | -0.0058 [-0.0114, -0.0003] | +0.0015, +0.0023 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | hotpotqa | zero-shot | 0.8546 | 0.8509 | -0.0037 [-0.0067, -0.0007] | -0.0055, -0.0130 | 0.0200 | WITHIN | WITHIN |
| L-hotpotqa | 2wiki | in-domain | 0.9340 | 0.9286 | -0.0054 [-0.0071, -0.0036] | -0.0020, +0.0011 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | webqsp | zero-shot | 0.3364 | 0.3183 | -0.0181 [-0.0295, -0.0069] | +0.0102, +0.0129 | 0.0233 | LOSS | **WITHIN** |

Re-call: reads with a GAIN: 0 of 12; with a LOSS: 1. PROMISING needs at least one GAIN and no LOSS.
Each fit, filed -> re-called: L-musique NO_GAIN -> NO_GAIN; L-hotpotqa NO_GAIN -> NO_GAIN.
Changed calls: L-hotpotqa webqsp LOSS -> WITHIN.
Within 0.00015 of the floor (the call stands): none.
