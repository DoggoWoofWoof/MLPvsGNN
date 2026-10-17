A4 (docs/A4_BLOCK_DROPOUT_LEAN.md): zrckpd (zrckp trained with A4's block dropout (0.2 per question and block)) against zrckp.

# scr-a4-zrckpd-pair re-called under the seed null: **NO_GAIN** (filed: NO_GAIN)

R@5 of zrckpd's p@swa minus zrckp's p@swa of the same split (zrckp's card fits), on the six s1eval carves (95% question bootstrap), called with each read's floor: max(0.0075, twice the seed-only spread of step 1's seeds 1 and 2 against seed 0) (docs/SCREENS.md, section 2, and docs/A4_BLOCK_DROPOUT_LEAN.md). D1, D2: the null's seed-only differences on the read. Bold: a changed call.

| split | dataset | read | zrckp R@5 | zrckpd R@5 (A4's block dropout) | delta R@5 [95% CI] | D1, D2 | floor | filed | re-call |
|---|---|---|---:|---:|---|---|---:|---|---|
| L-musique | metaqa | in-domain | 0.8019 | 0.7993 | -0.0026 [-0.0038, -0.0015] | +0.0008, -0.0023 | 0.0075 | WITHIN | WITHIN |
| L-musique | squad | in-domain | 0.9116 | 0.9103 | -0.0013 [-0.0035, +0.0007] | +0.0001, +0.0013 | 0.0075 | WITHIN | WITHIN |
| L-musique | musique | zero-shot | 0.5121 | 0.3927 | -0.1195 [-0.1280, -0.1100] | +0.0503, -0.0079 | 0.0720 | LOSS | LOSS |
| L-musique | hotpotqa | in-domain | 0.9065 | 0.9016 | -0.0050 [-0.0074, -0.0027] | +0.0014, +0.0016 | 0.0075 | WITHIN | WITHIN |
| L-musique | 2wiki | in-domain | 0.9347 | 0.9314 | -0.0033 [-0.0048, -0.0017] | -0.0002, -0.0007 | 0.0075 | WITHIN | WITHIN |
| L-musique | webqsp | zero-shot | 0.3627 | 0.3646 | +0.0019 [-0.0079, +0.0120] | -0.0070, -0.0018 | 0.0102 | WITHIN | WITHIN |
| L-hotpotqa | metaqa | in-domain | 0.8044 | 0.8020 | -0.0024 [-0.0034, -0.0014] | -0.0092, -0.0051 | 0.0149 | WITHIN | WITHIN |
| L-hotpotqa | squad | in-domain | 0.9127 | 0.9127 | -0.0001 [-0.0020, +0.0019] | +0.0002, +0.0007 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | musique | in-domain | 0.5745 | 0.5691 | -0.0053 [-0.0103, -0.0002] | +0.0015, +0.0023 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | hotpotqa | zero-shot | 0.8513 | 0.8529 | +0.0016 [-0.0016, +0.0045] | -0.0055, -0.0130 | 0.0200 | WITHIN | WITHIN |
| L-hotpotqa | 2wiki | in-domain | 0.9350 | 0.9340 | -0.0010 [-0.0026, +0.0007] | -0.0020, +0.0011 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | webqsp | zero-shot | 0.3475 | 0.3444 | -0.0030 [-0.0151, +0.0092] | +0.0102, +0.0129 | 0.0233 | WITHIN | WITHIN |

Re-call: reads with a GAIN: 0 of 12; with a LOSS: 1. PROMISING needs at least one GAIN and no LOSS.
Each fit, filed -> re-called: L-musique NO_GAIN -> NO_GAIN; L-hotpotqa NO_GAIN -> NO_GAIN.
Changed calls: none.
Within 0.00015 of the floor (the call stands): none.
