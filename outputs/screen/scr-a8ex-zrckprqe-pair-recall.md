A8ex (docs/A8EX_QDEPTH_PRIOR_ADOPTED.md): zrckprqe (zrckpr plus round 11's depth prior (A8e's), added to its final score) against zrckpr.

# scr-a8ex-zrckprqe-pair re-called under the seed null: **PROMISING** (filed: PROMISING)

R@5 of zrckprqe's p@swa minus zrckpr's p@swa of the same split (zrckpr's card fits), on the six s1eval carves (95% question bootstrap), called with each read's floor: max(0.0075, twice the seed-only spread of step 1's seeds 1 and 2 against seed 0) (docs/SCREENS.md, section 2, and docs/A8EX_QDEPTH_PRIOR_ADOPTED.md). D1, D2: the null's seed-only differences on the read. Bold: a changed call.

| split | dataset | read | zrckpr R@5 | zrckprqe R@5 (A8ex's depth prior) | delta R@5 [95% CI] | D1, D2 | floor | filed | re-call |
|---|---|---|---:|---:|---|---|---:|---|---|
| L-musique | metaqa | in-domain | 0.8012 | 0.7999 | -0.0014 [-0.0022, -0.0005] | +0.0008, -0.0023 | 0.0075 | WITHIN | WITHIN |
| L-musique | squad | in-domain | 0.9127 | 0.9111 | -0.0017 [-0.0034, +0.0000] | +0.0001, +0.0013 | 0.0075 | WITHIN | WITHIN |
| L-musique | musique | zero-shot | 0.5197 | 0.5195 | -0.0002 [-0.0047, +0.0046] | +0.0503, -0.0079 | 0.0720 | WITHIN | WITHIN |
| L-musique | hotpotqa | in-domain | 0.9083 | 0.9080 | -0.0003 [-0.0022, +0.0016] | +0.0014, +0.0016 | 0.0075 | WITHIN | WITHIN |
| L-musique | 2wiki | in-domain | 0.9361 | 0.9332 | -0.0029 [-0.0043, -0.0016] | -0.0002, -0.0007 | 0.0075 | WITHIN | WITHIN |
| L-musique | webqsp | zero-shot | 0.3762 | 0.3828 | +0.0066 [-0.0030, +0.0160] | -0.0070, -0.0018 | 0.0102 | WITHIN | WITHIN |
| L-hotpotqa | metaqa | in-domain | 0.8040 | 0.8039 | -0.0002 [-0.0010, +0.0006] | -0.0092, -0.0051 | 0.0149 | WITHIN | WITHIN |
| L-hotpotqa | squad | in-domain | 0.9114 | 0.9109 | -0.0005 [-0.0025, +0.0015] | +0.0002, +0.0007 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | musique | in-domain | 0.5779 | 0.5713 | -0.0067 [-0.0117, -0.0010] | +0.0015, +0.0023 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | hotpotqa | zero-shot | 0.8558 | 0.8550 | -0.0008 [-0.0039, +0.0020] | -0.0055, -0.0130 | 0.0200 | WITHIN | WITHIN |
| L-hotpotqa | 2wiki | in-domain | 0.9331 | 0.9305 | -0.0026 [-0.0043, -0.0010] | -0.0020, +0.0011 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | webqsp | zero-shot | 0.3411 | 0.3726 | +0.0315 [+0.0211, +0.0430] | +0.0102, +0.0129 | 0.0233 | GAIN | GAIN |

Re-call: reads with a GAIN: 1 of 12; with a LOSS: 0. PROMISING needs at least one GAIN and no LOSS.
Each fit, filed -> re-called: L-musique NO_GAIN -> NO_GAIN; L-hotpotqa PROMISING -> PROMISING.
Changed calls: none.
Within 0.00015 of the floor (the call stands): none.
