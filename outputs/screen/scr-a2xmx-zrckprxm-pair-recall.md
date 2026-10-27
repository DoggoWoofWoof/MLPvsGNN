A2xmx (docs/A2XMX_ZTOP50_ADOPTED.md): zrckprxm (zrckpr with zret's reference cut to the pool's top 50 retrieved rows by rrf (ztop50's form)) against zrckpr.

# scr-a2xmx-zrckprxm-pair re-called under the seed null: **NO_GAIN** (filed: NO_GAIN)

R@5 of zrckprxm's p@swa minus zrckpr's p@swa of the same split (zrckpr's card fits), on the six s1eval carves (95% question bootstrap), called with each read's floor: max(0.0075, twice the seed-only spread of step 1's seeds 1 and 2 against seed 0) (docs/SCREENS.md, section 2, and docs/A2XMX_ZTOP50_ADOPTED.md). D1, D2: the null's seed-only differences on the read. Bold: a changed call.

| split | dataset | read | zrckpr R@5 | zrckprxm R@5 (A2xmx's top-50 reference) | delta R@5 [95% CI] | D1, D2 | floor | filed | re-call |
|---|---|---|---:|---:|---|---|---:|---|---|
| L-musique | metaqa | in-domain | 0.8012 | 0.8004 | -0.0008 [-0.0018, +0.0002] | +0.0008, -0.0023 | 0.0075 | WITHIN | WITHIN |
| L-musique | squad | in-domain | 0.9127 | 0.9111 | -0.0016 [-0.0035, +0.0003] | +0.0001, +0.0013 | 0.0075 | WITHIN | WITHIN |
| L-musique | musique | zero-shot | 0.5197 | 0.5165 | -0.0032 [-0.0090, +0.0027] | +0.0503, -0.0079 | 0.0720 | WITHIN | WITHIN |
| L-musique | hotpotqa | in-domain | 0.9083 | 0.9060 | -0.0023 [-0.0045, -0.0002] | +0.0014, +0.0016 | 0.0075 | WITHIN | WITHIN |
| L-musique | 2wiki | in-domain | 0.9361 | 0.9356 | -0.0005 [-0.0019, +0.0009] | -0.0002, -0.0007 | 0.0075 | WITHIN | WITHIN |
| L-musique | webqsp | zero-shot | 0.3762 | 0.3856 | +0.0094 [-0.0019, +0.0208] | -0.0070, -0.0018 | 0.0102 | WITHIN | WITHIN |
| L-hotpotqa | metaqa | in-domain | 0.8040 | 0.7977 | -0.0064 [-0.0076, -0.0052] | -0.0092, -0.0051 | 0.0149 | WITHIN | WITHIN |
| L-hotpotqa | squad | in-domain | 0.9114 | 0.9114 | +0.0000 [-0.0019, +0.0019] | +0.0002, +0.0007 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | musique | in-domain | 0.5779 | 0.5723 | -0.0057 [-0.0120, +0.0000] | +0.0015, +0.0023 | 0.0075 | WITHIN | WITHIN |
| L-hotpotqa | hotpotqa | zero-shot | 0.8558 | 0.8540 | -0.0018 [-0.0051, +0.0015] | -0.0055, -0.0130 | 0.0200 | WITHIN | WITHIN |
| L-hotpotqa | 2wiki | in-domain | 0.9331 | 0.9243 | -0.0088 [-0.0108, -0.0070] | -0.0020, +0.0011 | 0.0075 | LOSS | LOSS |
| L-hotpotqa | webqsp | zero-shot | 0.3411 | 0.3382 | -0.0029 [-0.0163, +0.0103] | +0.0102, +0.0129 | 0.0233 | WITHIN | WITHIN |

Re-call: reads with a GAIN: 0 of 12; with a LOSS: 1. PROMISING needs at least one GAIN and no LOSS.
Each fit, filed -> re-called: L-musique NO_GAIN -> NO_GAIN; L-hotpotqa NO_GAIN -> NO_GAIN.
Changed calls: none.
Within 0.00015 of the floor (the call stands): none.
