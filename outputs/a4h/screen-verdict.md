# A4h's screen verdict: zrckph **NOT_ADOPTED**, zspkh **NOT_ADOPTED**

Rule: per arm: ADOPT_FOR_FULL_RUN when its re-call under the seed null has no LOSS on any of its twelve reads and at least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when its re-call is missing, not A4h's re-call of the arm, or not twelve reads (docs/A4H_BLOCK_DROPOUT_HALF.md, section 4).

- **zrckph** (zrckp trained with A4h's block dropout (0.1 per question and block)) against zrckp's card fit of each split (scr-p1c3p-zrckp, scr-p1c3p-zrckp-hp), after the seed-null re-call: 0 GAIN (none), 0 LOSS (none). Beside: L-musique metaqa WITHIN (+0.0006), L-musique musique WITHIN (-0.0530), L-musique webqsp WITHIN (-0.0004), L-hotpotqa metaqa WITHIN (-0.0036), L-hotpotqa musique WITHIN (-0.0064), L-hotpotqa webqsp WITHIN (+0.0127). Next: the rate closes; the idea goes on in A4f (the KT columns and rmatch's chain term dropped as blocks too), declared beside, on both models.
- **zspkh** (zspk trained with A4h's block dropout (0.1 per question and block), before its propagation) against zspk's card fit of each split (scr-p1c3-zspk, scr-p1c3-zspk-hp), after the seed-null re-call: 1 GAIN (L-hotpotqa webqsp), 1 LOSS (L-musique musique). Beside: L-musique metaqa WITHIN (+0.0023), L-musique musique LOSS (-0.1143), L-musique webqsp WITHIN (-0.0030), L-hotpotqa metaqa WITHIN (-0.0017), L-hotpotqa musique WITHIN (-0.0017), L-hotpotqa webqsp GAIN (+0.0248). Next: A4hs, the back-off over blocks: block dropout at 0.1 on the eight fixed blocks only, SEMB and with it rmatch's chain term always kept, on both models, its own declared file.

Beside, deciding nothing: musique in-domain (L-hotpotqa musique): zrckph WITHIN (-0.0064), zspkh WITHIN (-0.0017).

This verdict closes only this form (block dropout 0.1 per question and block over the models' blocks, at the top of the forward, in training); docs/SCREENS.md, 3b.
