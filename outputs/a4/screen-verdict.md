# A4's screen verdict: zrckpd **NOT_ADOPTED**, zspkd **NOT_ADOPTED**

Rule: per arm: ADOPT_FOR_FULL_RUN when its re-call under the seed null has no LOSS on any of its twelve reads and at least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when its re-call is missing, not A4's re-call of the arm, or not twelve reads (docs/A4_BLOCK_DROPOUT_LEAN.md, section 4).

- **zrckpd** (zrckp trained with A4's block dropout (0.2 per question and block)) against zrckp's card fit of each split (scr-p1c3p-zrckp, scr-p1c3p-zrckp-hp), after the seed-null re-call: 0 GAIN (none), 1 LOSS (L-musique musique). Beside: L-musique metaqa WITHIN (-0.0026), L-musique musique LOSS (-0.1195), L-musique webqsp WITHIN (+0.0019), L-hotpotqa metaqa WITHIN (-0.0024), L-hotpotqa musique WITHIN (-0.0053), L-hotpotqa webqsp WITHIN (-0.0030). Next: A4f, the family form: the inputs above zret dropped as blocks too (the KT columns and rmatch's chain term, each one more (question, block) cell at 0.2), on both models, its own declared file; the propagation's edge families are A9's.
- **zspkd** (zspk trained with A4's block dropout (0.2 per question and block), before its propagation) against zspk's card fit of each split (scr-p1c3-zspk, scr-p1c3-zspk-hp), after the seed-null re-call: 1 GAIN (L-musique webqsp), 1 LOSS (L-musique musique). Beside: L-musique metaqa WITHIN (+0.0006), L-musique musique LOSS (-0.1394), L-musique webqsp GAIN (+0.0134), L-hotpotqa metaqa WITHIN (-0.0018), L-hotpotqa musique WITHIN (-0.0032), L-hotpotqa webqsp WITHIN (+0.0192). Next: the back-off form, A4h: block dropout at half the rate (0.1 per question and block), on both models, its own declared file.

Beside, deciding nothing: musique in-domain (L-hotpotqa musique): zrckpd WITHIN (-0.0053), zspkd WITHIN (-0.0032).

This verdict closes only this form (block dropout 0.2 per question and block over the models' blocks, at the top of the forward, in training); docs/SCREENS.md, 3b.
