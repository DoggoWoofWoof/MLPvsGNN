# A2's screen verdict: zrckpa **NOT_ADOPTED**, zspka **NOT_ADOPTED**

Rule: per arm: ADOPT_FOR_FULL_RUN when its re-call under the seed null has no LOSS on any of its twelve reads and at least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when its re-call is missing, not A2's re-call of the arm, or not twelve reads (docs/A2_RANKS_BESIDE_Z.md, section 4).

- **zrckpa** (zrckp with A2's ranks and top-50 columns beside its z-scores) against zrckp's card fit of each split (scr-p1c3p-zrckp, scr-p1c3p-zrckp-hp), after the seed-null re-call: 0 GAIN (none), 1 LOSS (L-musique hotpotqa). Beside: L-musique metaqa WITHIN (-0.0039), L-musique musique WITHIN (+0.0104), L-musique webqsp WITHIN (-0.0097), L-hotpotqa metaqa WITHIN (-0.0042), L-hotpotqa musique WITHIN (-0.0050), L-hotpotqa webqsp WITHIN (-0.0129). Next: the halves apart: the rank columns alone beside, and the top-50 columns alone beside, each on both models, their own declared file.
- **zspka** (zspk with A2's ranks and top-50 columns beside its z-scores, before its propagation) against zspk's card fit of each split (scr-p1c3-zspk, scr-p1c3-zspk-hp), after the seed-null re-call: 0 GAIN (none), 2 LOSS (L-musique 2wiki, L-musique webqsp). Beside: L-musique metaqa WITHIN (-0.0002), L-musique musique WITHIN (+0.0249), L-musique webqsp LOSS (-0.0112), L-hotpotqa metaqa WITHIN (-0.0046), L-hotpotqa musique WITHIN (+0.0016), L-hotpotqa webqsp WITHIN (+0.0012). Next: the halves apart: the rank columns alone beside, and the top-50 columns alone beside, each on both models, their own declared file.

Beside, deciding nothing: musique in-domain (L-hotpotqa musique): zrckpa WITHIN (-0.0050), zspka WITHIN (+0.0016).

This verdict closes only this form (the rank and top-50 columns beside the z-scores, into l1); docs/SCREENS.md, 3b.
