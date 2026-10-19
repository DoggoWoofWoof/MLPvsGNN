# A4f's screen verdict: zrckpf **NOT_ADOPTED**, zspkf **NOT_ADOPTED**

Rule: per arm: ADOPT_FOR_FULL_RUN when its re-call under the seed null has no LOSS on any of its twelve reads and at least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when its re-call is missing, not A4f's re-call of the arm, or not twelve reads (docs/A4F_FAMILY_DROPOUT.md, section 4).

- **zrckpf** (zrckp trained with A4f's family dropout (0.2 per question on each block, the KT term and the chain term)) against zrckp's card fit of each split (scr-p1c3p-zrckp, scr-p1c3p-zrckp-hp), after the seed-null re-call: 0 GAIN (none), 2 LOSS (L-musique musique, L-hotpotqa webqsp). Beside: L-musique metaqa WITHIN (-0.0038), L-musique musique LOSS (-0.1147), L-musique webqsp WITHIN (+0.0026), L-hotpotqa metaqa WITHIN (-0.0057), L-hotpotqa musique WITHIN (-0.0035), L-hotpotqa webqsp LOSS (-0.0236). Next: A4fx, the family form on the adopted arms (G5's zspx and R1's zrckpr, R1's columns one more cell), its own declared file.
- **zspkf** (zspk trained with A4f's family dropout (0.2 per question on each block, the KT term and the chain term), before its propagation) against zspk's card fit of each split (scr-p1c3-zspk, scr-p1c3-zspk-hp), after the seed-null re-call: 0 GAIN (none), 2 LOSS (L-musique musique, L-musique webqsp). Beside: L-musique metaqa WITHIN (-0.0039), L-musique musique LOSS (-0.2264), L-musique webqsp LOSS (-0.0125), L-hotpotqa metaqa WITHIN (-0.0017), L-hotpotqa musique WITHIN (+0.0021), L-hotpotqa webqsp WITHIN (-0.0128). Next: A4fx, the family form on the adopted arms (G5's zspx and R1's zrckpr, R1's columns one more cell), its own declared file.

Beside, deciding nothing: musique in-domain (L-hotpotqa musique): zrckpf WITHIN (-0.0035), zspkf WITHIN (+0.0021).

This verdict closes only this form (family dropout 0.2 per question on each block, the KT term and the chain term, at the top of the forward, in training); docs/SCREENS.md, 3b.
