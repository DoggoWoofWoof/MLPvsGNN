# A5c's screen verdict: zrckpc **ADOPT_FOR_FULL_RUN**, zspkc **ADOPT_FOR_FULL_RUN**

Rule: per arm: ADOPT_FOR_FULL_RUN when its re-call under the seed null has no LOSS on any of its twelve reads and at least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when its re-call is missing, not A5c's re-call of the arm, or not twelve reads (docs/A5C_KIND_COUPLED.md, section 4).

- **zrckpc** (zrckp with zck's coupled kind offsets on typed graphs (zret's output layer and base weight, trained jointly)) against zrckp's card fit of each split (scr-p1c3p-zrckp, scr-p1c3p-zrckp-hp), after the seed-null re-call: 2 GAIN (L-musique webqsp, L-hotpotqa webqsp), 0 LOSS (none). Beside: L-musique metaqa WITHIN (-0.0025), L-musique 2wiki WITHIN (+0.0001), L-musique webqsp GAIN (+0.0189), L-hotpotqa metaqa WITHIN (-0.0005), L-hotpotqa 2wiki WITHIN (-0.0018), L-hotpotqa webqsp GAIN (+0.0239). Next: the arm fitted again on P1g-b's pools, then its full run, its own declared file.
- **zspkc** (zspk with zck's coupled kind offsets on typed graphs (zret's output layer and base weight, trained jointly), before its propagation) against zspk's card fit of each split (scr-p1c3-zspk, scr-p1c3-zspk-hp), after the seed-null re-call: 2 GAIN (L-hotpotqa musique, L-hotpotqa webqsp), 0 LOSS (none). Beside: L-musique metaqa WITHIN (+0.0021), L-musique 2wiki WITHIN (-0.0014), L-musique webqsp WITHIN (-0.0052), L-hotpotqa metaqa WITHIN (+0.0006), L-hotpotqa 2wiki WITHIN (+0.0041), L-hotpotqa webqsp GAIN (+0.0520). Next: the arm fitted again on P1g-b's pools, then its full run, its own declared file.

Beside, deciding nothing: musique in-domain (L-hotpotqa musique): zrckpc WITHIN (-0.0018), zspkc GAIN (+0.0083).

This verdict closes only this form (zck's offsets on zret's output layer and base weight on typed graphs, trained jointly with the shared weights); docs/SCREENS.md, 3b.
