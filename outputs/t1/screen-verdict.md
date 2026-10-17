# T1's screen verdict: zrckpt **NOT_ADOPTED**, zspkt **NOT_ADOPTED**

Rule: per arm: ADOPT_FOR_FULL_RUN when its re-call under the seed null has no LOSS on any of its twelve reads and at least one GAIN; NOT_ADOPTED otherwise; INCOMPLETE when its re-call is missing, not T1's, or not twelve reads (docs/T1_GRANULARITY_TRANSFORMS.md, section 3).

- **zrckpt** against zrckp's card fit of each split (scr-p1c3p-zrckp), metaqa as filed, after the seed-null re-call: 0 GAIN (none), 2 LOSS (L-musique webqsp, L-hotpotqa webqsp). Beside: L-musique metaqa WITHIN (-0.0050), L-musique webqsp LOSS (-0.0153), L-hotpotqa metaqa WITHIN (-0.0065), L-hotpotqa webqsp LOSS (-0.0342).
- **zspkt** against zspk's card fit of each split (scr-p1c3-zspk), metaqa as filed, after the seed-null re-call: 0 GAIN (none), 0 LOSS (none). Beside: L-musique metaqa WITHIN (-0.0020), L-musique webqsp WITHIN (-0.0003), L-hotpotqa metaqa WITHIN (-0.0072), L-hotpotqa webqsp WITHIN (+0.0062).
