# P1c3's screen (docs/P1C3_KT_INPUT.md): **ADOPT_FOR_FULL_RUN**

Rule: ADOPT_FOR_FULL_RUN when neither arm's re-call has a LOSS on any of its twelve reads and the two arms together have at least one GAIN; NOT_ADOPTED otherwise (docs/P1C3_KT_INPUT.md, section 4).

R@5 of each arm (KT input, KIND_T's pools) minus its family's card fit of the split on today's pools, on the six s1eval carves; every read called under the seed null (docs/SCREENS.md). A changed call shows filed -> re-call.

| arm | split | dataset | read | base R@5 | new R@5 | delta | floor | call |
|---|---|---|---|---|---|---|---|---|
| zrck | L-musique | metaqa | in-domain | 0.7782 | 0.8007 | +0.0225 | 0.0075 | GAIN |
| zrck | L-musique | squad | in-domain | 0.9105 | 0.9117 | +0.0013 | 0.0075 | WITHIN |
| zrck | L-musique | musique | zero-shot | 0.5235 | 0.5059 | -0.0176 | 0.0720 | LOSS -> WITHIN |
| zrck | L-musique | hotpotqa | in-domain | 0.9041 | 0.9063 | +0.0022 | 0.0075 | WITHIN |
| zrck | L-musique | 2wiki | in-domain | 0.8733 | 0.9365 | +0.0632 | 0.0075 | GAIN |
| zrck | L-musique | webqsp | zero-shot | 0.3384 | 0.3756 | +0.0372 | 0.0102 | GAIN |
| zrck | L-hotpotqa | metaqa | in-domain | 0.7793 | 0.8052 | +0.0259 | 0.0149 | GAIN |
| zrck | L-hotpotqa | squad | in-domain | 0.9112 | 0.9122 | +0.0009 | 0.0075 | WITHIN |
| zrck | L-hotpotqa | musique | in-domain | 0.5645 | 0.5694 | +0.0050 | 0.0075 | WITHIN |
| zrck | L-hotpotqa | hotpotqa | zero-shot | 0.8451 | 0.8564 | +0.0113 | 0.0200 | GAIN -> WITHIN |
| zrck | L-hotpotqa | 2wiki | in-domain | 0.8731 | 0.9128 | +0.0397 | 0.0075 | GAIN |
| zrck | L-hotpotqa | webqsp | zero-shot | 0.2865 | 0.3686 | +0.0820 | 0.0233 | GAIN |
| zspk | L-musique | metaqa | in-domain | 0.7794 | 0.8009 | +0.0215 | 0.0075 | GAIN |
| zspk | L-musique | squad | in-domain | 0.9125 | 0.9114 | -0.0011 | 0.0075 | WITHIN |
| zspk | L-musique | musique | zero-shot | 0.5111 | 0.5012 | -0.0100 | 0.0720 | LOSS -> WITHIN |
| zspk | L-musique | hotpotqa | in-domain | 0.9065 | 0.9080 | +0.0015 | 0.0075 | WITHIN |
| zspk | L-musique | 2wiki | in-domain | 0.8724 | 0.9383 | +0.0659 | 0.0075 | GAIN |
| zspk | L-musique | webqsp | zero-shot | 0.3245 | 0.3229 | -0.0017 | 0.0102 | WITHIN |
| zspk | L-hotpotqa | metaqa | in-domain | 0.7807 | 0.8040 | +0.0233 | 0.0149 | GAIN |
| zspk | L-hotpotqa | squad | in-domain | 0.9109 | 0.9135 | +0.0026 | 0.0075 | WITHIN |
| zspk | L-hotpotqa | musique | in-domain | 0.5647 | 0.5754 | +0.0107 | 0.0075 | GAIN |
| zspk | L-hotpotqa | hotpotqa | zero-shot | 0.8435 | 0.8499 | +0.0064 | 0.0200 | WITHIN |
| zspk | L-hotpotqa | 2wiki | in-domain | 0.8744 | 0.9333 | +0.0589 | 0.0075 | GAIN |
| zspk | L-hotpotqa | webqsp | zero-shot | 0.2262 | 0.2680 | +0.0418 | 0.0233 | GAIN |

- zrck: 6 GAIN (L-musique metaqa, L-musique 2wiki, L-musique webqsp, L-hotpotqa metaqa, L-hotpotqa 2wiki, L-hotpotqa webqsp), 0 LOSS (none); its re-call PROMISING
- zspk: 6 GAIN (L-musique metaqa, L-musique 2wiki, L-hotpotqa metaqa, L-hotpotqa musique, L-hotpotqa 2wiki, L-hotpotqa webqsp), 0 LOSS (none); its re-call PROMISING

ADOPT_FOR_FULL_RUN starts the reads of the full run (docs/P1C3_FULL_RUN.md); NOT_ADOPTED leaves its fits unread.
