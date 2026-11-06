# IRA: zg6r with 2 rounds and 4 aligned slots against zg6r (ML1), on PKO's pools

## Screen: ADOPT_FOR_FULL_RUN

counts: {"GAIN": 3, "LOSS": 0, "WITHIN": 13}

| split | read | base s0, s1, s2 | IRA s0 | delta | floor | call |
|---|---|---|---:|---:|---:|---|
| WQ-musique | metaqa | 0.3615, 0.3477, 0.3461 | 0.3964 | +0.0349 | 0.0292 | GAIN |
| WQ-musique | squad | 0.9142, 0.9145, 0.9163 | 0.9132 | -0.0010 | 0.0075 | WITHIN |
| WQ-musique | musique | 0.5086, 0.5039, 0.5134 | 0.5129 | +0.0043 | 0.0095 | WITHIN |
| WQ-musique | hotpotqa | 0.8593, 0.8596, 0.8562 | 0.8594 | +0.0001 | 0.0075 | WITHIN |
| WQ-musique | 2wiki | 0.9269, 0.9297, 0.9215 | 0.9296 | +0.0027 | 0.0086 | WITHIN |
| WQ-musique | webqsp | 0.6632, 0.6586, 0.6593 | 0.6718 | +0.0086 | 0.0085 | GAIN |
| WQ-hotpotqa | metaqa | 0.3519, 0.3903, 0.3651 | 0.3612 | +0.0093 | 0.0574 | WITHIN |
| WQ-hotpotqa | squad | 0.9139, 0.9140, 0.9127 | 0.9116 | -0.0023 | 0.0075 | WITHIN |
| WQ-hotpotqa | musique | 0.5912, 0.5988, 0.5939 | 0.5945 | +0.0033 | 0.0114 | WITHIN |
| WQ-hotpotqa | hotpotqa | 0.8004, 0.7943, 0.7840 | 0.8065 | +0.0061 | 0.0247 | WITHIN |
| WQ-hotpotqa | 2wiki | 0.9296, 0.9290, 0.9276 | 0.9288 | -0.0008 | 0.0075 | WITHIN |
| WQ-hotpotqa | webqsp | 0.6658, 0.6512, 0.6523 | 0.6700 | +0.0042 | 0.0281 | WITHIN |
| WQ-musique | musique-3hop | 0.4329, 0.4272, 0.4404 | 0.4325 | -0.0004 | 0.0133 | WITHIN |
| WQ-musique | musique-4hop | 0.3130, 0.3167, 0.3179 | 0.3123 | -0.0006 | 0.0087 | WITHIN |
| WQ-hotpotqa | musique-3hop | 0.5333, 0.5443, 0.5430 | 0.5360 | +0.0026 | 0.0207 | WITHIN |
| WQ-hotpotqa | musique-4hop | 0.3765, 0.3833, 0.3772 | 0.3864 | +0.0099 | 0.0096 | GAIN |

## Full run: NOT_ADOPTED

| split | read | d0 | d1 | d2 | floor | call |
|---|---|---:|---:|---:|---:|---|
| WQ-musique | metaqa | +0.0349 | +0.0210 | +0.0226 | 0.0443 | WITHIN |
| WQ-musique | squad | -0.0010 | -0.0015 | -0.0022 | 0.0075 | WITHIN |
| WQ-musique | musique | +0.0043 | +0.0132 | +0.0135 | 0.0161 | WITHIN |
| WQ-musique | hotpotqa | +0.0001 | -0.0020 | +0.0011 | 0.0075 | WITHIN |
| WQ-musique | 2wiki | +0.0027 | -0.0045 | +0.0057 | 0.0079 | WITHIN |
| WQ-musique | webqsp | +0.0086 | +0.0011 | -0.0005 | 0.0188 | WITHIN |
| WQ-hotpotqa | metaqa | +0.0093 | -0.0135 | -0.0222 | 0.0472 | WITHIN |
| WQ-hotpotqa | squad | -0.0023 | -0.0006 | +0.0005 | 0.0075 | WITHIN |
| WQ-hotpotqa | musique | +0.0033 | -0.0027 | +0.0012 | 0.0082 | WITHIN |
| WQ-hotpotqa | hotpotqa | +0.0061 | +0.0069 | +0.0117 | 0.0212 | WITHIN |
| WQ-hotpotqa | 2wiki | -0.0008 | -0.0004 | +0.0029 | 0.0075 | WITHIN |
| WQ-hotpotqa | webqsp | +0.0042 | +0.0120 | +0.0096 | 0.0225 | WITHIN |
| WQ-musique | musique-3hop | -0.0004 | +0.0206 | +0.0101 | 0.0254 | WITHIN |
| WQ-musique | musique-4hop | -0.0006 | -0.0006 | +0.0185 | 0.0251 | WITHIN |
| WQ-hotpotqa | musique-3hop | +0.0026 | +0.0018 | -0.0026 | 0.0183 | WITHIN |
| WQ-hotpotqa | musique-4hop | +0.0099 | -0.0049 | +0.0025 | 0.0125 | WITHIN |

## Beside (deciding nothing)

RTR's screen on the same base: NO_GAIN {"GAIN": 0, "LOSS": 0, "WITHIN": 16}

| fit | lambda | ira_bz | ira_b0 | norm E | norm e | norm a |
|---|---:|---:|---:|---:|---:|---:|
| scr-ira-k4-s0 | 0.1142 | 1.7751 | -0.0022 | 4.9622 | 0.4593 | 6.5699 |
| scr-ira-k4-s0-hp | 0.0000 | 1.5334 | -0.0011 | 5.1017 | 0.5212 | 5.1592 |
| scr-ira-k4-s1 | 0.0000 | 1.8591 | 0.0015 | 4.9152 | 0.4580 | 5.6633 |
| scr-ira-k4-s1-hp | 0.0000 | 1.5064 | 0.0003 | 4.8543 | 0.6352 | 5.0952 |
| scr-ira-k4-s2 | 0.0000 | 1.3220 | 0.0009 | 4.0899 | 0.3193 | 4.5973 |
| scr-ira-k4-s2-hp | 0.0000 | 1.6935 | -0.0000 | 4.7458 | 0.5768 | 4.8013 |

next form by section 6's rule: three_rounds
