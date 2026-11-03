# DEP: zg6r with 4 message-passing layers against zg6r (ML1), on PKO's pools

## Screen: ADOPT_FOR_FULL_RUN

counts: {"GAIN": 1, "LOSS": 0, "WITHIN": 15}

| split | read | base s0, s1, s2 | DEP s0 | delta | floor | call |
|---|---|---|---:|---:|---:|---|
| WQ-musique | metaqa | 0.3615, 0.3477, 0.3461 | 0.3583 | -0.0032 | 0.0292 | WITHIN |
| WQ-musique | squad | 0.9142, 0.9145, 0.9163 | 0.9136 | -0.0006 | 0.0075 | WITHIN |
| WQ-musique | musique | 0.5086, 0.5039, 0.5134 | 0.5146 | +0.0060 | 0.0095 | WITHIN |
| WQ-musique | hotpotqa | 0.8593, 0.8596, 0.8562 | 0.8571 | -0.0022 | 0.0075 | WITHIN |
| WQ-musique | 2wiki | 0.9269, 0.9297, 0.9215 | 0.9279 | +0.0010 | 0.0086 | WITHIN |
| WQ-musique | webqsp | 0.6632, 0.6586, 0.6593 | 0.6551 | -0.0081 | 0.0085 | WITHIN |
| WQ-hotpotqa | metaqa | 0.3519, 0.3903, 0.3651 | 0.4113 | +0.0594 | 0.0574 | GAIN |
| WQ-hotpotqa | squad | 0.9139, 0.9140, 0.9127 | 0.9129 | -0.0010 | 0.0075 | WITHIN |
| WQ-hotpotqa | musique | 0.5912, 0.5988, 0.5939 | 0.5920 | +0.0008 | 0.0114 | WITHIN |
| WQ-hotpotqa | hotpotqa | 0.8004, 0.7943, 0.7840 | 0.7922 | -0.0082 | 0.0247 | WITHIN |
| WQ-hotpotqa | 2wiki | 0.9296, 0.9290, 0.9276 | 0.9255 | -0.0041 | 0.0075 | WITHIN |
| WQ-hotpotqa | webqsp | 0.6658, 0.6512, 0.6523 | 0.6563 | -0.0095 | 0.0281 | WITHIN |
| WQ-musique | musique-3hop | 0.4329, 0.4272, 0.4404 | 0.4399 | +0.0070 | 0.0133 | WITHIN |
| WQ-musique | musique-4hop | 0.3130, 0.3167, 0.3179 | 0.3198 | +0.0068 | 0.0087 | WITHIN |
| WQ-hotpotqa | musique-3hop | 0.5333, 0.5443, 0.5430 | 0.5355 | +0.0022 | 0.0207 | WITHIN |
| WQ-hotpotqa | musique-4hop | 0.3765, 0.3833, 0.3772 | 0.3809 | +0.0043 | 0.0096 | WITHIN |
