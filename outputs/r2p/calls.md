# R2P: zrckpb (zrckp + R2's bridge head (an MLP arm under the user's ruling of 2 November 2026)) against DOB's CPU zrckp, on PKO's pools, on the host's CPU

## Screen: NOT_ADOPTED

counts: {"GAIN": 2, "LOSS": 2, "WITHIN": 8}

| split | dataset | base s0, s1, s2 | zrckpb s0 | delta | floor | call |
|---|---|---|---:|---:|---:|---|
| WQ-musique | metaqa | 0.3406, 0.3670, 0.3860 | 0.4138 | +0.0732 | 0.0743 | WITHIN |
| WQ-musique | squad | 0.9127, 0.9122, 0.9138 | 0.9107 | -0.0020 | 0.0075 | WITHIN |
| WQ-musique | musique | 0.5152, 0.5086, 0.5134 | 0.5204 | +0.0052 | 0.0097 | WITHIN |
| WQ-musique | hotpotqa (C1R) | 0.8411, 0.8424, 0.8388 | 0.8389 | -0.0022 | 0.0075 | WITHIN |
| WQ-musique | 2wiki | 0.9070, 0.9079, 0.9049 | 0.9067 | -0.0003 | 0.0075 | WITHIN |
| WQ-musique | webqsp | 0.6522, 0.6583, 0.6566 | 0.6725 | +0.0203 | 0.0106 | GAIN |
| WQ-hotpotqa | metaqa | 0.3847, 0.4037, 0.3797 | 0.4038 | +0.0191 | 0.0278 | WITHIN |
| WQ-hotpotqa | squad | 0.9138, 0.9129, 0.9133 | 0.9106 | -0.0032 | 0.0075 | WITHIN |
| WQ-hotpotqa | musique (C1R) | 0.5749, 0.5718, 0.5752 | 0.5667 | -0.0082 | 0.0075 | LOSS |
| WQ-hotpotqa | hotpotqa | 0.8103, 0.8099, 0.8101 | 0.8020 | -0.0083 | 0.0075 | LOSS |
| WQ-hotpotqa | 2wiki | 0.8987, 0.9082, 0.9063 | 0.9054 | +0.0067 | 0.0172 | WITHIN |
| WQ-hotpotqa | webqsp | 0.6446, 0.6524, 0.6514 | 0.6610 | +0.0164 | 0.0146 | GAIN |

Beside, deciding nothing: R2's term and the seconds per fit:

| fit | share of training questions with the term | L_br first epoch | L_br last epoch | train s | read s |
|---|---:|---:|---:|---:|---:|
| scr-r2p-zrckpb-s0 | 0.9933 | 5.6751 | 4.8496 | 3209.8 | 4097.3 |
| scr-r2p-zrckpb-s0-hp | 0.9934 | 6.1496 | 5.0666 | 5818.9 | 3556.0 |
