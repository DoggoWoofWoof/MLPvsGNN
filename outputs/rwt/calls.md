# RWT: question weights in zrckpn's loss, seed 0 on WQ-hotpotqa, against LTM's scr-ltm-zrckpn-s0-hp

## F, 1 - the question's softmax mass on its gold rows: NOT_ADOPTED

counts: {"GAIN": 0, "LOSS": 2, "WITHIN": 4}; mean delta over the six reads: -0.0046

| read | zrckpn s0, s1, s2 | F s0 | delta | floor | call |
|---|---|---:|---:|---:|---|
| squad | 0.9132, 0.9120, 0.9132 | 0.9143 | +0.0011 | 0.0075 | WITHIN |
| musique | 0.5806, 0.5803, 0.5777 | 0.5718 | -0.0088 | 0.0075 | LOSS |
| 2wiki | 0.9135, 0.9166, 0.9135 | 0.9128 | -0.0007 | 0.0075 | WITHIN |
| webqsp | 0.6469, 0.6597, 0.6512 | 0.6476 | +0.0007 | 0.0191 | WITHIN |
| musique-3hop | 0.5351, 0.5333, 0.5263 | 0.5171 | -0.0180 | 0.0127 | LOSS |
| musique-4hop | 0.3574, 0.3562, 0.3562 | 0.3556 | -0.0019 | 0.0075 | WITHIN |
| hotpotqa | 0.8193, 0.8182, 0.8136 | 0.8207 | +0.0014 | 0.0082 | WITHIN (beside) |
| metaqa | 0.3668, 0.4261, 0.4131 | 0.3859 | +0.0191 | 0.1064 | WITHIN (beside) |

## G, the question's gold rows capped at 5: NOT_ADOPTED

counts: {"GAIN": 0, "LOSS": 1, "WITHIN": 5}; mean delta over the six reads: -0.00147

| read | zrckpn s0, s1, s2 | G s0 | delta | floor | call |
|---|---|---:|---:|---:|---|
| squad | 0.9132, 0.9120, 0.9132 | 0.9143 | +0.0011 | 0.0075 | WITHIN |
| musique | 0.5806, 0.5803, 0.5777 | 0.5758 | -0.0048 | 0.0075 | WITHIN |
| 2wiki | 0.9135, 0.9166, 0.9135 | 0.9134 | -0.0001 | 0.0075 | WITHIN |
| webqsp | 0.6469, 0.6597, 0.6512 | 0.6558 | +0.0089 | 0.0191 | WITHIN |
| musique-3hop | 0.5351, 0.5333, 0.5263 | 0.5206 | -0.0145 | 0.0127 | LOSS |
| musique-4hop | 0.3574, 0.3562, 0.3562 | 0.3580 | +0.0006 | 0.0075 | WITHIN |
| hotpotqa | 0.8193, 0.8182, 0.8136 | 0.8169 | -0.0024 | 0.0082 | WITHIN (beside) |
| metaqa | 0.3668, 0.4261, 0.4131 | 0.4018 | +0.0350 | 0.1064 | WITHIN (beside) |

first full run (section 6): None

| fit | train s | read s | steps | weighted | F flat | w min | w max | curve |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| scr-rwt-zrckpnf-s0-hp | 4765.0 | 4258.4 | 17137 | 16095 | 0 | 0.0 | 5.231016635894775 | {"p": [2.9859, 2.5992, 2.5041, 2.4915, 2.4477, 2.3872, 2.3704, 2.3448]} |
| scr-rwt-zrckpng-s0-hp | 5347.2 | 3748.5 | 17137 | 10359 | 0 | 0.2380952537059784 | 2.777777910232544 | {"p": [2.7442, 2.3365, 2.2251, 2.2034, 2.1583, 2.0962, 2.0842, 2.0453]} |
