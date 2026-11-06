# FSV A1: the fast versions of zg6r, against ML1's zg6r, on the host's GPU

Declared in `docs/FSV_FAST_VERSIONS.md` (amendment A1). Device NVIDIA RTX 4500 Ada Generation, torch 2.8.0+cu128.

**Fast version:** L2. **Next:** nothing.

## L3: without topo_STRUCT, depth_STRUCT, SEED, DISTS, WALK, WALKF

### Screen: FAIL

counts: {"GAIN": 2, "LOSS": 3, "WITHIN": 3}

| split | read | full s0, s1, s2 | cut s0 | delta | floor | call |
|---|---|---|---:|---:|---:|---|
| WQ-musique | hotpotqa | 0.8593, 0.8596, 0.8562 | 0.8563 | -0.0030 | 0.0075 | WITHIN |
| WQ-musique | 2wiki | 0.9269, 0.9297, 0.9215 | 0.9359 | +0.0090 | 0.0086 | GAIN |
| WQ-musique | musique | 0.5086, 0.5039, 0.5134 | 0.5316 | +0.0230 | 0.0095 | GAIN |
| WQ-hotpotqa | musique | 0.5912, 0.5988, 0.5939 | 0.5692 | -0.0220 | 0.0114 | LOSS |
| WQ-hotpotqa | 2wiki | 0.9296, 0.9290, 0.9276 | 0.9324 | +0.0028 | 0.0075 | WITHIN |
| WQ-hotpotqa | hotpotqa | 0.8004, 0.7943, 0.7840 | 0.7951 | -0.0053 | 0.0247 | WITHIN |
| WQ-hotpotqa | musique-3hop | 0.5333, 0.5443, 0.5430 | 0.5075 | -0.0259 | 0.0207 | LOSS |
| WQ-hotpotqa | musique-4hop | 0.3765, 0.3833, 0.3772 | 0.3654 | -0.0111 | 0.0096 | LOSS |
| WQ-musique | metaqa | 0.3615, 0.3477, 0.3461 | 0.3821 | +0.0206 | 0.0292 | WITHIN (beside) |
| WQ-musique | squad | 0.9142, 0.9145, 0.9163 | 0.9117 | -0.0025 | 0.0075 | WITHIN (beside) |
| WQ-musique | webqsp | 0.6632, 0.6586, 0.6593 | 0.6865 | +0.0233 | 0.0085 | GAIN (beside) |
| WQ-hotpotqa | metaqa | 0.3519, 0.3903, 0.3651 | 0.4237 | +0.0718 | 0.0574 | GAIN (beside) |
| WQ-hotpotqa | squad | 0.9139, 0.9140, 0.9127 | 0.9074 | -0.0065 | 0.0075 | WITHIN (beside) |
| WQ-hotpotqa | webqsp | 0.6658, 0.6512, 0.6523 | 0.6716 | +0.0058 | 0.0281 | WITHIN (beside) |
| WQ-musique | musique-3hop | 0.4329, 0.4272, 0.4404 | 0.4469 | +0.0140 | 0.0133 | GAIN (beside) |
| WQ-musique | musique-4hop | 0.3130, 0.3167, 0.3179 | 0.3377 | +0.0247 | 0.0087 | GAIN (beside) |


## L2: without topo_STRUCT, depth_STRUCT, WALK, WALKF

### Screen: PASS

counts: {"GAIN": 1, "LOSS": 0, "WITHIN": 7}

| split | read | full s0, s1, s2 | cut s0 | delta | floor | call |
|---|---|---|---:|---:|---:|---|
| WQ-musique | hotpotqa | 0.8593, 0.8596, 0.8562 | 0.8579 | -0.0014 | 0.0075 | WITHIN |
| WQ-musique | 2wiki | 0.9269, 0.9297, 0.9215 | 0.9311 | +0.0042 | 0.0086 | WITHIN |
| WQ-musique | musique | 0.5086, 0.5039, 0.5134 | 0.5248 | +0.0162 | 0.0095 | GAIN |
| WQ-hotpotqa | musique | 0.5912, 0.5988, 0.5939 | 0.5893 | -0.0019 | 0.0114 | WITHIN |
| WQ-hotpotqa | 2wiki | 0.9296, 0.9290, 0.9276 | 0.9324 | +0.0028 | 0.0075 | WITHIN |
| WQ-hotpotqa | hotpotqa | 0.8004, 0.7943, 0.7840 | 0.8022 | +0.0018 | 0.0247 | WITHIN |
| WQ-hotpotqa | musique-3hop | 0.5333, 0.5443, 0.5430 | 0.5386 | +0.0053 | 0.0207 | WITHIN |
| WQ-hotpotqa | musique-4hop | 0.3765, 0.3833, 0.3772 | 0.3704 | -0.0062 | 0.0096 | WITHIN |
| WQ-musique | metaqa | 0.3615, 0.3477, 0.3461 | 0.3911 | +0.0296 | 0.0292 | GAIN (beside) |
| WQ-musique | squad | 0.9142, 0.9145, 0.9163 | 0.9130 | -0.0012 | 0.0075 | WITHIN (beside) |
| WQ-musique | webqsp | 0.6632, 0.6586, 0.6593 | 0.6708 | +0.0076 | 0.0085 | WITHIN (beside) |
| WQ-hotpotqa | metaqa | 0.3519, 0.3903, 0.3651 | 0.3743 | +0.0224 | 0.0574 | WITHIN (beside) |
| WQ-hotpotqa | squad | 0.9139, 0.9140, 0.9127 | 0.9126 | -0.0013 | 0.0075 | WITHIN (beside) |
| WQ-hotpotqa | webqsp | 0.6658, 0.6512, 0.6523 | 0.6840 | +0.0182 | 0.0281 | WITHIN (beside) |
| WQ-musique | musique-3hop | 0.4329, 0.4272, 0.4404 | 0.4557 | +0.0228 | 0.0133 | GAIN (beside) |
| WQ-musique | musique-4hop | 0.3130, 0.3167, 0.3179 | 0.3210 | +0.0080 | 0.0087 | WITHIN (beside) |


### Full run: ADMITTED

counts: {"GAIN": 1, "LOSS": 0, "UNCONFIRMED_GAIN": 0, "WITHIN": 7}

| split | read | d0 | d1 | d2 | floor | call |
|---|---|---:|---:|---:|---:|---|
| WQ-musique | hotpotqa | -0.0014 | +0.0003 | +0.0044 | 0.0075 | WITHIN |
| WQ-musique | 2wiki | +0.0042 | +0.0031 | +0.0060 | 0.0075 | WITHIN |
| WQ-musique | musique | +0.0162 | +0.0292 | +0.0228 | 0.0156 | GAIN |
| WQ-hotpotqa | musique | -0.0019 | -0.0096 | -0.0055 | 0.0081 | WITHIN |
| WQ-hotpotqa | 2wiki | +0.0028 | +0.0048 | +0.0059 | 0.0075 | WITHIN |
| WQ-hotpotqa | hotpotqa | +0.0018 | +0.0072 | +0.0141 | 0.0180 | WITHIN |
| WQ-hotpotqa | musique-3hop | +0.0053 | -0.0013 | +0.0018 | 0.0164 | WITHIN |
| WQ-hotpotqa | musique-4hop | -0.0062 | -0.0056 | -0.0148 | 0.0129 | WITHIN |
| WQ-musique | metaqa | +0.0296 | +0.0408 | +0.0704 | 0.0329 | WITHIN (beside) |
| WQ-musique | squad | -0.0012 | -0.0034 | -0.0034 | 0.0075 | WITHIN (beside) |
| WQ-musique | webqsp | +0.0076 | +0.0089 | +0.0103 | 0.0075 | GAIN (beside) |
| WQ-hotpotqa | metaqa | +0.0224 | +0.0262 | +0.0268 | 0.0612 | WITHIN (beside) |
| WQ-hotpotqa | squad | -0.0013 | -0.0034 | -0.0012 | 0.0075 | WITHIN (beside) |
| WQ-hotpotqa | webqsp | +0.0182 | +0.0154 | +0.0118 | 0.0331 | WITHIN (beside) |
| WQ-musique | musique-3hop | +0.0228 | +0.0289 | +0.0202 | 0.0106 | GAIN (beside) |
| WQ-musique | musique-4hop | +0.0080 | +0.0111 | +0.0241 | 0.0229 | WITHIN (beside) |


