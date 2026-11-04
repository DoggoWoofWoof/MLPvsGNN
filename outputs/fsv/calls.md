# FSV: the fast versions of zrckpn, against LTM's zrckpn, on the host's CPU

Declared in `docs/FSV_FAST_VERSIONS.md`. Device cpu x6, torch 2.8.0+cpu.

**Fast version:** none yet. **Next:** nothing.

## L3: without topo_STRUCT, depth_STRUCT, SEED, DISTS, WALK, WALKF

### Screen: FAIL

counts: {"GAIN": 0, "LOSS": 6, "WITHIN": 2}

| split | read | full s0, s1, s2 | cut s0 | delta | floor | call |
|---|---|---|---:|---:|---:|---|
| WQ-musique | hotpotqa | 0.8460, 0.8469, 0.8491 | 0.8296 | -0.0164 | 0.0075 | LOSS |
| WQ-musique | 2wiki | 0.9155, 0.9159, 0.9133 | 0.9122 | -0.0033 | 0.0075 | WITHIN |
| WQ-musique | musique | 0.5190, 0.5227, 0.5381 | 0.5430 | +0.0240 | 0.0275 | WITHIN |
| WQ-hotpotqa | musique | 0.5806, 0.5803, 0.5777 | 0.5629 | -0.0177 | 0.0075 | LOSS |
| WQ-hotpotqa | 2wiki | 0.9135, 0.9166, 0.9135 | 0.9047 | -0.0088 | 0.0075 | LOSS |
| WQ-hotpotqa | hotpotqa | 0.8193, 0.8182, 0.8136 | 0.7991 | -0.0202 | 0.0082 | LOSS |
| WQ-hotpotqa | musique-3hop | 0.5351, 0.5333, 0.5263 | 0.5158 | -0.0193 | 0.0127 | LOSS |
| WQ-hotpotqa | musique-4hop | 0.3574, 0.3562, 0.3562 | 0.3370 | -0.0204 | 0.0075 | LOSS |
| WQ-musique | metaqa | 0.4126, 0.4256, 0.4167 | 0.4605 | +0.0479 | 0.0193 | GAIN (beside) |
| WQ-musique | squad | 0.9138, 0.9125, 0.9149 | 0.9084 | -0.0054 | 0.0075 | WITHIN (beside) |
| WQ-musique | webqsp | 0.6457, 0.6498, 0.6552 | 0.6485 | +0.0028 | 0.0146 | WITHIN (beside) |
| WQ-hotpotqa | metaqa | 0.3668, 0.4261, 0.4131 | 0.4120 | +0.0452 | 0.1064 | WITHIN (beside) |
| WQ-hotpotqa | squad | 0.9132, 0.9120, 0.9132 | 0.9105 | -0.0027 | 0.0075 | WITHIN (beside) |
| WQ-hotpotqa | webqsp | 0.6469, 0.6597, 0.6512 | 0.6541 | +0.0072 | 0.0191 | WITHIN (beside) |
| WQ-musique | musique-3hop | 0.4544, 0.4596, 0.4702 | 0.4833 | +0.0289 | 0.0235 | GAIN (beside) |
| WQ-musique | musique-4hop | 0.3086, 0.3148, 0.3352 | 0.3235 | +0.0148 | 0.0385 | WITHIN (beside) |


## L2: without topo_STRUCT, depth_STRUCT, WALK, WALKF

### Screen: FAIL

counts: {"GAIN": 0, "LOSS": 1, "WITHIN": 7}

| split | read | full s0, s1, s2 | cut s0 | delta | floor | call |
|---|---|---|---:|---:|---:|---|
| WQ-musique | hotpotqa | 0.8460, 0.8469, 0.8491 | 0.8448 | -0.0012 | 0.0075 | WITHIN |
| WQ-musique | 2wiki | 0.9155, 0.9159, 0.9133 | 0.9155 | +0.0000 | 0.0075 | WITHIN |
| WQ-musique | musique | 0.5190, 0.5227, 0.5381 | 0.5391 | +0.0201 | 0.0275 | WITHIN |
| WQ-hotpotqa | musique | 0.5806, 0.5803, 0.5777 | 0.5757 | -0.0049 | 0.0075 | WITHIN |
| WQ-hotpotqa | 2wiki | 0.9135, 0.9166, 0.9135 | 0.9128 | -0.0007 | 0.0075 | WITHIN |
| WQ-hotpotqa | hotpotqa | 0.8193, 0.8182, 0.8136 | 0.8178 | -0.0015 | 0.0082 | WITHIN |
| WQ-hotpotqa | musique-3hop | 0.5351, 0.5333, 0.5263 | 0.5276 | -0.0075 | 0.0127 | WITHIN |
| WQ-hotpotqa | musique-4hop | 0.3574, 0.3562, 0.3562 | 0.3481 | -0.0093 | 0.0075 | LOSS |
| WQ-musique | metaqa | 0.4126, 0.4256, 0.4167 | 0.3860 | -0.0266 | 0.0193 | LOSS (beside) |
| WQ-musique | squad | 0.9138, 0.9125, 0.9149 | 0.9122 | -0.0016 | 0.0075 | WITHIN (beside) |
| WQ-musique | webqsp | 0.6457, 0.6498, 0.6552 | 0.6602 | +0.0145 | 0.0146 | WITHIN (beside) |
| WQ-hotpotqa | metaqa | 0.3668, 0.4261, 0.4131 | 0.4506 | +0.0838 | 0.1064 | WITHIN (beside) |
| WQ-hotpotqa | squad | 0.9132, 0.9120, 0.9132 | 0.9134 | +0.0002 | 0.0075 | WITHIN (beside) |
| WQ-hotpotqa | webqsp | 0.6469, 0.6597, 0.6512 | 0.6577 | +0.0108 | 0.0191 | WITHIN (beside) |
| WQ-musique | musique-3hop | 0.4544, 0.4596, 0.4702 | 0.4811 | +0.0268 | 0.0235 | GAIN (beside) |
| WQ-musique | musique-4hop | 0.3086, 0.3148, 0.3352 | 0.3340 | +0.0253 | 0.0385 | WITHIN (beside) |


## L1: without topo_STRUCT, depth_STRUCT

### Screen: FAIL

counts: {"GAIN": 0, "LOSS": 1, "WITHIN": 7}

| split | read | full s0, s1, s2 | cut s0 | delta | floor | call |
|---|---|---|---:|---:|---:|---|
| WQ-musique | hotpotqa | 0.8460, 0.8469, 0.8491 | 0.8473 | +0.0013 | 0.0075 | WITHIN |
| WQ-musique | 2wiki | 0.9155, 0.9159, 0.9133 | 0.9174 | +0.0019 | 0.0075 | WITHIN |
| WQ-musique | musique | 0.5190, 0.5227, 0.5381 | 0.5405 | +0.0215 | 0.0275 | WITHIN |
| WQ-hotpotqa | musique | 0.5806, 0.5803, 0.5777 | 0.5763 | -0.0043 | 0.0075 | WITHIN |
| WQ-hotpotqa | 2wiki | 0.9135, 0.9166, 0.9135 | 0.9135 | +0.0000 | 0.0075 | WITHIN |
| WQ-hotpotqa | hotpotqa | 0.8193, 0.8182, 0.8136 | 0.8214 | +0.0021 | 0.0082 | WITHIN |
| WQ-hotpotqa | musique-3hop | 0.5351, 0.5333, 0.5263 | 0.5268 | -0.0083 | 0.0127 | WITHIN |
| WQ-hotpotqa | musique-4hop | 0.3574, 0.3562, 0.3562 | 0.3457 | -0.0117 | 0.0075 | LOSS |
| WQ-musique | metaqa | 0.4126, 0.4256, 0.4167 | 0.3714 | -0.0412 | 0.0193 | LOSS (beside) |
| WQ-musique | squad | 0.9138, 0.9125, 0.9149 | 0.9128 | -0.0010 | 0.0075 | WITHIN (beside) |
| WQ-musique | webqsp | 0.6457, 0.6498, 0.6552 | 0.6542 | +0.0085 | 0.0146 | WITHIN (beside) |
| WQ-hotpotqa | metaqa | 0.3668, 0.4261, 0.4131 | 0.3977 | +0.0309 | 0.1064 | WITHIN (beside) |
| WQ-hotpotqa | squad | 0.9132, 0.9120, 0.9132 | 0.9107 | -0.0025 | 0.0075 | WITHIN (beside) |
| WQ-hotpotqa | webqsp | 0.6469, 0.6597, 0.6512 | 0.6546 | +0.0077 | 0.0191 | WITHIN (beside) |
| WQ-musique | musique-3hop | 0.4544, 0.4596, 0.4702 | 0.4833 | +0.0289 | 0.0235 | GAIN (beside) |
| WQ-musique | musique-4hop | 0.3086, 0.3148, 0.3352 | 0.3222 | +0.0136 | 0.0385 | WITHIN (beside) |


