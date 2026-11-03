# DOB under the WWW rule: the seven drops on the six in-domain reads

Each drop's six reads: the four dataset reads are DOB's calls rows; musique's 3-hop and 4-hop slices on WQ-hotpotqa come from the read files (zdep.slice_r5), with section 4's floor from the CPU base's three seeds.

| drop | block | own call | WWW screen | GAIN reads | LOSS reads |
|---|---|---|---|---|---|
| xdcos | dense_cos | MIXED | NOT_ADOPTED |  | WQ-musique hotpotqa, WQ-musique 2wiki |
| xtopo | topo_STRUCT | HURTS | ADOPT_FOR_FULL_RUN | WQ-hotpotqa musique-3hop |  |
| xdepth | depth_STRUCT | HURTS | NO_GAIN |  |  |
| xseed | SEED | HURTS | NO_GAIN |  |  |
| xwalk | WALK | HURTS | NO_GAIN |  |  |
| xdists | DISTS | IDLE | NO_GAIN |  |  |
| xwalkf | WALKF | HURTS | NO_GAIN |  |  |

| split | read | base s0, s1, s2 | floor | xdcos | xtopo | xdepth | xseed | xwalk | xdists | xwalkf |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| WQ-musique | hotpotqa | 0.8411, 0.8424, 0.8388 | 0.0075 | -0.0105* | -0.0006 | +0.0028 | +0.0006 | +0.0033 | +0.0007 | +0.0007 |
| WQ-musique | 2wiki | 0.9070, 0.9079, 0.9049 | 0.0075 | -0.0080* | +0.0069 | +0.0027 | +0.0019 | +0.0034 | +0.0019 | -0.0063 |
| WQ-hotpotqa | musique | 0.5749, 0.5718, 0.5752 | 0.0075 | -0.0063 | +0.0035 | +0.0000 | -0.0036 | -0.0032 | -0.0030 | -0.0045 |
| WQ-hotpotqa | 2wiki | 0.8987, 0.9082, 0.9063 | 0.0172 | +0.0065 | +0.0127 | +0.0075 | +0.0023 | +0.0059 | +0.0035 | +0.0080 |
| WQ-hotpotqa | musique-3hop | 0.5241, 0.5237, 0.5232 | 0.0075 | -0.0048 | +0.0154* | -0.0009 | -0.0013 | +0.0004 | -0.0061 | +0.0013 |
| WQ-hotpotqa | musique-4hop | 0.3475, 0.3556, 0.3531 | 0.0138 | -0.0012 | +0.0056 | +0.0062 | +0.0043 | +0.0037 | +0.0019 | +0.0074 |

\* a GAIN or a LOSS at the read's floor.
