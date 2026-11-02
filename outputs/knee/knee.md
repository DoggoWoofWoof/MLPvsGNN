# PKO depth knee (declared in docs/PKO_DEPTH_KNEE.md)

## scr-ml1-zrckp-s0 (zrckp): reproduces ML1

| carve | no cut | 500 | 1000 | 1500 | 2000 | 3300 | knee 0.01 | knee 0.02 |
|---|---|---|---|---|---|---|---|---|
| 2wiki=s1eval | 0.9045 | 0.9045 (499 rows, reach 0.984) | 0.9045 (937 rows, reach 0.988) | 0.9045 (1079 rows, reach 0.990) | 0.9045 (1081 rows, reach 0.990) | 0.9045 (1081 rows, reach 0.990) | 500 | 500 |
| hotpotqa=s1eval | 0.8423 | 0.8423 (478 rows, reach 0.974) | 0.8423 (691 rows, reach 0.979) | 0.8423 (706 rows, reach 0.979) | 0.8423 (706 rows, reach 0.979) | 0.8423 (706 rows, reach 0.979) | 500 | 500 |
| metaqa=s1eval | 0.3992 | 0.4050 (500 rows, reach 0.900) | 0.4020 (1000 rows, reach 0.956) | 0.4008 (1500 rows, reach 0.977) | 0.4002 (2000 rows, reach 0.986) | 0.3994 (3300 rows, reach 0.994) | 500 | 500 |
| musique=s1eval | 0.5051 | 0.5051 (500 rows, reach 0.922) | 0.5051 (1000 rows, reach 0.947) | 0.5051 (1500 rows, reach 0.956) | 0.5051 (2000 rows, reach 0.966) | 0.5051 (3299 rows, reach 0.976) | 500 | 500 |
| squad=s1eval | 0.9139 | 0.9139 (50 rows, reach 0.980) | 0.9139 (50 rows, reach 0.980) | 0.9139 (50 rows, reach 0.980) | 0.9139 (50 rows, reach 0.980) | 0.9139 (50 rows, reach 0.980) | 500 | 500 |
| webqsp=s1eval | 0.6562 | 0.6149 (500 rows, reach 0.806) | 0.6388 (1000 rows, reach 0.881) | 0.6450 (1500 rows, reach 0.910) | 0.6469 (2000 rows, reach 0.926) | 0.6529 (3300 rows, reach 0.950) | 2000 | 1000 |

common depth within 0.01: 2000; within 0.02: 1000; DEPTH_FLAT: True; carve flags: none

