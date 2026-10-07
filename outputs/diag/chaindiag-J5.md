# Within the depth that holds a gold: what tells it from the top-1 (a diagnosis; no training, decides nothing)

at-depth miss: the top-1 is not gold and sits at a depth that holds a gold; g the gold at that depth the fit scores highest. same_own: g and the top-1 equal on every non-SEMB column of the fit's blocks; same_all: on all 112 (step 1's nine blocks and rel's three). favours_gold: the share where g's value is above the top-1's, ties half. qrel_max: the largest over the twelve question-relation match columns.

**Reading (declared before its numbers):** MATCH_GAP (full_rel/J5@p@swa, metaqa=s1eval, 3hop; INPUT_GAP if same_rel >= 0.5, else TRAINING_GAP if qrel_max >= 0.65, else MATCH_GAP)

## metaqa s1eval

### all: 9785 questions

| fit | at-depth misses (share) | same_own | same_rel | same_all | columns differing (mean) | qrel_max (column) | gnn0 prefers gold | twin0 prefers gold |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| step1/J5@p@swa | 2590 (0.2647) | 0.0000 | 0.0004 | 0.0000 | 39.74 | 0.6448 (typed_v2:relpath_max_h3) | 0.8853 | 0.7402 |
| full_rel/J5@p@swa | 1436 (0.1468) | 0.0000 | 0.0021 | 0.0000 | 39.99 | 0.5129 (typed_rel:relchain2_max) | 0.7389 | 0.3914 |

step1/J5@p@swa, the 15 columns furthest from one half (favours_gold): typed_v2:relpath_max_h3 0.645, typed_rel:relchain2_max 0.631, dense_cos:0 0.370, typed_rel:relmean_in 0.625, typed_v2:relpath_mean_h3 0.610, typed_rel:relmax_in 0.605, rank:2 0.395, typed_v2:relpath_min_h3 0.603, ordered:opath_h3_q1 0.582, rank:0 0.419, SEED:0 0.423, ordered:opath_h3_q2 0.576, SEED:1 0.438, ordered:opath_h3_q3 0.553, depth_STRUCT:12 0.449

full_rel/J5@p@swa, the 15 columns furthest from one half (favours_gold): typed_v2:qsupport_h3 0.392, depth_STRUCT:12 0.402, DISTS:11 0.402, rank:2 0.436, typed_v2:relpath_max_h3 0.443, dense_cos:0 0.449, depth_STRUCT:0 0.548, typed_v2:relpath_mean_h3 0.457, ordered:opath_h3_q2 0.463, depth_STRUCT:4 0.537, typed_v2:relpath_min_h3 0.464, rank:1 0.465, rank:0 0.466, typed_rel:rel_ief 0.466, ordered:opath_h3_q1 0.467

### 1hop: 2498 questions

| fit | at-depth misses (share) | same_own | same_rel | same_all | columns differing (mean) | qrel_max (column) | gnn0 prefers gold | twin0 prefers gold |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| step1/J5@p@swa | 400 (0.1601) | 0.0000 | 0.0025 | 0.0000 | 43.77 | 0.7338 (typed_rel:relmean_in) | 0.7975 | 0.76 |
| full_rel/J5@p@swa | 235 (0.0941) | 0.0000 | 0.0128 | 0.0000 | 45.71 | 0.5809 (typed_v2:qsupport_h3) | 0.6596 | 0.5234 |

step1/J5@p@swa, the 15 columns furthest from one half (favours_gold): typed_rel:relmean_in 0.734, typed_rel:relmax_seed 0.731, typed_v2:relpath_max_h3 0.730, typed_v2:relpath_min_h3 0.726, ordered:opath_h3_adj23 0.701, typed_v2:relpath_mean_h3 0.694, ordered:opath_h3_q3 0.691, typed_v2:qsupport_h3 0.671, typed_rel:relmax_in 0.665, dense_cos:0 0.345, DISTS:9 0.632, depth_STRUCT:10 0.376, WALK:2 0.383, topo_STRUCT:5 0.385, topo_STRUCT:6 0.385

full_rel/J5@p@swa, the 15 columns furthest from one half (favours_gold): dense_cos:0 0.332, rank:2 0.360, typed_rel:relmax_in 0.366, typed_v2:relpath_mean_h3 0.366, typed_v2:relpath_min_h3 0.366, ordered:opath_h3_q3 0.370, typed_rel:relmax_seed 0.374, typed_v2:relpath_max_h3 0.379, WALKF:9 0.389, rank:0 0.394, ordered:opath_h3_adj23 0.396, typed_rel:relmean_in 0.398, depth_STRUCT:12 0.602, DISTS:11 0.602, WALK:9 0.400

### 2hop: 3718 questions

| fit | at-depth misses (share) | same_own | same_rel | same_all | columns differing (mean) | qrel_max (column) | gnn0 prefers gold | twin0 prefers gold |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| step1/J5@p@swa | 874 (0.2351) | 0.0000 | 0.0 | 0.0000 | 40.76 | 0.9068 (typed_rel:relchain2_max) | 0.9554 | 0.8318 |
| full_rel/J5@p@swa | 263 (0.0707) | 0.0000 | 0.0 | 0.0000 | 43.05 | 0.6198 (typed_rel:relchain2_max) | 0.7947 | 0.4639 |

step1/J5@p@swa, the 15 columns furthest from one half (favours_gold): typed_rel:relchain2_max 0.907, typed_rel:relmean_in 0.732, typed_rel:relmax_in 0.717, dense_cos:0 0.378, typed_v2:qsupport_h2 0.605, rank:2 0.399, SEED:0 0.401, rank:0 0.409, SEED:1 0.418, depth_STRUCT:2 0.423, DISTS:10 0.423, DISTS:5 0.424, depth_STRUCT:4 0.570, depth_STRUCT:7 0.433, SEED:2 0.435

full_rel/J5@p@swa, the 15 columns furthest from one half (favours_gold): typed_rel:relchain2_max 0.620, SEED:2 0.597, typed_rel:rel_ief 0.405, typed_v2:qsupport_h3 0.424, depth_STRUCT:12 0.426, DISTS:11 0.426, depth_STRUCT:6 0.428, SEED:0 0.567, DISTS:5 0.567, topo_STRUCT:6 0.435, depth_STRUCT:0 0.565, depth_STRUCT:4 0.565, depth_STRUCT:10 0.435, SEED:1 0.563, WALK:1 0.437

### 3hop: 3569 questions

| fit | at-depth misses (share) | same_own | same_rel | same_all | columns differing (mean) | qrel_max (column) | gnn0 prefers gold | twin0 prefers gold |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| step1/J5@p@swa | 1316 (0.3687) | 0.0000 | 0.0 | 0.0000 | 37.83 | 0.6968 (typed_v2:relpath_max_h3) | 0.8655 | 0.6733 |
| full_rel/J5@p@swa | 938 (0.2628) | 0.0000 | 0.0 | 0.0000 | 37.70 | 0.5325 (ordered:opath_h3_q3) | 0.7431 | 0.338 |

step1/J5@p@swa, the 15 columns furthest from one half (favours_gold): typed_v2:relpath_max_h3 0.697, ordered:opath_h3_q1 0.646, typed_v2:relpath_mean_h3 0.641, ordered:opath_h3_q2 0.639, dense_cos:0 0.372, depth_STRUCT:12 0.373, DISTS:11 0.373, typed_v2:relpath_min_h3 0.620, rank:2 0.390, rank:0 0.421, ordered:opath_h3_adj23 0.422, SEED:0 0.442, ordered:opath_h3_adj12 0.552, depth_STRUCT:6 0.547, typed_v2:typed_walks_h3 0.541

full_rel/J5@p@swa, the 15 columns furthest from one half (favours_gold): typed_v2:qsupport_h3 0.335, depth_STRUCT:12 0.345, DISTS:11 0.345, depth_STRUCT:6 0.577, topo_STRUCT:5 0.574, WALK:11 0.572, WALKF:11 0.566, depth_STRUCT:11 0.565, DISTS:13 0.565, rank:2 0.438, WALK:2 0.558, WALKF:2 0.556, topo_STRUCT:6 0.553, typed_v2:relpath_max_h3 0.450, depth_STRUCT:4 0.549

## webqsp s1eval

### all: 1503 questions

| fit | at-depth misses (share) | same_own | same_rel | same_all | columns differing (mean) | qrel_max (column) | gnn0 prefers gold | twin0 prefers gold |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| step1/J5@p@swa | 214 (0.1424) | 0.0000 | 0.0 | 0.0000 | 78.15 | 0.7874 (typed_rel:relmax_in) | 0.9579 | 0.9393 |
| full_rel/J5@p@swa | 326 (0.2169) | 0.0000 | 0.0 | 0.0000 | 76.48 | 0.6074 (typed_rel:relmax_in) | 0.865 | 0.7975 |

step1/J5@p@swa, the 15 columns furthest from one half (favours_gold): rank:1 0.033, rank:2 0.056, typed_rel:relmax_in 0.787, typed_v2:relpath_mean_h3 0.771, ordered:opath_h3_q3 0.762, SEED:2 0.243, DISTS:4 0.750, typed_v2:relpath_max_h3 0.750, WALKF:10 0.255, topo_STRUCT:6 0.720, WALK:11 0.715, WALK:10 0.290, WALK:1 0.708, DISTS:12 0.708, typed_v2:relpath_min_h3 0.708

full_rel/J5@p@swa, the 15 columns furthest from one half (favours_gold): rank:1 0.144, rank:2 0.192, depth_STRUCT:0 0.242, rank:3 0.265, depth_STRUCT:12 0.307, DISTS:11 0.307, DISTS:9 0.314, DISTS:1 0.321, SEED:2 0.328, typed_rel:dir_in_frac 0.330, rank:0 0.337, dense_cos:0 0.337, WALKF:3 0.347, WALKF:10 0.348, SEED:0 0.350

