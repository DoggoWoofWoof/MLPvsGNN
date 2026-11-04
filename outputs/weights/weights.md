# Weight counts of the two models of record

Read on 5 Nov 2026 from each fit's saved `p@swa` state (`outputs/weights/count_weights.py`, read-only); the six fits of each model (seeds 0 to 2, both training sets) hold the same tensors, so the count is one number.

| model | weights | tensors | parts |
|---|---:|---:|---|
| zrckpn (the MLP of record) | **226,950** | 23 | lookup tables U, V 114,752; chain match (cm_*) 53,795; scoring layers l1, l2, out 58,241; pointwise KT head (kt_*) 161; scale (base_w) 1 |
| zg6r (the GNN of record) | **404,955** | 59 | encoder (15 tensors, copied from the zspk fit) 224,737; own message-passing stack (g6_*, g6r_*) 180,218 |

The GNN's saved model holds 1.78 times the MLP's weights. Its own stack alone (180,218, the count in docs/G6RW_G6R_UNDER_WQ.md) is 0.79 times the MLP's; the rest of its state is an encoder of the MLP's shape (lookup tables, chain match, two layers), copied from the zspk fit and carried into every read.
Quote the totals (226,950 and 404,955) when the two sizes are set side by side; the 180,218 alone leaves the GNN's encoder out.
