# P0 A1 MetaQA identity compatibility amendment

Status: **FROZEN BEFORE METAQA TEST EVALUATION**.

This amendment changes no candidate, split, label, fusion, selection, or
evaluation rule in `p0-rank-controls-protocol-v1`. It only specifies how the
existing MetaQA entity-string gold IDs are mapped to the already-frozen global
node rows when evaluating directly from the local CRAG source tree.

## Trigger

The five passage datasets use gold IDs whose numeric suffix is the frozen node
row. MetaQA instead uses entity strings such as `metaqa_ent_carol reed`. The
local `metaqa/gte_qwen` directory has no materialized `node_ids.json`, so the
numeric-suffix compatibility rule is inapplicable.

## Frozen identity rule

For MetaQA only, when `node_ids.json` is absent:

1. read the existing sibling artifact `metaqa/splade_doc_embs.pkl`;
2. use only its frozen `id_to_idx` field;
3. require a bijection onto every integer row in `[0, N)`;
4. require `N` to be greater than every node row present in the frozen Dense
   and SPLADE rank arrays;
5. map gold entity IDs through this bijection; and
6. fingerprint the identity source in the result artifact.

The sparse SPLADE matrix in that pickle is not used. `graph.pt`, graph edges,
node embeddings, query embeddings, partitions, and labels other than the
evaluation gold IDs are not loaded for rank-only scoring.

## Failure behavior

Evaluation fails closed if the sidecar is missing, malformed, non-bijective,
does not cover the rank-array node rows, or omits any gold entity ID. The code
must not infer a row from entity spelling, regenerate identities, or modify the
read-only CRAG source tree.

## Reproducibility note

The original protocol and implementation remain preserved by their existing
tags. This amendment and its implementation receive separate commits/tags so
the post-freeze compatibility correction is explicit.
