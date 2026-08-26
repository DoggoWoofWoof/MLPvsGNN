# SA-MLP feature leakage audit

Status: **passed before the legacy 2Wiki/MuSiQue relaunch**.

## Allowed inputs

The fixed-structure feature values are functions only of:

- frozen dense top-5 and SPLADE top-5 seed IDs;
- frozen candidate IDs and their stable order;
- the frozen directed graph topology;
- fixed numerical settings registered in `configs/sa_mlp_screen.yaml`.

No textual edge confidence exists in the frozen graph contract, so none is
used. No support annotation, gold document ID, relevance position, training
label, validation metric, prediction, or learned model state enters feature
generation.

## Code-path audit

`build_static_features` reads only `graph.pt` and produces degree, PageRank,
hub percentile, coreness, and clustering columns. `_seed_arrays` reads only
`CompleteQuery.retrieval_seed_local`. `_candidate_arrays` reads only query and
candidate positions. `_local_feature_chunk` accepts packed seed arrays, packed
candidate-induced topology, and fixed damping/iteration constants; it has no
label or gold argument.

The candidate contract hash used in cache metadata includes gold IDs so that a
changed experimental artifact cannot silently reuse a cache. That hash affects
cache identity only; it is never provided to a feature formula or learned
model.

## Mutation test

`tests/test_structural_features.py::test_structural_cache_is_finite_aligned_and_label_independent`
constructs the full cache, changes both `relevant_local` and `relevant_global`,
reconstructs the cache separately, and requires the static and query-local
arrays to remain bit-identical. The confirmation tests additionally require
`retrieval_seed_labels_used == false` and verify that the seed indicator is
constructed exclusively from frozen local seed positions.

## Conclusion

SA-MLP is **not topology-free**. It is a non-message-passing, fixed-structure
model. Its graph computation is deterministic and inference-safe, while the
learned scorer never receives adjacency or aggregates neighbor embeddings.
