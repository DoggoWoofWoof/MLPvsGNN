# Data contract and status

No CRAG corpus, embedding matrix, checkpoint, or generated experiment artifact
is committed to this repository. Source datasets have their own licenses and
the useful CRAG substrates occupy tens of gigabytes.

## L2 candidate artifacts (primary experiment)

`L2CandidateDataset` stores:

- the global corpus graph in CSR form;
- names of the candidate evidence signals;
- one query record containing global candidate IDs, raw expert scores, an
  authoritative availability mask, global/local relevance IDs, and a split;
- source paths and SHA-256 hashes in metadata.

The candidate graph for a query is the subgraph induced by its candidate IDs.
The MLP and GNN receive exactly the same candidate features. Only the GNN sees
the induced edges.

Current `results/L2/signals_*` CRAG caches contain test queries only. The L2
exporter marks these artifacts `pilot_test_only`; the runner refuses to train on
them unless `--allow-pilot-resplit` is supplied. That flag is exclusively for
engineering and emits `NOT_PAPER_VALID_PILOT` in the result.

Canonical paper artifacts must be regenerated with:

1. stable query IDs and official train/validation/test membership;
2. candidate IDs and scores for every split;
3. raw typed edges retained before graph construction;
4. a documented candidate-generation checkpoint frozen before reranker work;
5. train-only normalization statistics;
6. hashes and upstream license metadata.

## Global graph artifacts (secondary experiment)

`GraphRetrievalData` supports open-corpus node retrieval. It stores node/query
features, global edges, optional edge types, query relevance sets, and split
labels. This is useful for checking whether the L2 phenomenon survives outside
a candidate pool, but it is not the first experiment to run.

## Never copy into Git

- `master_nodes_*.json`
- `nodes.index`, `nodes.npy`, or query embedding arrays
- `graph.pt`
- `relsig_feats_*.pt`
- checkpoints, predictions, or per-run outputs

Generate local artifacts under `data/processed/`; the directory is ignored.

