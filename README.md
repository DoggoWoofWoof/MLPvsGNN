# Relational Operators versus Message Passing for Retrieval

This is a standalone research repository for a NeurIPS-oriented study of when
graph message passing is worth its cost for retrieval. The frozen six-dataset
result is conditional: a plain MLP wins R@5 on 2Wiki and MuSiQue, message
passing wins on WebQSP, HotpotQA, and MetaQA, and SQuAD is neutral. The MLP is
3.64--9.92x faster with lower incremental peak GPU memory on every dataset.
The aim is to explain and predict this boundary, not manufacture a universal
MLP or GNN win.

## Scientific contract

- Same node/query features, relevance labels, candidates, loss, negative
  batches, validation budget, and seeds for every paired MLP/GNN run.
- Full-gradient GNN training. Detached candidate banks may be used only in a
  separately labeled scaling ablation.
- Validation-only selection and untouched test/OOD evaluation.
- Query-level paired confidence intervals plus hierarchical analysis across
  datasets, perturbations, architectures, and seeds.
- Retrieval effectiveness and mechanistic diagnostics are both required.
- CRAG is an experimental substrate, not a claimed contribution of this repo.

## Repository map

```text
configs/                       preregistered experiment grids
data/                          data contract; generated tensors are ignored
docs/NEURIPS_RESEARCH_PLAN.md  full paper and experiment blueprint
docs/OFFSET_OPERATOR_PROTOCOL.md frozen immediate screen and result gate
docs/OFFSET_SCREEN_RESULTS.md  complete-data one-seed screening result
docs/CONFIRMATION_PROTOCOL.md  frozen five-seed and capacity-selection gate
docs/CONFIRMATION_RESULTS.md five-seed replication and coverage diagnosis
docs/COVERAGE_VARIANT_PROTOCOL.md frozen single set-coverage Offset gate
docs/COVERAGE_VARIANT_RESULTS.md preregistered negative mechanism result
docs/SIX_DATASET_PROTOCOL.md frozen six-dataset comparison contract
docs/SIX_DATASET_RESULTS.md  five-seed effectiveness, hop, and systems tables
docs/SA_MLP_PROTOCOL.md      frozen fixed-structure MLP one-seed gate
docs/SA_MLP_SCREEN_RESULTS.md preregistered SA-MLP screen and fairness limit
docs/CRAG_EXTRACTION_AUDIT.md  what was retained, rewritten, or excluded
docs/PILOT3_RESULTS.md         non-reportable protocol-pilot audit and next gate
legacy/crag_snapshot/          exact provenance snapshots; not production code
scripts/export_crag.py         deterministic CRAG-to-neutral-format exporter
scripts/modal_pilot3.py        isolated parallel Modal execution entry point
src/mp_retrieval/              reusable models, perturbations, metrics, schema
tests/                         fast unit and contract tests
```

## Quick start

```bash
python -m pip install -e ".[graph,dev]"
pytest
```

Export one CRAG substrate without copying CRAG itself:

```bash
python scripts/export_crag.py \
  --crag-root ../CRAG \
  --dataset 2wiki_clean \
  --output data/2wiki_clean.pt \
  --encoder-subdir gte_qwen
```

The exporter records source paths, hashes, split policy, feature source, and
edge provenance in the artifact manifest. See [the data contract](data/README.md)
before running it.

## Current status

The six-dataset contract is preserved by the annotated tag
`six-dataset-protocol-v1`. Its stop gate is complete across 2Wiki, MuSiQue,
WebQSP, HotpotQA, SQuAD, and all 407,513 local MetaQA queries. Every primary
comparison uses five paired seeds; the four new-dataset GNN comparators were
selected from GCN, GraphSAGE, GAT, and GIN on validation R@5 only. See
`docs/SIX_DATASET_PROTOCOL.md` and `docs/SIX_DATASET_RESULTS.md`.

The result does not support a universal topology-free claim: R@5 has two MLP
wins, three GNN wins, and one neutral dataset under paired seed-level 95%
intervals. It does establish a consistent systems boundary: explicit message
passing costs 3.64--9.92x inference latency and 104--2,431 MiB of additional
incremental peak GPU memory while parameters are effectively matched. MetaQA's
GNN advantage is largest at one hop, so hop count alone cannot explain the
boundary. No new topology perturbation, mechanism predictor, Offset rescue, or
architecture was launched inside that stop gate.

The bounded SA-MLP screen frozen in `docs/SA_MLP_PROTOCOL.md` is complete and
passed on all three datasets. At seed 0, SA-MLP exceeds the frozen selected GNN
by 4.49 R@5 points on MetaQA, 1.55 on WebQSP, and 15.18 on HotpotQA. The
query-local descriptors explain nearly all of the MetaQA gain and most of the
HotpotQA gain; global degree/PageRank/core/clustering descriptors alone are
harmful. These are screening results, not five-seed paper estimates.

The screen also exposed the fairness control required before a causal claim:
its distance-0 feature reveals membership in the frozen dense/SPLADE seed set,
while the old GNN was not given that indicator. The combined SA architecture
will remain unchanged during confirmation. A seed-only MLP and seed-aware GNN
must separate the retrieval prior from the value of fixed graph paths/PPR.
See `docs/SA_MLP_SCREEN_RESULTS.md`.

Earlier pilot and rewiring outputs remain `NOT_PAPER_VALID_PILOT`; old C-RAG
results are hypothesis-generating only. The prior Offset screen, confirmation,
and failed coverage variant remain recorded in their corresponding protocol
and results documents.

To run the gated complete-data operator screen on isolated Modal storage:

```bash
python experiments.py run operator-screen --backend modal
```

Run the frozen confirmation gate on isolated Modal storage:

```bash
python experiments.py run confirmation --backend modal
```

Run the frozen SA-MLP one-seed gate on isolated Modal storage:

```bash
python experiments.py run sa-mlp-screen --backend modal
```

Compute credentials are resolved at runtime from an ignored local file or
environment variables. Use `configs/compute.local.yaml.example`; never copy a
private CRAG credential file into this repository.
