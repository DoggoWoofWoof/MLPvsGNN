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
docs/SA_MLP_CONFIRMATION_PROTOCOL.md frozen five-seed seed-prior controls
docs/SA_MLP_CONFIRMATION_RESULTS.md six-dataset fairness result and stop decision
docs/SA_FEATURE_LEAKAGE_AUDIT.md label-free fixed-structure code-path audit
docs/LEGACY_CANDIDATE_COMPATIBILITY.md bit-exact legacy digest bridge
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

The fairness confirmation frozen at `sa-mlp-confirmation-protocol-v1` is now
complete on all six datasets and five paired seeds. The unchanged SA-MLP is
compared separately with plain MLP, seed-only MLP, and a seed-aware copy of the
validation-selected GNN. Fixed graph summaries add significant R@5 signal over
the seed prior on MetaQA (+6.86 points), WebQSP (+4.11), and HotpotQA (+3.70).
The seed prior recovers at least 80% of the SA gain only on HotpotQA, so the
registered seed-prior explanation fails (1/3; required 2/3).

Using the conservative requirement that both paired-seed and paired-query 95%
intervals clear the -1 point margin, SA-MLP is non-inferior to the seed-aware
GNN on MetaQA and HotpotQA (2/3; gate passed); WebQSP is inconclusive at query
level. Across all six datasets, SA-MLP is 2.49--7.08x faster online and saves
90--2,418 MiB of incremental peak GPU allocation, with preprocessing and disk
cache costs reported separately. The boundary is explicit: SA-MLP trails the
seed-aware GNN by 1.44 R@5 points on 2Wiki and therefore fails the one-point
margin; MuSiQue's -0.96 mean is too uncertain to certify non-inferiority, while
SQuAD is non-inferior. The confirmation gate is closed: these models must not
be tuned further against test data, and any new perturbation, mechanism, or
practical-width experiment requires a separate preregistration. See
`docs/SA_MLP_CONFIRMATION_RESULTS.md`.

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

Run the frozen six-dataset confirmation on isolated Modal storage:

```bash
python experiments.py run sa-mlp-confirmation --backend modal
```

Compute credentials are resolved at runtime from an ignored local file or
environment variables. Use `configs/compute.local.yaml.example`; never copy a
private CRAG credential file into this repository.
