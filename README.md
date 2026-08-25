# Relational Operators versus Message Passing for Retrieval

This is a standalone research repository for a NeurIPS-oriented study of when
a lightweight query-conditioned relational operator can replace explicit graph
message passing for retrieval.

The immediate question is whether an Offset-MLP can match or beat strong GNN
operators in retrieval quality while improving parameters, memory, and latency.
The broader phase diagram remains a conditional follow-up, not the current
execution priority. The aim is to discover the outcome, not manufacture an MLP
win.

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

The exact protocol baseline is preserved by the annotated tag
`paper-protocol-v0`. The current gate is the complete-data, one-seed seven-model
screen on WebQSP, 2Wiki, and MuSiQue defined in
`docs/OFFSET_OPERATOR_PROTOCOL.md`. Earlier pilot and rewiring outputs remain
`NOT_PAPER_VALID_PILOT`; old C-RAG results are hypothesis-generating only.

To run the gated complete-data operator screen on isolated Modal storage:

```bash
python experiments.py run operator-screen --backend modal
```

Compute credentials are resolved at runtime from an ignored local file or
environment variables. Use `configs/compute.local.yaml.example`; never copy a
private CRAG credential file into this repository.
