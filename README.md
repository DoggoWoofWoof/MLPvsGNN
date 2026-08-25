# When Is Message Passing Worth It for Retrieval?

This is a standalone research repository for a NeurIPS-oriented study of the
conditions under which graph message passing improves retrieval and the
conditions under which it damages otherwise useful node representations.

The intended contribution is a **predictive phase diagram**, not the isolated
observation that an MLP can beat a GNN. We compare parameter-matched MLP and
message-passing encoders while controlling graph quality, feature quality,
degree, hubness, density, edge semantics, topology noise, and training size.

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
`paper-protocol-v0`. A three-dataset Modal pilot and one audited 25%
degree-preserving-rewiring point have run; both are stamped
`NOT_PAPER_VALID_PILOT` because the current caches are incomplete/test-only.
See `docs/PILOT3_RESULTS.md` for the fairness audit, results, limitations, and
next-run gate. The old CRAG results remain **hypothesis-generating only**.

To run the registered three-dataset pilot on isolated Modal storage:

```bash
python experiments.py run pilot3 --backend modal --intervention clean --rate 0
```

Compute credentials are resolved at runtime from an ignored local file or
environment variables. Use `configs/compute.local.yaml.example`; never copy a
private CRAG credential file into this repository.
