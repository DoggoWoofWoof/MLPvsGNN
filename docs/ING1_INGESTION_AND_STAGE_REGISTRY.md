# ING1: one ingestion seam, one stage registry, pinned rules per stage

Declared 26 Nov 2026, 22:58 IST, before any ING1 code exists. This page declares structure and tooling. It declares no research number, edits no frozen module, refiles nothing and starts no host job.

## 1. Why this page exists

The user, 26 Nov, in their words: "we need to properly maintain this repo whatever dataset in future we will get make sure we can properly process it a centralized ingestion system then encoding, links, pool and its rules etc then our models, still am not sure about the models and pool integration but atleast whatever we can unify make sure we properly do, cant reprocess everything again and again and keep changing rules".

Three requests inside it: (1) any future dataset enters through one door and then gets the same encoding, links and pools; (2) what can be unified is unified, and the models and pool integration are allowed to stay open; (3) rules stop changing silently, and reprocessing is never repeated by accident.

## 2. What exists (read from the repository on 26 Nov)

- **The seam below the stages already exists: the package.** The six original datasets live in `final_canonical/<dataset>/` with `DATASET.json` (record, nodes, queries, graph, embeddings, shas) and one `CANONICAL_FREEZE.json`. MultiHop-RAG (`docs/MHR1_MULTIHOPRAG_PACKAGE.md`) and the three STaRK knowledge bases (`docs/STK1_STARK_PACKAGE.md`) were built to the same layout, each with its own `DATASET.json` and a `PACKAGE_FREEZE.json`. Encoding (`zenc.py`) reads any package through `served_root`, which `zmhr.py` and `zstk1.py` point at their packages.
- **Above the package there is no seam.** Dataset names are tuples inside the stage modules (`zenc.SIX`, `zul2p.DS7`, `SEVEN` in the pool modules). Forty-one modules in `outputs/mp_unified` name `multihoprag` (a grep count, 26 Nov): adding that one dataset was touched in every stage by hand, and the loaders differ (`load_host` against `load_mhr` in `zul2p.py`). STK2 (links) and STK3 (pools) would repeat that for three more datasets.
- **No table says what is done.** The six's e3v vectors and pools are on the host; the laptop holds MultiHop-RAG's e3v only. Which stage of which dataset was produced under which rule and code is written inside each stage's own record (most carry a page and a `code_sha256`), but nothing reads them together.
- **Rules are pinned by pages and code hashes and the code is frozen.** That is the honesty mechanism and it stays. What is missing is the other half: when a rule does change, nothing lists what it makes stale, so the answer is rebuilt by hand each time.

## 3. What ING1 declares

Nothing below edits a frozen module. Everything new is a new file, written once, and amended only through a dated amendment on this page.

**3.1 Registry** (`configs/datasets.json`). One row per dataset, ten rows: name, kind (`passage` or `kb`), the package root (the candidates on the laptop and on the host), the adapter that builds the package (module and page), the S source (hyperlinks, triples, none), the node-text rule (the original passage text, name-only, T1), the encoder arms that exist, the published comparator page. This is the only place a dataset name is listed. New code reads it and never a tuple.

**3.2 Rule sets** (`configs/rule_sets.json`). Per stage, the current rule: page, commit, code hash, and a status **FINAL** or **OPEN**. A new dataset is processed under the FINAL rules only; an OPEN stage is listed as open and no rule is claimed for it. The file is written from the filed records, not from memory. Its first content (stages and what the records say today):

| stage | rule | status |
|---|---|---|
| ingest | the package contract above; node text of the original datasets as filed; T1 for the STaRK KBs | FINAL |
| encode | e3v (D101), one arm, frozen | FINAL |
| first stage | e3v and SPLADE fused by reciprocal rank, depth 2,500; BM25 as a read | FINAL |
| links, layout | S, R, K families, deduplicated S > R > K, S never removed, labels by rule | FINAL |
| links, tightening | none adopted; the base link set stands | OPEN (the user has not said the link set is final) |
| pools | PL6's picks reach 0.95 on four of seven; no rule carried | OPEN |
| grade | ALL-golds coverage at distinct nodes | FINAL |
| models | one MLP and one GNN per family; how they use the pools | OPEN |

**3.3 Stage status** (`ingest.py status`). For each dataset and stage, read the record the stage already wrote and compare it to the rule set: **DONE** (the record exists and names the pinned rule and code), **STALE** (a record exists under another rule or code), **MISSING**, **N/A** (the stage does not apply: S on the passage datasets that have no structure). Nothing is recomputed. A cell also says where the artifact is (laptop or host).

**3.4 The rule-change protocol.** A rule change is a new entry in `rule_sets.json`, committed with its page before any number. `status` then lists every cell it makes STALE as a plan, with the cost seen in past rows. **Nothing re-runs until the user names which datasets.** This is how reprocessing stops being repeated by accident: it becomes a visible, priced, chosen action.

**3.5 Generic package gates** (`ingest.py check --package DIR`). The consistency gates that STK1 wrote as P1, made independent of the builder, reading only the package's files: node keys contiguous and equal to `DATASET.json`, the splits partition the queries, golds exist, are distinct and in range, edges are in range and have a named relation, manifests have as many rows as nodes and queries, the shas equal the record and the freeze. A new dataset enters the system when its package passes.

**3.6 Plan** (`ingest.py plan DATASET`). From the registry and the rule sets, print the feed rows for the stages that are MISSING or STALE, with the sizes of past rows. It prints; queueing is a separate, explicit step.

**3.7 The world loader** (ING1-B, `outputs/mp_unified/zworld.py`, a new declared module). One read-only `load(dataset)` for all ten: node ids and texts, query rows with split and golds, S edges, paths of the vectors, shas. Conformance gate **G-W**: on the original six and MultiHop-RAG the loader's counts and content hashes equal the legacy loaders' (`zenc` readers, `zul2p.load_host`, `zul2p.load_mhr`). Frozen stages keep their own loaders and nothing is refiled. New stages (STK2 links, STK3 pools, every later one) are written against `zworld` only after G-W passes.

## 4. Phases

- **A** (this page's first step): `configs/datasets.json`, `configs/rule_sets.json`, `scripts/ingest/ingest.py` with `check`, `status`, `plan`. Read-only, laptop-runnable, host-runnable through `rx exec`.
- **B**: `zworld.py` and G-W.
- **C**: STK2 and STK3 on the three STaRK knowledge bases, written against `zworld`; each page says which rule of the seven it applies. They are the first consumers, so they test whether the pool machinery accepts a package without per-dataset edits. That is an open question, not a claim.
- **D, open**: models. The input contract is whatever the pools hand over; nothing here unifies the model code, and the user has said this part is unsettled.

## 5. What ING1 does not do

It does not rewrite, rename or refile any frozen module or number; does not decide the link tightening, the final pool rule or the models; moves and deletes no artifact (deletes stay on the user's per-action OK); and runs nothing on the host except read-only `status`.

## 6. Expectations, written before any run

- **E1** `check` passes on all ten packages, or names exactly which gate fails and why. The six were built by an older builder, so some gates (a node-id format, a manifest field) may not apply; where one does not, it is reported as N/A with the reason, not silently skipped.
- **E2** `status` agrees with what is known: STaRK's encode shows running or missing; the original datasets' links and pools show DONE under their pages; no cell is guessed.
- **E3** G-W passes on the six and MultiHop-RAG, or the differing counts are reported and not forced.
- **E4** `plan` for a made-up eleventh dataset produces the complete chain of rows, and for STaRK produces only what is not already queued.

## 7. Result

(empty until the tooling has run)
