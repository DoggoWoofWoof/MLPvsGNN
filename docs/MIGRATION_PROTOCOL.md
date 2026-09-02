# Cross-workspace migration protocol

Status: **ACTIVE — infrastructure only.** Nothing in this document changes a
scientific protocol, a candidate pool, a frozen result, or a selected rate. It
describes how frozen state is moved to a second Modal workspace after the
original workspace hit a spend limit, and what has to be true before any runner
is allowed to resume there.

## 0. Why a migration at all

`ResourceExhaustedError: Workspace ac-1Zd8AkijYgSgLk37ju340f has exceeded its
spend limit` blocks every function call in the source workspace. Volume **reads**
still succeed, which is what makes a migration possible without any compute
there.

A Modal Volume lives in exactly one workspace and there is no cross-workspace
copy, so the data has to travel through local staging: `download` under the
source profile, `upload` under the target profile. Volume names are
workspace-scoped, so once a slice exists in the target, `MODAL_PROFILE` alone
retargets a job — no launcher change, which is the same mechanism CRAG uses.

**The migration must not become a rerun.** A workspace with capacity is not a
substitute for missing state; anything already computed is carried across and
skipped, not recomputed.

## 1. Two stages, two manifests, no shared queue

Phase −1 and E2 need different things, and Phase −1 needs much less. They are
staged separately so a topological measurement is never queued behind data it
does not read.

| | Phase −1 | E2 resume |
|---|---|---|
| runner | `run_graph_substrate_audit.py` | `run_phase_confirmation.py` |
| dataset-root files | 6 (topology only) | 8 |
| embedding matrices | **no** | yes |
| `derived/` caches | no | `packed_topology_v1`, `fixed_structural_features_v1` |
| `outputs/` trees | no | yes |
| files | 107 | 1,826 |
| bytes | **4.2 GB** | 32.4 GB |

The manifests are emitted by `scripts/migration_provenance.py manifests` and
derived from the code that opens the files, so they cannot drift from the
runners:

- `outputs/migration/phase1_required_manifest.json`
- `outputs/migration/e2_resume_manifest.json`

### Why Phase −1 carries no embeddings

`load_complete_dataset` consults `nodes.npy` and `queries_all.npy` only for
`.ndim` and `.shape`, and only to cross-check alignment. The substrate audit
reads `rowptr`, `col`, `num_nodes`, `queries`, `split(...)` and the candidate
contract metadata, and never an embedding value. `require_embeddings=False`
makes that explicit and states in the error message what validation it gives up.
The alternative — shipping 12 GB of unread bytes, or worse, shipping stubs — was
rejected. Payload fell from 16.2 GB to 4.2 GB.

## 2. What is deliberately not migrated

`phase_confirmation_cache/` — **193.6 GB, 85% of the volume, not migrated.**

It is what `build_or_load_perturbed_topologies` and
`build_or_load_structural_features` write, keyed by intervention contract, and
both rebuild it when it is absent. Only the three topology axes populate it;
`feature_mask` masks node features on device and reuses the clean caches.

Skipping it is a claim, and the claim is tested rather than assumed — see §4.

## 3. Artifacts that cannot be regenerated

`run_phase_confirmation` reuses the **seed-0 validation checkpoint trained during
E1** instead of retraining it, and verifies the checkpoint file's SHA-256 against
the value E1 recorded before loading it:

```
checkpoint_path = Path(source["checkpoint_path"])
if _sha256(checkpoint_path) != source["checkpoint_file_sha256"]:
    raise ValueError("Validation-screen seed-0 checkpoint failed SHA-256")
```

That makes the 192 `outputs/phase_screen/**/checkpoints/*/seed_0.pt` files
(96 cells × 2 models) the load-bearing artifacts of this migration. If one does
not arrive intact, seed 0 cannot be reproduced without retraining, and retraining
it would break the `seed_zero_validation_checkpoint_reused_without_test_peeking`
contract.

Two properties make this safe rather than merely hoped-for:

1. **The expected hashes come from the frozen E1 results, not from the transfer.**
   `migration_provenance.py provenance` resolves all 192 from the local E1
   records, so the check is independent of the copy being checked.
2. **The runner is the last line of defence.** A corrupted checkpoint makes E2
   refuse to start rather than silently retrain — the failure mode is a stop, not
   a quietly different number.

Absolute paths inside the frozen results resolve unchanged because the volume
mounts at the same `storage_root` in both workspaces
(`volumes={STORAGE_ROOT: result_volume}`).

## 4. The regeneration gate

Before relying on the omission in §2, a declared sample of cache cells is
captured from the source and compared against cells regenerated from the frozen
clean inputs. Spec: `configs/cache_equivalence.yaml`. Runner:
`scripts/run_cache_equivalence.py`. Launcher: `scripts/modal_cache_equivalence.py`.

**The representative set is declared before any comparison runs**
(`migration_provenance.reference_cache_cells`): every topology axis at the lowest
and highest non-zero rate on the smallest dataset, plus one mid-rate cell per
topology axis on the second-smallest — 9 cells. The failure being tested is a
property of the generator, not of a particular cell, so breadth across axes and
rate extremes is worth more than volume.

**Equivalence is not byte-identity.** `metadata.json` records wall-clock build
seconds, so two correct runs always differ there. The comparison is:

- every `.npy` compared on dtype, shape and **every element, with no tolerance**;
- every metadata field compared exactly, **except** a declared set of timing keys
  which is printed in the output rather than applied quietly;
- `contract_sha256` compared explicitly, for both cache kinds, because that is
  the field `build_or_load_structural_features` itself refuses to load past.

**Quarantine.** Captures are written under `migration_reference/` and
regenerations under `migration_regenerated/`, never under
`phase_confirmation_cache/`. At their own paths the captures are exactly what
`build_or_load_*` looks for, so the regeneration would never happen and the
comparison would be a file against itself. `replicate_volume.py` refuses to
upload the `cache_reference` slice without a quarantine prefix, and the runner
refuses to regenerate into a non-empty directory.

The gate is not closed by the local run. A local pass is supporting evidence on
a different OS and CPU; the binding claim is that the **target container**
rebuilds the same cache, because that is where E2 resumes.

## 5. The E2 resume plan comes from an integrity matrix

Not from an ordinal position in a job list. `migration_provenance.py matrix`
classifies all 96 cells using the rule `run_phase_confirmation` enforces on
itself:

| state | action | meaning |
|---|---|---|
| `COMPLETE` | skip | status is `PHASE_CONFIRMATION_CELL_COMPLETE` **and** all 2 models × 5 seeds recorded |
| `PARTIAL` | resume | contract valid, model-seeds still missing |
| `MISSING` | launch | no `result.json` at the cell path |
| `INVALID` | diagnose | unreadable, contract mismatch, or a completion claim the payload contradicts |

`INVALID` never becomes a relaunch. A result claiming completion while missing
seeds is a corrupt artifact; overwriting it destroys the only evidence of
whatever produced it.

The four contract fields checked — `dataset`, `axis`, `rate`,
`data_fingerprint_sha256` — are exactly the four the runner refuses to continue
past, so the matrix cannot classify as resumable anything the runner would reject.

Cell paths are `outputs/phase_confirmation/<dataset>/<fingerprint[:16]>/<axis>_<rate>/`.
Note that `fingerprint[:16]` is **not** the dataset directory name: 2wiki's data
root is `.../6e4ac5ee0e1355ad` while its cell prefix is `d7c2da85e2b65680`. A
matrix built on the directory name would report a finished sweep as 96 `MISSING`
cells and invite a full, expensive re-run.

The same trap is one directory level away. Cell paths are volume-relative and
already carry the `outputs/` component, so `--results-root <staging>/outputs`
searches `outputs/outputs/` and reports 96 `MISSING` — identical output to a
sweep that never ran. `misrooted_hint` refuses that case and names the root that
would have worked; an honest zero is still reported as zero.

### Measured state at migration time

| dataset | COMPLETE | PARTIAL | MISSING | INVALID | model-seed units |
|---|---|---|---|---|---|
| 2wiki_clean | 16 | 0 | 0 | 0 | 160/160 |
| musique_clean | 16 | 0 | 0 | 0 | 160/160 |
| webqsp | 16 | 0 | 0 | 0 | 160/160 |
| hotpotqa_clean | 0 | 10 | 6 | 0 | 88/160 |
| squad_clean | 0 | 0 | 16 | 0 | 0/160 |
| metaqa | 0 | 0 | 16 | 0 | 0/160 |
| **total** | **48** | **10** | **38** | **0** | **480/960 in COMPLETE cells** |

Three datasets are finished. Nothing is `INVALID`. The spend limit landed inside
hotpotqa's GNN arm: every partial cell has all five QLS-MLP seeds and is missing
GNN seed 4, sometimes 3 as well — 12 model-seed units across ten cells. The
remaining work is those 12 units plus 380 in the 38 unstarted cells, 392 of 960.

The 48 `COMPLETE` cells are skipped, not recomputed. That is the whole point of
deriving the plan rather than restarting the package in a workspace that happens
to have capacity.

## 6. Transfer integrity

`scripts/replicate_volume.py` — `plan`, `download`, `upload`, `verify`.

- the source volume is opened `create_if_missing=False` and only ever read;
- `create_if_missing=True` appears in exactly one place, under `upload`;
- every file records size and SHA-256; `verify --deep` re-reads the target and
  re-hashes, concurrently, so the gate before a launch is not the slowest step;
- `download` and `upload` are resumable and skip what is already in place;
- a slice narrowed with `--datasets` keeps its own manifest, so a partial run can
  never pass for the whole slice.

### Defects found and fixed during the migration

Each is recorded because the failure mode matters more than the fix.

**Truncated read accepted silently by the transport.**
`dense_top200_all.npy` delivered 150,913,344 of 156,563,328 bytes and the
generator simply stopped — no exception. Only the byte-count check caught it.
Fix: bounded retries on a fresh stream (a short read is a transport fault, not a
corrupt source), the partial file discarded rather than resumed, and per-file
failures collected so one bad file no longer discards the record of the other
hundred that arrived intact. `_load_manifest` refuses a staging directory with
recorded failures, so an incomplete slice cannot be promoted to a replica.

**`volume.commit()` from a local client.**
`commit()` is for a volume mounted inside a container. `batch_upload` already
commits on block exit, so the extra call raised *after* 4 GB had landed — every
upload would have "failed" with the data in place.

**`listdir(".")` reported as an absent tree.**
The volume root is `/`. `_walk` swallowed the `NotFoundError` and returned
nothing, so a fully populated replica verified as 107 `MISSING` files. Fix: the
root spelling is normalised, and walks against a replica are `strict=True` — a
listing failure is raised rather than reported as "nothing arrived".

## 7. Provenance record

`migration_provenance.py provenance` emits
`outputs/migration/migration_provenance.json`: source and target workspace,
volume names, mount point, slice; git SHA, branch, working-tree cleanliness and
tags at HEAD; per-protocol tag, tag commit and **document SHA-256** (the tag pins
history, the hash pins the file as it stands, and recording both makes an edit
after the tag visible); per-dataset data root, fingerprint, cell prefix, selected
GNN and candidate-contract compatibility; the 192 non-regenerable checkpoints
with their E1-recorded hashes; and the staged file list with sizes and hashes.

All non-regenerable target artifacts are verified against this record before E2
resumes.

## 8. What this migration does not touch

- **CRAG** — strictly read-only, never modified.
- **E1** — 96/96 complete, integrity-audited, frozen. Never restarted.
- **Packages A, B, C, D** — frozen results, carried across unchanged.
- **Package F** — sealed and unopened.
- Candidate pools, candidate hashes, historical pools, selected rates, protocol
  definitions, historical tags.

Scientific protocols do not change because of infrastructure inconvenience. If
a migration constraint and a protocol conflict, the migration gives way.
