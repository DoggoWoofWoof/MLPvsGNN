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

### What the local run found

Non-binding, and it found two separate things. Recorded here in full because a
gate that only reports its verdict is a gate nobody can check.

**The perturbation regenerates exactly. All nine cells, both datasets, all
three axes, both rate extremes.** `packed_topology_v1` came back bit-identical
every time -- on `webqsp/degree_rewire/0.10` that is `edge_index`
(2 x 3,666,732 int32), `edge_ptr` (1,579 int64) and `query_position`
(1,578 int64), all exactly equal -- and its cache `contract_sha256` matched too.
The generator is seeded and order-stable, and its cache contract does not bind
anything environmental.

**The structural features differ by exactly one ulp.** `local.npy` (float16)
differed in 92-501 of 5.4M elements depending on the cell, always with a maximum
absolute difference of 0.00048828125; `static.npy` (float32) differed in 54,135
elements on 2wiki_clean and 639,808 on webqsp, with maxima of
1.7881393432617188e-07 and 1.1920928955078125e-07. Every one of those maxima is
exactly one ulp at its dtype -- the signature of a different floating-point
reduction order, here Windows local against a Linux container. This is why the
local run cannot close the gate in either direction: E2 resumes in a Linux
container, and the original cells were built in one too.

**Two contract hashes differ, and the second is a consequence of the first.**
Not three: the perturbation *cache* contract is stable, as above. What differs
is the intervention record in `perturbation.json`.
`perturb_packed_topologies` puts `build_seconds` -- a wall-clock measurement --
inside the dict it hashes into that `contract_sha256`, so two correct runs of the
same computation always produce different intervention contracts. The attribution
diagnostic reports `semantic_metadata_differences: {}` on all nine cells: kind,
rate, seed and every recorded edge count are identical, and the sole differing
key is the hash itself. That hash then feeds `feature_fingerprint`, which is why
the structural-feature contract differs as well.

That defect is *not* fixed here. Changing what the hash covers changes it for
every future build and interacts with cells that are already frozen, so it is a
decision to take deliberately rather than in passing. What was added instead is
an attribution diagnostic: it records whether a contract difference survives
once the two records are given the same build time, so a reader can tell "the
cache is genuinely different" from "the hash bound a clock". **The
pre-registered verdict is unchanged** -- a differing `contract_sha256` is still
reported as a difference.

### A defect the local run exposed in the gate itself

All three `2wiki_clean` cells failed before regenerating, with *"Candidate
contract differs from the frozen record; refusing to regenerate"*. The pool had
not changed.

Two of the six frozen pools -- `2wiki_clean` and `musique_clean` -- were hashed
before candidate hop metadata existed. `configs/phase_confirmation.yaml` marks
them `candidate_contract_compatibility: pre_hop_metadata_v1`, and the frozen E1
result records both hashes side by side along with a
`BIT_EXACT_FROZEN_CANDIDATE_EQUIVALENCE` proof that they describe the same
pool. For 2wiki_clean: `expected_contract_sha256` is `11609cd7…` while
`current_contract_with_hop_metadata_sha256` is `e763a748…`, which is exactly
what a fresh `load_complete_dataset` computes today.

The gate compared the frozen hash against the fresh one with `!=`, so it read a
field-set difference as a changed candidate pool. `webqsp` passed only because
it has no legacy mode and all three of its hashes coincide.

The fix is not a looser comparison. `regenerate_cell` now delegates to the same
`validate_candidate_contract` the experiment runners use, with the compatibility
mode read from `configs/phase_confirmation.yaml` and carried on the declared
cell — so the gate asks exactly the question E2 asks when it resumes, and still
refuses a pool whose query IDs, split, candidate order or gold IDs really
differ. The mode and the resulting proof are recorded in the output, because a
legacy-mode result otherwise prints a frozen hash the fresh load never produces
and reads as a contradiction.

Nothing was overwritten: the gate raises before it writes.

### What the container run found -- this is the binding result

Nine cells, both datasets, all three topology axes, both rate extremes, run on
the target workspace in the same image E2 runs in. All nine returned
`CACHE_REGENERATION_DIFFERS`, and the reason is the same on all nine and is
now fully characterised.

**Everything semantic regenerates bit-identically.** `packed_topology_v1`
matched completely: `edge_index.npy`, `edge_ptr.npy` and `query_position.npy`
equal in dtype, shape and every byte, and its metadata equal on every compared
key. `fixed_structural_features_v1` matched on all four arrays --
`candidate_ptr.npy`, `query_position.npy`, `static.npy` (float32) and
`local.npy` (float16). The one-ulp float differences the *local* run reported
do not appear here; they were an artifact of regenerating on Windows, not a
property of the pipeline. In the environment that matters, the regeneration is
byte-for-byte.

**Exactly two metadata keys differ, and neither can ever match.** On every cell
the differing keys are `fixed_structural_features_v1`'s `contract_sha256` and
`source_fingerprint_sha256`, and the second causes the first. Tracing them:

```
source_fingerprint = sha256(data_fingerprint
                            + intervention["contract_sha256"]
                            + "clean_global_static_features")
```

and `intervention["contract_sha256"]` is computed in
`local_topology_perturbations.perturb_packed_topologies` as a digest over
`json.dumps(metadata, sort_keys=True)` -- where `metadata` contains
**`build_seconds`**, a `time.perf_counter()` measurement. A hash over a
wall-clock duration is not reproducible by construction. It is not a
regeneration defect; it is a hash that never had a chance of matching, and no
amount of determinism upstream would change it.

**Nothing compares those hashes against anything frozen.** This is the part
that decides the migration, and it was traced rather than assumed:

*   `build_or_load_perturbed_topologies` validates a cached `perturbation.json`
    on `kind`, `requested_rate` and `seed` only. It never checks
    `contract_sha256`.
*   `build_or_load_structural_features` checks the cached
    `contract_sha256` against one it recomputes from the *same*
    `source_fingerprint` it was handed -- a within-cell consistency check, not
    a comparison to a frozen value.
*   `run_phase_confirmation._screen_seed_zero` validates the E1 screen result
    on status, dataset, axis, rate, `data_fingerprint_sha256`,
    `test_metrics_computed is False` and `training_seed == 0`. It does **not**
    compare any feature-cache or perturbation contract. (The check at
    `run_sa_mlp_confirmation.py:543` that does compare a recorded feature
    contract belongs to `_reuse_screen_seed`, which E2 does not call.)

So a regenerated cell is self-consistent and indistinguishable to E2 from a
transferred one, and a transferred cell keeps its original hashes and is
equally acceptable. Both paths work; they simply produce different values for a
field nothing reads.

**Decision: the 193.6 GB `phase_confirmation_cache` is not migrated.** It is
regenerated from the clean inputs at the target. The 23.8 GB of cache belonging
to the ten PARTIAL cells is not migrated either, and for a second reason
besides size: a partially written cache can carry a complete `metadata.json`
beside a truncated array, which the loader would accept. Regenerating into an
empty root cannot hit that.

**What is *not* claimed.** The perturbation contract hash is not reproducible,
and this protocol does not change its definition to make it so -- that hash is
frozen and a result-driven redefinition is exactly what is forbidden. The claim
is narrower and is what the gate measured: every array and every semantic
metadata key regenerates byte-for-byte, and the fields that differ are not read
by any consumer. The nine result JSONs are kept under
`outputs/cache_equivalence/` and record which hash matched, under which
compatibility mode, with the proof digest.

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

### The one artifact that cannot be regenerated, verified before launch

The 192 seed-0 checkpoints are the only thing in this migration that cannot be
rebuilt. `run_phase_confirmation` loads each one and refuses on a SHA-256
mismatch -- but it does that *after* its container is up and its inputs are
mounted, so a single corrupted transfer costs a GPU container per affected
cell.

`migration_provenance.py provenance` now hashes every staged copy against the
`checkpoint_file_sha256` the frozen E1 result recorded for it, and exits
non-zero unless all 192 verify. On the staged `e2_resume` copy:

```
screen seed-0 checkpoints required: 192/192
staged copies verified by SHA-256: 192/192
```

Only the file hash is checked here. E2 also verifies a hash of the loaded
state dict, which catches a file that hashes correctly but deserializes to
something else; that check needs torch and is left where it already lives
rather than reimplemented.

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
