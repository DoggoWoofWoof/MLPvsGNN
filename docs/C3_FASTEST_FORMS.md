# C3: every model's fastest exact form, under one rule

Declared 10 October 2026, about 03:50, before any timed number. Stage C3 of docs/PROGRAM_2026_10_09.md. It follows
C2 (docs/C2_MLP_COLD_FAST.md), whose result sent it here: the MLP's cold lead over the six GNN was under 6× on every
untyped dataset.

C2 gave the MLP its exact path, and the GNN kept C1's. The comparison was not even. C3 gives **both families** their
fastest form under the **same rule**, so the MLP gets no advantage the GNN was denied.

No model is trained, chosen or changed. The models are C1's:
- the MLP (zrc) and the GNN track's base (zsp), from outputs/b1b/fits/{zrc,zsp}/J5, candidate p@swa;
- the six GNN (u_gnn_v2_ef).

## The rule (the same for every model)

An optimised serving form is admitted only if it is the **same function on the same weights**. On every timed
question:
- its scores are within **TOL = 1e-4** of the reference form's scores (max absolute difference);
- its top 5 is the **same set** as the reference form's.

The tolerance exists because a fused or folded form adds the same terms in a different order. Nothing is approximated.
No pruning, no quantisation, no caching across questions and no change to any input the model reads is allowed.

## What C3 changes

All the forms are in `outputs/mp_unified/c3_fast.py` (new; frozen once run).

**The six GNN: `FusedGNN`**, a subclass of gnn_fast.FastGNN.
- Every module's own GEMMs stay in torch.
- The per-edge elementwise work runs in one numba pass per destination node: gathers, the attention sum, the segment
  softmax and the message sum. Profiled cold on musique, this work was three quarters of the forward.
- The relation encoder, and the cell's rel1, rel2 and msg_r, run on the **distinct** edge-attribute rows. On musique
  these are about 900 of 55,000. Each edge then gathers its row.
- Each family's segment mean of the node projections is summed in edge order per node, the same additions as
  index_add_ (bit for bit; the selftest checks it).

Its compile, pool and pack are C2's GNN path unchanged: the full fast compile, then `fast_pack`. The GNN reads all 197
columns, so it has no columns to drop.

**The MLP (zrc) and the GNN track (zsp): `FastZ`.**
- Each block's [raw, z-score, mask] concatenation is folded into the first layer. The mask columns are 1 at serving,
  so they become a bias.
- Every block's z-scores come in one pass over the question's rows.
- zsp's propagation step, with each row's neighbour z-score mean, soft maximum and degree per family, is one numba pass
  over the links.

**The MLP's inputs.**
- `seed_dists_fast` builds lean_cache.store_seed_dists' SEED and DISTS blocks:
  - the seed products P @ P[S]ᵀ are computed once, where dist_fast had computed them a second time;
  - the pairs come from a bucket sort (`pairs_fast`), the same arrays as lean_mlp2.pairs' np.unique;
  - the distance kernel is unchanged.
- `links_fast` gives zsp's links (c1_cold.links_of, by the same bucket sort).

Both are **bit-identical** to C2's: the checks compare the rows and the links bit for bit.

**Batch 16 (warm).**
- `fastz_batch` is FastZ's folded forward over a group's rows, with lean_screen3's per-question z-scores and zprop's
  propagation inputs.
- FusedGNN runs on the group's packed graphs.

**Development check, before this file.** `c3_fast.py check` ran on 30 questions each of 2wiki and musique:
- every question passes every check below;
- batch 16 is within 5e-6 (2wiki) and 4e-5 (musique) of batch 1.

The check's times are development numbers, not filed numbers.

## What is timed

C1's protocol, as C2 ran it:
- laptop, pinned to logical processor 2, EcoQoS off, ABOVE_NORMAL priority (own process only);
- one BLAS, numba and torch thread; session idle;
- the first 200 s1eval questions, cold;
- numba's compile on the carve's 201st question, reported as index time.

Five paths. **Each runs its whole chain on its own**: pool, embedding read, its compile and its own stages. Nothing is
shared between paths, and their order rotates question by question.

| path | what it is |
| --- | --- |
| **zrc3** | the MLP: C2's `compile_mlp`, C2's lean inputs with `seed_dists_fast`, then FastZ |
| **zsp3** | the GNN track: zrc3's inputs, plus `links_fast`, then FastZ with the propagation step |
| **gnn3** | the six GNN: C2's GNN path (full fast compile, `fast_pack`), then FusedGNN |
| **zrc2** | the MLP in C2's form, the reference for zrc3, timed in the same run |
| **gnn6** | the six GNN in C1/C2's form (FastGNN), the reference for gnn3, timed in the same run |

**Warm** is reported beside cold and labelled warm:
- **batch 1:** the fast paths' forward repeated on the inputs they have just built (`warm_forward`);
- **batch 16:** one forward per fast path over each group of 16 questions, 13 groups, inputs built at batch 1.

Warm counts the forward and the top 5. It does not count building the inputs.

Datasets: the four untyped ones, 2wiki, hotpotqa, squad and musique, as C1 and C2 filed. Typed graphs (metaqa,
webqsp) follow in their own file.

## Checks (untimed, every question)

- **pools:** every path's pool equals the look's;
- **seeds:** each MLP path's seeds and buckets equal the look's;
- **rows bits:** zrc3's and zsp3's float16 rows and store codes equal zrc2's bit for bit;
- **links bits:** zsp3's links equal c1_cold.links_of's on zrc2's edges, in value and dtype;
- **zrc3:** within TOL of zrc2's scores, same top 5;
- **zsp3:** within TOL of zsp's C1/C2 forward on zrc2's rows and links, same top 5;
- **gnn3:** within TOL of gnn6's scores, same top 5;
- **gnn6:** against the look's stored scores, with the same top 5, as C1 checks it;
- **batch 16:** each fast path's batched scores against its batch-1 scores (the largest difference is reported).

A question failing any check is reported, and its timings are kept out of the summaries. More than 2% failing on a
dataset stops that dataset.

## What is reported

`outputs/c3/<dataset>.json` per dataset, and `outputs/c3/report.{json,md}`. All latencies are per question.

**Cold, batch 1:**
- p50, p95 and p99 per path, each p50 with a bootstrap CI;
- **gnn3 / zrc3**: the six GNN over the MLP, both at their fastest. This is the headline;
- gnn3 / zsp3 and zsp3 / zrc3;
- the speed-ups gnn6 / gnn3 and zrc2 / zrc3;
- gnn6 / zrc2, C2's ratio, read again in this run;
- per-stage p50 per path, split as C1 splits cost: index (once), pool and compile (query-local), inputs,
  forward.

**Warm:** batch-1 forward p50 per fast path and gnn3 / zrc3; batch-16 per-question forward p50 per fast path.

There are no thresholds. C3 reads the ratios and files them with their CIs. If gnn3 / zrc3 is under 6× cold on a
dataset, the per-stage table names the stage that holds the MLP back. Those stages are almost all retrieval and
compile, which the two families share. They are the next stage's target, under this file's rule.

## Not in this stage

- Typed graphs.
- The GNN track's own exact forms for other GNN arms.
- Warm caches of compiled pools: latency is cold, as in C1; warm is only the forward.
- Any server or batching policy beyond batch 16.

## Result (filed 10 October 2026, about 03:55)

Run on the laptop under the protocol above (outputs/c3/{2wiki,hotpotqa,squad,musique}.json, report.{json,md}).

**Every check passed on every question: 800 of 800.**
- The pools, seeds, rows and links are bit for bit on all four datasets.
- The largest score differences are well inside TOL:
  - zrc3, against zrc2: 8.5e-5 (musique) and 3.6e-6 to 7.6e-6 elsewhere;
  - zsp3: 5.6e-5;
  - gnn3, against gnn6: 3.8e-6.
- Every fast form has the same top 5 as its reference on every question.
- Batch 16 is within 8.3e-5 of batch 1.

**Cold, batch 1, p50 ms per question** (bootstrap 95% CI on the ratios):

| dataset | MLP zrc3 | GNN track zsp3 | six GNN gnn3 | **gnn3 / zrc3** | gnn3 / zsp3 | gnn6 / gnn3 (GNN's speed-up) | zrc2 / zrc3 (MLP's speed-up) |
| --- | ---: | ---: | ---: | --- | --- | --- | --- |
| 2wiki | 6.2 | 7.0 | 11.4 | **1.84 [1.73, 1.90]** | 1.62 [1.58, 1.70] | 1.10 [1.05, 1.14] | 1.57 [1.48, 1.62] |
| hotpotqa | 5.7 | 6.2 | 10.4 | **1.84 [1.72, 1.95]** | 1.67 [1.59, 1.74] | 1.09 [1.05, 1.14] | 1.57 [1.50, 1.66] |
| squad | 2.6 | 3.0 | 5.9 | **2.23 [2.17, 2.31]** | 1.98 [1.94, 2.05] | 1.17 [1.14, 1.25] | 2.10 [2.04, 2.15] |
| musique | 52.9 | 60.1 | 179.3 | **3.39 [3.34, 3.41]** | 2.98 [2.95, 3.01] | 1.84 [1.83, 1.87] | 1.24 [1.23, 1.25] |

p99 cold, MLP vs six GNN:

| dataset | MLP | six GNN |
| --- | ---: | ---: |
| 2wiki | 13.9 | 19.8 |
| hotpotqa | 10.9 | 15.8 |
| squad | 4.4 | 9.7 |
| musique | 65.0 | 212.5 |

**Warm forward, p50 ms per question** (inputs already built; labelled warm, not the headline):

| dataset | MLP b1 | six GNN b1 | **gnn3 / zrc3 b1** | MLP b16 | six GNN b16 | gnn3 / zrc3 b16 |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| 2wiki | 0.91 | 4.34 | **4.77 [4.62, 4.90]** | 0.58 | 3.07 | 5.3× |
| hotpotqa | 0.84 | 4.04 | **4.84 [4.68, 5.06]** | 0.51 | 2.97 | 5.8× |
| squad | 0.60 | 3.16 | **5.28 [5.09, 5.56]** | 0.31 | 1.84 | 6.0× |
| musique | 8.11 | 102.9 | **12.69 [12.50, 12.88]** | 11.95 | 111.3 | 9.3× |

### What it says

**Both families are at their fastest exact form, under the same rule.**
- The six GNN gained 1.09–1.84× cold, most on musique, where its forward is large.
- The MLP gained 1.24–2.10×.
- The MLP is faster than the six GNN on every dataset:
  - cold: **1.8–3.4×**;
  - warm at batch 1: **4.8–12.7×**;
  - warm at batch 16: **5.3–9.3×**.
- The GNN track's base (zsp) costs 1.08–1.14× the MLP cold. It stays 1.6–3.0× faster than the six GNN.

**The cold lead is under 6× everywhere.** The per-stage p50s say why:

| stage | 2wiki (MLP) | 2wiki (GNN) | musique (MLP) | musique (GNN) |
| --- | ---: | ---: | ---: | ---: |
| pool + embedding read (shared) | 1.42 | 1.40 | 11.27 | 10.91 |
| compile | 2.53 | 4.33 | 17.05 | 60.44 |
| inputs (edges, lean or pack) | 0.78 | 0.24 | 15.50 | 3.38 |
| forward | 1.36 | 5.07 | 8.81 | 102.98 |

- The pool and the embedding read are the same for both families: 21–23% of the MLP's cold time.
- On the small graphs, the MLP's compile is 41% of its cold time and its forward only 22%.
- On musique, the MLP's compile and lean inputs are 61% of its cold time.

On 2wiki, even a free MLP forward and free MLP inputs would leave the MLP at about 4 ms against the GNN's 11.4. The
MLP's remaining cold cost is its compile (retrieval, the pool's edges, topology and the depth basis) and its lean
inputs (walks, seed distances). The GNN pays for most of the same compile, and more.

**Next (each in its own file):**
- **C4, exact:** fuse the MLP's compile kernels and lean inputs (walk with walkf, the depth basis, the seed distances)
  under this file's rule. Whatever kernel the GNN's compile shares gets the same fused form, so the comparison stays
  even.
- **Cost-aware inputs (a model change):** an MLP that reads cheaper inputs is a training question, not a serving form.
  It belongs to the joint feature-selection stage (docs/PROGRAM_2026_10_09.md), where accuracy and cold cost are read
  together on all six datasets.
