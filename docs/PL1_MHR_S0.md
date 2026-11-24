# PL1 S0: MultiHop-RAG's pool record chain (declared 25 November 2026, before any number of this page is read)

Governing pages: `docs/PL1_POOLS_ON_UL2_LAYOUT.md` (section 2f: "the MultiHop-RAG record chain is new modules, declared as step S0 after a
read-only survey"; section 2g: the e3v pools carry no EM relation type and no score class) and `docs/UL2_LAYOUT_ON_E3V.md` section 13
(MultiHop-RAG's populations and layout). The user's ruling: MultiHop-RAG is treated as any other dataset, every rule universal, nothing
special-cased, and the pools are finished first. This page declares the chain that gives MultiHop-RAG what the six had before PL1, so that its
RK arm is graded like theirs. It adds **no new rule**: every rule below is one the six were built with, applied to MultiHop-RAG's own data.

## 1. What the six had, and what MultiHop-RAG lacks

The six datasets reached PL1 with a frozen **own construction** (M3B's knee rule on the headroom grid, 13 September), U2m's cells and U2s's
record of them, and PKO's pools under pick O. MultiHop-RAG has none of these: `canonical.Dataset` refuses its name, `carves.json`,
`op.construction`, U2s's record, `zu2c.graph_csr`, M3B's CSR cache and E0's files are all keyed to the six. The survey (read-only, done before
this page) found that no pinned module has to change: `zpko.pools_pass(name, carve, ..., opener=, rule=)` takes the opener and the rule as
arguments, and `zpko.QUESTIONS`, `U2S_CARVE` and `SHARD_Q` are tables the PL1 modules already patch in-process. So S0 is one new module,
`outputs/mp_unified/zpl1h.py`, that imports the pinned modules, edits none, and patches in-process (every patch put back on exit).

## 2. Inputs (all filed; each is sha-recorded in S0's records)

- Package: 609 nodes, 2,255 queries (train 901, dev 224, test 1,130; ids in `queries/query_ids.json` order, train then dev then test).
- First stage: **e3v dense** (the e3v vectors of MHR1, searched over all 609 nodes by `zenc.search`, the call `zmhr.py first` made) and
  **SPLADE** (MHR1's `retrieval_cache/splade_top1000.npz`). Lists are 609 deep; the pool code asks for 1,000, so both are padded with the
  sentinel node id n and NaN scores exactly as `zpl1w.world` pads E0's lists. The frozen-encoder lists (MHR1's `dense_top1000.npz`) are read
  once, beside, for 4.
- Links: `outputs/mhr1/links/rule/{F0T,W2_S}.npz`, `links/e3v/K.npz`; F0T's per-pair weights (which only order each node's neighbour row) are
  rebuilt in-process by `zul1.build_entries` and `zul1.forms_from_entries` and joined by `d83stores.form_pairs`, the rule `zpl1m.f0t_triple` reads
  from files for the six (UL1's own gate G2 shows these are U2r's CN_all weights). No carried forms: UL2's call for MultiHop-RAG is not filed.
- Populations (UL2 13.2): **s1sel = dev (224 questions), s1eval = test (1,130)**, in query-file order, zero-gold questions excluded and counted
  (`m3b_compile.Population`, digest = `step4.digest` of the kept ids).

## 3. The chain (each step's record is committed before the next step's number is read)

| step | what | output (under `outputs/pl1/multihoprag/RK/`) |
| --- | --- | --- |
| S0a | The first stage: the e3v and SPLADE lists for every question; the gold table by question id (positions of `gold_node_ids`, in E0's gold-cache shape `ids, flat, off`) | `s0/first_e3v.npz`, `s0/gold.npz`, `s0/first.json` |
| S0b | The layout: `zpl1m.tables(n, "RK", S=None, W2_S, F0T, K_e3v, carried=None)` and `zpl1m.gates`; the typed store holds W2_S (label R), the ner store F0T, the knn store K_e3v after the S > R > K dedup. Same file shape as the six's (`layout.npz`, `layout.json`); `layout_kind` says B0, because the carried forms are not filed | `layout.npz`, `layout.json` |
| S0c | **The own construction**, by M3B's knee rule on dev: `m3b_contract.apply_rule(cells, KNEE 0.90, BOUND 2500)` through `m3b_contract.finish_record`, over the 31 cells the committed headroom grid has per dataset (seven retrieval pools: dense_top200, splade_top200, frozen_union, equal_rrf_budget_{50,100,200,400}; four graph regimes STRUCT, NER, KNN, FULL x three settings h1_c25, h1_c100, h2_c25 on the bases equal_rrf_budget_50 and _200), computed with `m3a_headroom`'s own `prefix_rows`, `rrf_fused_rows`, `stable_union_rows`, `FamilyCSR`, `expand`, `pool_union_rows` and `cell` on dev's golds, in the e3v world and on S0b's three stores. The six's "others" reading (the rule over every cell, no ruled family) applies; no prospective cell is added. The construction is then frozen like the six's; a different pool later is a new declared block. Beside, decides nothing: the same read on the frozen-encoder lists | `s0/knee_e3v.json`, `s0/knee_frozen.json` |
| S0d | The record, in the fields `zpko.open_pko` checks (cells = `zu2m.cells_of(construction)`, own_cell, cell_base, graph_detail, walk, G, pick, beta, alpha, eps, nmax) plus the null types' shas. alpha, eps, nmax, G, beta, pick and walk are the carried constants, checked equal to the four passage datasets' U2s records. There are no U2s arrays: U2s's stats stage is not re-run, because the pick is carried (U2s's verdict, pick O) and the pool rule needs only the own cell | `s0/record.json` |
| S0e | The pools on dev (`s1sel`): `zpko.pools_pass(name, carve, opener=zpl1h's, rule=zpl1w.make_rule(...))`, one shard, 2 threads, then the places | `dev/pools/parts/`, `dev/places/` |
| S0f | The pools on test (`s1eval`), only after S0e's gates pass: one shard (`SHARD_Q` is 3,000 for a passage dataset), the file names zpl1g reads (`multihoprag__s1eval__0of1`) | `pools/parts/`, `places/` |

The opener (S0e, S0f) is `zu2d_pools.open_carve("new", ...)` run unchanged under in-process patches: a stand-in `Dataset` for the package
(nodes, queries, caches, vectors, graph, `name`, `n_nodes`, `n_queries`, `query_ids`), `E.open_dataset` and `C4.populations` for the two
carves, `op.construction["multihoprag"]` from S0c, `U2C.graph_csr` and `ZP.u2e_graph` for W2_S's CSR, `m3b_pools.load_or_build_store` for the
ner and knn stores (`zpl1m.to_stores`), and `E.union_pools` as `zpl1w.world` patches it (B = base united with the seeds, I = the prepared pools).
The context is `open_pko`'s with 2g's null types in memory (`zpl1w.null_type_arrays`: mu zero, every entry the null class, the pair weights
exactly 1): `ty = (pair_types(cls, NULL, k),) + no_types()[1:]`, `ws = P.weights(zeros, beta)`, walk = "RWe@0.5 x LW@0.5". The rule is
`zpko.rule_batch` as it is; the places are written from the spy `make_rule` already uses and change no fact.

## 4. Gates (a failure stops the chain; none of the later numbers is read)

| gate | what | stops |
| --- | --- | --- |
| G-K | **The knee code is the rule the six were read with.** S0c's loop and `finish_record`, run on MuSiQue's dev population with its own frozen lists and M3B's graph, reproduce `outputs/m3a/headroom/musique.json`'s 31 committed rows (candidates_mean and fraction_of_attainable@5, each to 1e-9) and `outputs/m3b/contract/musique.json`'s construction | S0c |
| G-E | MultiHop-RAG's e3v and SPLADE top-20 equal `outputs/mhr1/first/lists.npz` for every question; the e3v hit and R@5 on all questions equal `outputs/mhr1/first.json` | S0a |
| G-F | F0T's in-process pair keys equal `links/rule/F0T.npz` (25,080) and CN_all equals `F0.npz` (25,005) | S0b |
| G-L | `zpl1m.gates` PASS (disjointness, labels, union) and the counts equal MHR1's `links.json`: W2_S 2, R 25,080, K store 263, RK 25,343; each store's row order equals `FamilyCSR`'s on the same pairs | S0b |
| G-P | The populations: dev 224 and test 1,130 questions, zero-gold excluded counted, every `gold_node_id` resolves, ids unique | S0e |
| G-R | The record's carried constants equal the four passage datasets' U2s records' | S0d |
| G-W | 2g's three: (i) the context's types are the null type on every entry and the weights are exactly 1 (the selftest asserts omega, class and pair weights through `zu2m_a` and `zu2p`); (ii) `zu2m_a`'s own self-test (omega 1 equals the carried kernels bit for bit) passes; (iii) the dev pass runs end to end on the null path, and its record says weights "off (2g)" with zero non-null entries | S0e |
| G-N | **Gold-free, deterministic.** The pool arrays of the dev pass are identical (every array, bit for bit) with the plain `rule_batch` and with `make_rule`'s places, on two different shard layouts and 1 and 2 numba threads; no `POOL_GOLD` field is in any pool file; the places files are separate files | S0f |

The record's own check against a filed analogue is G-R and G-K together: MultiHop-RAG has no U2s arrays, so PKO's section 5 (a) (per-question
fields against U2s's) cannot run; the constants it would carry are checked against the four records, and the rule that picks the cell against
the six's own filed numbers.

## 5. What S0 does not do

- No GPU, no host job unless a step is too big for `rx exec` (the dataset is 609 nodes: expected laptop-only, one core, bounded); no edit of any
  pinned module, of the feed, `paper/` or the progress log; no new rule, constant or cut; no type fit (2g); no ablation, no grade, no call.
- No carried forms: they arrive with UL2's MultiHop-RAG call. The layout and the pools are then rebuilt once, as a new declared block, and the
  construction is not re-read unless the user says so.
- No U2s stats stage and no section 5 (a) check on MultiHop-RAG (4 says why); no pick comparison.
- No pool or place number is read before the gates above pass for its step; a gate that fails is reported, not worked around.

## 6. Points raised for the user (S0 does not decide them)

1. **The knee is read on the e3v lists** (the arm's own first stage: the pools are e3v pools); the six's were read on the frozen-encoder lists on
   13 September. The frozen read is filed beside (`s0/knee_frozen.json`); if the two cells differ the user chooses, and nothing is rebuilt
   without the word.
2. MultiHop-RAG's read is on **its own B0 stores** (W2_S, F0T, K_e3v after dedup), where the six were read on the 13 September packages. The
   graph here is almost all name co-mention (W2_S holds 2 pairs, S none).
3. `zpl1g` expects `outputs/enc/e0/gold/multihoprag.npz` and `outputs/enc/e0/e3v/multihoprag.npz`. S0a writes the gold table in E0's shape under
   `s0/`; E0's own folder is not written by S0.
4. The per-dataset "own" base size is non-universal in pick O (already flagged by the parent); S0 inherits it.
