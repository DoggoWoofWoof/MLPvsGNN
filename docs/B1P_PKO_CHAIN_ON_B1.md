# B1P: PKO's pools chain on HippoRAG 2's settings, read by ML1's GNN and MLP (no LLM, no hyperlinks)

Declared 2 November 2026, before any B1P read. Follows [B1F_W2S_ON_B1.md](B1F_W2S_ON_B1.md) (b1w: U2b's W2_S rule cut to B1's
settings), [ML1_REFITS_ON_PKO.md](ML1_REFITS_ON_PKO.md) (the GNN zg6r and the MLP zrckp of record, fitted on PKO's pools) and
[PKO_PICK_O_POOLS_UNDER_WQ.md](PKO_PICK_O_POOLS_UNDER_WQ.md) with its amendments A1 to A3 (the pools rule and its chain).
Code: `outputs/mp_unified/zb1p.py` (new). Outputs only under `outputs/b1p/`.

Nothing trains. No filed graph, pool, fit or record changes. B1 is never used for selection: B1P decides nothing about any model,
pool or rule. It is a measurement.

## 1. Why

The paper's Table 1 rows are B1b's fits (zsp, split J5), trained on the package graphs, read on b1 (the corpus hyperlinks). The
models of record are now ML1's zg6r (the GNN) and zrckp (the MLP). Both were trained on PKO's pools, which are built over U2b's W2_S
rule graph, with no LLM and no hyperlinks. They have never been read on B1. The user's priorities of 2 November are musique as the
main target, hotpotqa and 2wiki above their best published rows, and the in-domain reads first. The headline the user asks for is
that a link rule recreates the hyperlinks' effect and carries over to other datasets. B1P measures three things:

1. Table 1 with the LLM-free, hyperlink-free pair.
2. The rule's carry-over: the paired difference b1w - b1t, the same fits read with and without the rule's links.
3. Each in-domain read against the best published row of its setting.

Unlike B1f, reading these fits on b1w is no shift of graph family: they were trained on W2_S graphs of the full corpora, and b1w is
the same rule cut to the setting.

## 2. The carves

| carve | setting files | the structural graph PKO's rule walks (`zu2c.graph_csr`, "chosen") |
|---|---|---|
| **b1w** | B1f's `outputs/bench/hipporag2_w/<dataset>` | U2b's chosen graph (W2_S) of the full corpus, cut to the setting's passages (both ends inside) |
| **b1t** | U1a's `outputs/bench/hipporag2_t/<dataset>` | none (an empty graph: the setting's text links only) |

- **The cut is B1f's b1w graph exactly.** `zb1p.py settings` checks that the cut's undirected pairs equal the variant's structural
  graph (B1f's `graph_structural.npz`), with no self-loops or repeats: musique 41,417 pairs, 2wiki 2,030, hotpotqa 4,315; b1t 0 on all
  three. It ran before this file on the laptop, read no model and no question, and runs again on the host.
- **The rest is the setting's own:**
  - the first stage: dense lists over the setting's passages, and SPLADE lists from the package's vectors where the machine has
    them, otherwise B1a's `b1_lists.npz`; each pools record names which;
  - the NER and kNN families (`look_b1`'s setting stores);
  - B1a's 1,000 questions per setting (`look_x_six.carve_population`);
  - the name index, built from the setting's passage names only.
- **b1 (the hyperlinks) is not read.** The fits are hyperlink-free; the paper's hyperlink row stays B1b's and B1f's.

## 3. What runs (`zb1p.py`)

| step | what |
|---|---|
| `settings` | The cut graph and the type arrays restricted to its entries, in its entry order: U2m (a)'s EM types and U2p's score types, each the full corpus's per-entry class. Checked against B1f's graph as in section 2. |
| `pools` | PKO's pools pass (`zpko.pools_pass` with PKO's section 2 rule, unchanged) in one shard, inside `on_b1`. `on_b1` opens the package's dataset as the setting in process and restores every patch on exit. U2s's record is re-targeted to the cut graph's description (`graph_detail`); every other field of it must still match as PKO checks it (cells, own cell and base, walk, alpha, eps, nmax, the type files' shas). Every B, I and expansion row is checked to lie inside the setting. |
| `chain` | Each step in its own process, a step already filed skipped: merge; look shards (`look_u2d`, with B1's population, first stage and stores); `lean_cache`'s bases and parts; links (`zlink`, on the look's edges); KT; the carve's record (no step-1 cache: not applicable). Every graph the cache, links and reads see comes from the look's chunks, so b1t's chain never sees a W2_S edge. |
| `gate` | `zpko.gate` on the six carves, with PKO's own section 5 checks (`outputs/pko/check`, PASS on the host): `outputs/b1p/gate.json`, READY or NOT_READY. |
| `fork` | ML1's twelve fits (zg6r and zrckp, seeds 0 to 2, WQ-musique and WQ-hotpotqa) copied to `outputs/b1p/fits/<w|t>/<fit>` without their reads; `models.pt` sha-checked. |
| `read` | Each fork read on `musique=c,2wiki=c,hotpotqa=c` (c = b1w or b1t), with zml1s's read code: `run_arm` for zg6r, the body of `zml1s.read` for zrckp, ML1's rows caps, and ML1's device check (the host's RTX 4500 Ada, torch 2.8.0+cu128, `CUBLAS_WORKSPACE_CONFIG=:4096:8`). The read runs on B1P's roots and gate. No ML1 folder is written: no `zml1s.stamp`, `zml1s.read` or `zml1s.fit` is called. |
| `report` | Section 4's table: `outputs/b1p/report.json` and `report.md`. |

## 4. The reads and the report (fixed now)

- **Metrics.** B1b's candidate p@swa: R@5 per question, with FC@5 and hit@1 beside. 95% bootstrap intervals over the setting's
  1,000 questions (B1b's: 2,000 resamples, seed 0).
- **In-domain read of a setting:** the fits whose split trained on the setting's dataset.
  - musique: WQ-hotpotqa (trains musique's s1fit);
  - hotpotqa: WQ-musique (trains hotpotqa's fit);
  - 2wiki: both splits.

  Per arm, each question's R@5 is averaged over the three seeds (on 2wiki, over the six fits of both splits), and the interval is
  taken over the questions.
- **Zero-shot reads**, beside: musique by WQ-musique, hotpotqa by WQ-hotpotqa.
- **Training overlap.** B1's questions are inside each dataset's s1eval (the 14 October finding), so no fit trained on them. The
  report checks each setting's question ids (B1P's KT `ids.json`, the package's ids) against the in-domain training carves' KT ids
  (PKO's). Expected: 0 everywhere.

## 5. Reported, deciding nothing

- **Table 1 row.** Per setting and arm, the in-domain b1w R@5 against the best published row of the setting (`b1f.PUBLISHED`), of any
  kind and peer-reviewed only: ABOVE, AT or BELOW by the interval (`b1b.call`). The best rows today:

  | setting | any kind | peer-reviewed |
  |---|---|---|
  | 2wiki | EviReform 98.45 | G-reasoner 98.2 |
  | hotpotqa | BridgeRAG 98.75 | PropRAG 97.4 |
  | musique | SAG2 80.68 | PropRAG 78.3 |
- **The rule's carry-over.** Per arm and setting, the paired b1w - b1t (in-domain and zero-shot), with its interval: the gain the link
  rule brings on settings it was never fitted on.
- **Beside:** B1f's reads of the hyperlink-trained fits (b1, b1u, b1w, b1t), so the hyperlink row and the earlier rule rows sit next
  to B1P's.

## 6. Deviations, stated before any number

1. **Name links** are built from the setting's passage names only (names outside the setting are blanked).
2. **PKO's thresholds** (pick O's budget, tau, ADA's k) were fixed on the open corpora. They are used unchanged on settings of 6,119
   to 11,654 passages.
3. **The W2_S cut and its types** inherit the full corpora's fits: W2_S's gate and title forms, and each entry's EM and score class.
4. **U2s's checks are re-targeted**: the record's `graph_detail` is the setting's description; the rest must match unchanged.
5. **The chain's records keep PKO's declaration** (`zpko.DECLARED`), because ML1's read checks for PKO's READY gate.
6. **The first stage** may come from B1a's lists (section 2), as B1a, B1f and `look_b1` do; the pools records name the source.

## 7. Risks

- **Pool composition differs.** The fits were trained on pools over the full corpora: 1 to 12 million passages and many
  distractors. A setting holds 6,119 to 11,654 passages, so the pools hold other rows than s1eval's. That is what B1 measures; it
  may not track the s1eval reads.
- **musique's cut has a hub**: degree 1,467 in B1f's cut.
- **b1t is a shift**: the models trained with a structural family see none on b1t. b1w - b1t is read from the same fits, so it is
  the rule's links as these models use them, not a model trained without links.
- **The reads wait on the GPU.** They are queued beside LTG's first fit, after ISO's last hp fit. CPU: three chains at a time
  (each b1t chain after its b1w chain), about 1.1 hours each.

## 8. Commit policy

Commit only `outputs/b1p/report.json`, `report.md` and `b1p_gate.json`, masked with `mask_host.py`, with `git add -f`. Never commit
the settings arrays, pools, looks, caches, links, KT, forks or reads.

## 9. Files

    outputs/mp_unified/zb1p.py                        the stage (new)
    outputs/b1p/settings/<w|t>/<dataset>.{npz,json}   the cut graphs, maps and restricted-type records
    outputs/b1p/{pools,look,cache,links,kt,records}/  PKO's chain on the six carves
    outputs/b1p/steps/                                 B1P's step records (first stage, pool sizes, seconds)
    outputs/b1p/gate.json, b1p_gate.json               the gate
    outputs/b1p/fits/<w|t>/<fit>/                      the forks and their reads
    outputs/b1p/report.json, report.md                 the table

## 10. Development runs before this file (disclosed; no model read)

- `zb1p.py settings` on the laptop: the pair counts of section 2, and each restricted type array checked against its record.
- `zb1p.py pools --dataset 2wiki --variant w|t --limit 20` on the laptop: the first 20 questions of each carve, nothing written.
  It checks that the pass opens and that every row stays inside the setting. Its seconds and prepared rows are in section 11.

## 11. Development numbers (laptop, no model read)

`zb1p.py pools --dataset 2wiki --variant w|t --limit 20`, two threads:

| carve | questions | pass | per question | open | prepared rows (1,000 questions) | structural entries |
|---|---|---|---|---|---|---|
| b1w | 20 | 35.9 s | 1.80 s | 211.6 s | 53,936 (53.9 per question) | 4,060 |
| b1t | 20 | 34.9 s | 1.74 s | 181.7 s | 50,022 (50.0 per question) | 0 |

- **First stage:** both passes computed the dense lists over the setting's 6,119 passages. `look_b1` counted 39 dense lists with
  near-tie swaps and none with repeats dropped, and no SPLADE list with either.
- **Inside check:** every B, I and expansion row of the 40 questions lay inside the setting. The check raises otherwise.
- **Not shown:** a limited pass writes nothing and prints no pool sizes. The host's full passes record them in
  `outputs/b1p/steps/`.
- **Time:** at these rates a full carve's pass takes about 30 minutes on the laptop.
- **Fix before queueing:** `clean_look` now also removes `look_x_six`'s temporaries (`c#####.tmp.npz`). Before, it would have
  stopped on their names. The self-test covers it.
