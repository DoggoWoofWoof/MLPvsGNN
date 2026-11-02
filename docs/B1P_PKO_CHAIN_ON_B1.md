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
- **Fix after the first host look (fce3654, no number read):** the host's musique b1w look stopped at 18:16 on metaqa's
  stores. The six pair's look opens all six datasets (`m3b_run.open_contexts` through `universal_v2_six.open_six`). Inside
  `on_b1`'s look, the other five datasets' stores now come from the original loader, as under `look_b1` for B1f; only the
  setting's dataset gets the setting's stores. The pools pass still refuses other datasets. Two checks followed:
  - The laptop's 8-question look on the host's filed musique b1w pools held every `look_u2d` check. Its structural store
    held 82,834 entries, the cut's 41,417 pairs twice.
  - 2wiki's b1w look on the host ran on the fixed code (1,000 questions, 94 s), and its cache checked IDENTICAL against Carve3.

## 12. Results (host, 2 November; filed 3 November 2026)

`zb1p.py report` wrote `outputs/b1p/report.json` and `.md` on the host at 18:14 UTC on 2 November (zb1p.py sha256
9bce0abf...). Nothing in sections 1 to 9 changed after the reads.

- **Complete.** 72 reads: 12 fits per variant (two arms, three seeds, two splits), each read on the three settings. Missing: 0.
- **No training overlap.** 0 of each setting's 1,000 questions are in the in-domain splits' training carves (PKO's KT ids), as
  expected.
- **Pools** (`outputs/b1p/steps/`), rows per question on b1w and b1t: musique 3,433.6 and 3,313.0, 2wiki 88.5 and 50.0, hotpotqa
  65.3 and 50.0. No record step found a problem.

**In-domain reads.** R@5 x100 of p@swa, seed means (2wiki: over both splits), 95% intervals over the questions. FC@5 and hit@1 are
fit means.

| setting | arm | b1t | b1w (CI) | w - t (CI) | b1w FC@5 | b1w hit@1 | calls (any / peer) |
|---|---|---:|---:|---|---:|---:|---|
| musique | GNN zg6r | 66.0 | 69.0 (67.2-70.7) | +3.0 (+1.8, +4.0) | 39.6 | 80.4 | BELOW / BELOW |
| musique | MLP zrckp | 62.5 | 66.9 (65.2-68.6) | +4.4 (+3.3, +5.6) | 36.8 | 79.8 | BELOW / BELOW |
| 2wiki | GNN zg6r | 78.1 | 94.9 (94.1-95.7) | +16.8 (+15.5, +18.2) | 87.2 | 99.3 | BELOW / BELOW |
| 2wiki | MLP zrckp | 75.9 | 94.0 (93.1-94.8) | +18.1 (+16.8, +19.4) | 84.8 | 98.9 | BELOW / BELOW |
| hotpotqa | GNN zg6r | 91.1 | 94.3 (93.3-95.3) | +3.1 (+2.4, +4.0) | 89.0 | 91.2 | BELOW / BELOW |
| hotpotqa | MLP zrckp | 90.0 | 93.4 (92.4-94.5) | +3.4 (+2.5, +4.3) | 87.3 | 90.5 | BELOW / BELOW |

The best published rows (section 5): musique SAG2 80.68 / PropRAG 78.3; 2wiki EviReform 98.45 / G-reasoner 98.2; hotpotqa
BridgeRAG 98.75 / PropRAG 97.4.

**Zero-shot reads, beside:**

| setting (split) | arm | b1t | b1w (CI) | w - t (CI) |
|---|---|---:|---:|---|
| musique (WQ-musique) | GNN zg6r | 62.1 | 60.8 (59.1-62.5) | -1.3 (-2.6, +0.1) |
| musique (WQ-musique) | MLP zrckp | 58.6 | 60.9 (59.2-62.6) | +2.3 (+0.9, +3.5) |
| hotpotqa (WQ-hotpotqa) | GNN zg6r | 89.4 | 92.9 (91.9-94.0) | +3.5 (+2.6, +4.4) |
| hotpotqa (WQ-hotpotqa) | MLP zrckp | 87.9 | 92.2 (91.2-93.3) | +4.3 (+3.4, +5.2) |

**B1f's reads beside** (B1b's fits on split J5, trained on the package graphs with the hyperlinks; GNN zsp / MLP zrc):

| variant | musique | 2wiki | hotpotqa |
|---|---|---|---|
| b1 (hyperlinks; musique: title links) | 66.3 / 66.1 | 97.2 / 96.4 | 98.2 / 98.2 |
| b1u (plain title rule) | 66.3 / 66.1 | 95.9 / 95.3 | 95.0 / 94.8 |
| b1w (W2_S) | 65.7 / 65.3 | 95.2 / 94.2 | 94.3 / 94.8 |
| b1t (text links only) | 62.9 / 63.4 | 77.3 / 75.6 | 90.0 / 88.7 |

**What the reads show (deciding nothing):**

1. **The rule's links carry over in-domain.** b1w - b1t is above zero, with its interval, in all six in-domain reads. That is
   +3.0 (GNN) and +4.4 (MLP) on musique, +16.8 and +18.1 on 2wiki, and +3.1 and +3.4 on hotpotqa. Zero-shot, three of four reads
   are above zero. The GNN on musique under WQ-musique is -1.3 (-2.6, +0.1).
2. **musique.** The GNN with no hyperlinks and no LLM reads 69.0 on musique's setting. B1b's J5 fits on b1 read 66.3 (GNN) and
   66.1 (MLP); musique has no hyperlinks, so its b1 is the title links. The MLP reads 66.9. These rows come from different models
   and splits (J5 against WQ-hotpotqa), and B1P compares them by no rule.
3. **2wiki and hotpotqa.** The GNN reads 94.9 on 2wiki's b1w, against B1b's GNN at 97.2 on the hyperlinks. On hotpotqa it reads
   94.3, against 98.2.
4. **Table 1.** All six in-domain b1w reads are BELOW both best published rows:
   - musique, against PropRAG's 78.3: by 9.3 (GNN) and 11.4 (MLP) points;
   - 2wiki, against G-reasoner's 98.2: by 3.3 and 4.2;
   - hotpotqa, against PropRAG's 97.4: by 3.1 and 4.0.
5. **The MLP beside the GNN** on b1w in-domain: 66.9 against 69.0 (musique), 94.0 against 94.9 (2wiki), 93.4 against 94.3
   (hotpotqa). The MLP gains more from the rule's links than the GNN on all three settings.

**What follows.** B1P decides nothing (section 1).
- Its in-domain b1w rows are the LLM-free, hyperlink-free Table 1 rows of ML1's models of record.
- Its b1w - b1t rows measure the rule's carry-over on settings it was never fitted on.
- A model that replaces zg6r or zrckp before the 6-7 November freeze is read on B1 with this chain (`zb1p.py`), in a stage of
  its own.
