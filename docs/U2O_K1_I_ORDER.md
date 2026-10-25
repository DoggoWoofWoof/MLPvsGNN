# U2o: U2e's pool rule with K1 as the I's row order

Declared 25 October 2026, before any code or number. It is the follow-up that U2k's filing fixes
([U2K_LEARNED_LINK_SCORER.md](U2K_LEARNED_LINK_SCORER.md), section 10). It replaces U2j's R4 follow-up; R4 is
not run here. Code: `outputs/mp_unified/zu2o.py`, which imports zu2m.py, zu2e.py, zu2k.py, zu2j.py and zu2i.py
unchanged. Outputs go only under `outputs/u2o/`. Pools only: nothing trains here. The scorers are U2k's filed
ones, unchanged, and no row order reads a question, a gold or the target dataset's hyperlinks.

## 1. The question

U2k found that one learned link score, fitted on one Wikipedia corpus's hyperlinks and applied zero-shot to the
others, orders W2_S's rows so that the frozen construction's expansion (the I) holds more golds:
- 2wiki: 0.8369 → 0.8670;
- hotpotqa: 0.8985 → 0.9012;
- musique: 0.7751 → 0.7816.

U2o asks whether that I raises the pools. U2e's carried pick is `own/kind_t/kt_cos:ada_f12_k0@0.85` at level 0.955
and 1,812.6 rows, with a cap of 1,953.5 (U2m, sections 12 and 15). U2o rebuilds U2e's whole candidate grid on the
I read through K1 and chooses again by the same rule.

## 2. What changes, and what does not

- **The graph is W2_S on every passage dataset**, the graph U2m swaps in (`zu2c.graph_csr(D, "chosen")`). Its entries
  are unchanged. Only the order of each row changes.
- **K1's row order** is U2k's section 4. Each row's entries are sorted by s(u, v), the higher of the pair's directed
  scores, highest first; sibling-only entries go last; ties go to the node id.
  - Every row is reordered, on every passage dataset whose construction reads links. U2k reordered only the rows its
    grading read on the one-hop datasets; a pool rule reads no gold, so here all rows are reordered.
  - The scorer is the zero-shot one, as U2k grades: A (fitted on 2wiki) for hotpotqa and musique, and H (fitted on
    hotpotqa) for 2wiki.
- **Only the construction's expansion reads K1's rows.** That expansion is `m3b_compile.prepare`'s pools, the I. The
  walks (A1, K, KT, R0), N1, U2e's second link, B, the seeds and the cosines all read W2_S in node-id order,
  exactly as U2m's filed run read them.
- **So per question, the I can change, and through it:**
  - the fill, KIND_T's first max(0, |I| − |B|) nodes outside B, which U2e's adaptive budgets read;
  - the `cos`, `kt_cos` and `kt_cos_r0` orders, which draw from a union that holds the I's nodes outside B
    (`zu2e.question_orders`' `iout`).
  - B, the seeds, KIND_T, R0, N1, the second link and the `kt` and `kt_n1` orders do not change.
- **The datasets that do not change:**
  - squad's construction is RETRIEVAL and reads no links, so its I is unchanged;
  - metaqa and webqsp are KB graphs and U2k does not touch them.
  - Their candidates are U2m's filed arrays, as filed.

## 3. The candidates and the choice

- **The candidates:** U2e's grid on the own construction with recipe `kind_t`, that is, U2e's 14 orders × its
  budgets × its τ, in U2m's candidate order (`zu2m.index_grid`). Each candidate's ALL and rows come from
  `zu2m.table_arrays`, unchanged. The arrays are U2m's, with the own cell's `kind_t` columns (`fill`, `lens`, `pos`)
  replaced by those of the K1 I on musique, hotpotqa and 2wiki.
- **The choice** is U2m's section 7 rule, with U2e's cost (P1f's weights), the levels, the cap of 1,953.5 and the
  targets min(t, C_d − 0.005) with (c) and (e)'s filed ceilings C_d. So a level means what it meant in U2e's
  verdict.

| verdict | when |
| --- | --- |
| K1_RAISED | the pick's level is above 0.955 |
| K1_CHEAPER | the pick's level is 0.955 and its cost is below 1,812.6 |
| NO_GAIN | otherwise: U2e's pick stays |

**Carried forward.**
- When RAISED or CHEAPER, K1 becomes the I's row order and the pick is the pools' rule. U2m's (d) and (a) arrays,
  which read the carried pick, are then computed again on it for U2n, each in its own file.
- When NO_GAIN, the pools stay as they are.
- Either way, s(u, v) remains an input for both models (U2k, section 8).

## 4. Checks

Any failure stops the dataset, and nothing is written.
- **The graph.** The W2_S that K1 reorders is the file U2m's runs read: its sha256 equals the graph_detail's
  `npz_sha256`, and the choice is W2_S.
  - Every K1 row holds exactly the same entries as the W2_S row.
  - With every score tied, the reorder returns W2_S's columns unchanged.
  - The scorer files are U2k's filed ones: zu2o.py pins their sha256 as committed in 2ee7446 and refuses on a
    mismatch.
- **Only the I changes.** Per question, the stats stage rebuilds the I with the construction's own expansion
  (`m3b_pools.expand_hops`, then `m3b_compile.build_pool`), twice: on W2_S's order and on K1's.
  - On W2_S's order that I equals `prepare`'s pools.
  - The own cell's `kind_t` columns computed from it (`fill`, `lens`, `pos`) equal U2m's filed ones, question for
    question. So at R0 the run reproduces U2m's filed run.
  - On K1's order, the `kt` and `kt_n1` columns are equal to those on W2_S's order.
- **The table.** On the unchanged datasets, and at R0 on every dataset, the table equals U2m's filed table for all of
  U2e's candidates.
- **Self-test** (`--selftest`), before the commit that adds the code:
  - the vectorized reorder against a per-row Python sort on toy graphs, siblings and ties included;
  - the column splice: replacing the own cell's `kind_t` columns leaves every other column equal.
- **Development runs** (`--limit`) write nothing and log no number.

## 5. Beside, deciding nothing

- Per dataset: the I's ALL and mean size on W2_S's order and on K1's, with U2k's filed I's ALL beside them. Both
  are the frozen construction on s1sel, so they are expected to be equal; a difference is reported, not a stop.
- Per dataset: the pick's ALL and rows at U2e's point under K1.
- The pick's misses by U2m's causes (TAU_CUT, BUDGET, UNREACHED), against U2e's pick.
- webqsp's s1eval, unchanged, as everywhere.

## 6. Where it runs

- **`graph D`** builds K1's columns on the laptop, for musique, hotpotqa and 2wiki (U2k's grading built the same
  scores there). It writes `outputs/u2o/graphs/<D>_K1.npz` and `.json` (the array stays uncommitted).
- **`stats D`** recomputes the walks, because the orders that read the I also read KIND_T's order, which U2m did not
  store. Its cost is that of U2m's stats on one cell and one recipe.
  - musique runs on the laptop;
  - hotpotqa and 2wiki run on Modal (U2m's image and volumes, a new output directory named by zu2o.py's sha256,
    the K1 array uploaded as a new file) if the remaining credit covers them, or on the lab host once it is
    reachable. Otherwise they are held, and with them the verdict.
- **`verdict`** runs on the laptop and writes `outputs/u2o/verdict.json` and `.md`.

The four passage corpora are all Wikipedia, so this measures transfer between Wikipedia corpora only (U2k's caveat).

A verdict closes only the form tested (docs/SCREENS.md, section 3b): here K1's order as the I's row order, with U2e's
grid. K1 as the walks' row order is a different form, and so is a prune that reads its cut from each graph. Neither
is closed.
