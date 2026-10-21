# U2c: today's pool rules on U2b's chosen graph, cut to fewer rows, with every miss by cause

Declared 22 October 2026, before any number on the chosen graph and before U2b's choice. The user's direction of 22
October: find improved links, improve all the datasets, then improve and compress the pools and remove the upstream
failures. Code: `outputs/mp_unified/zu2c.py`, importing P1g, P1g2 and P1h unchanged. Pools only: nothing trains,
nothing is fit, and no cut reads a gold.

## 1. Why

U2b (docs/U2B_LINK_RULE.md) picks one LLM-free link rule for the four passage datasets. Today's pools are built on
today's graph: the Wikipedia hyperlinks that hotpotqa and 2wiki ship with, and the package's structural graph on
musique and squad. P1g2's rule `g:f1.5+cut_cos_5000` meets the user's target there (ALL ≥ 0.95 on all five training
datasets' s1sel), at 1,975.5 rows (P1f's cost), against today's pool at 699.7.

U2c rebuilds the same pool rules on the chosen graph and adds P1h's cut to fewer rows. It then chooses one rule with
P1g's own choice. It also lists every gold the chosen pool misses, with its cause. U2b section 7 promised that ledger,
the list of golds no pool built on one link can hold.

## 2. The graph

- **Where it is swapped.** On musique, squad, hotpotqa and 2wiki, the `structural` family is U2b's chosen graph. It is
  read from `outputs/u2b/<D>/graphs/<graph>.npz`, and the file's sha256 must be the one U2b's build record names.
- **Where it is not.** metaqa and webqsp keep their KB, which is their graph under both runs. ner and knn are
  unchanged on every dataset.
- **What reads it.** Everything that reads `structural` reads the swapped graph:
  - the frozen I (`m3b_compile.prepare`, unchanged, given the swapped store);
  - KIND_T's walks, over both the structural-only union and the all-family union;
  - the beams and cuts, over all three families' adjacency.
- **What does not change.** Base and seeds B do not depend on the graph. They are step 4d's, checked through the
  cached graph as every pool stage checks them. The seeds on the swapped graph must equal those.
- **Today's fill on the graph.** It is |I^G_q| − |B_q|, from the frozen I rebuilt on the graph, as step 4h did.
- **How the swap is made.** The store is built from the graph's symmetric CSR. It equals `FamilyStore.from_graph` on
  the same edges, field for field (the self-test checks this). Inside the process, `m3b_pools.load_or_build_store`
  returns that store for `structural` on the dataset; every other family loads from M3B's cache as before.

## 3. The candidates and the cut

- **P1g2's 1,603 rules** at τ = 1: P1g's 805 (seven fill fractions × 115 extra sets), then the 798 gated forms, in
  P1g2's order.
- **Each rule without a beam** (none or a cut: 567 rules) also enters at P1h's nine τ below 1: 0.99, 0.975, 0.95,
  0.9, 0.85, 0.8, 0.7, 0.6 and 0.5. Each comes right after its rule, from 0.99 down. That adds 5,103, for **6,706
  candidates** in all.
- **Why no beam below τ = 1.** A beam has no single order over its nodes, so it enters at τ = 1 only.
- **The cut is P1h's** (docs/P1H_FEWER_ROWS.md, section 2). Each question keeps the shortest prefix of its build order
  whose rank mass, the sum of 1/(60 + p), reaches τ of the order's total. It never keeps fewer than |B| and never more
  than the pool.
- **The build order:**
  1. B;
  2. KIND_T's first n_f nodes outside B;
  3. the extra set's nodes outside those, by their rank under the extra's own order.

  The cut reads two sizes per question and no gold.

**What a pass stores.** Per question and per P1g candidate, the rows. Per gold, its place in that candidate's build
order: 0 in B, p the p-th expansion node, −1 outside. A gold that only a beam adds is stored as held, at no place. The
gated forms and every τ follow from these arrays, as P1g2 does.

## 4. The choice

The choice is P1g's rule (docs/P1G_POOL_BEAM.md, section 5), on the five training datasets' s1sel.

- **Ceiling.** C_d is the best ALL any candidate reaches on dataset d over the graph.
- **Target.** min(0.95, C_d − 0.005) on every dataset.
- **Cost.** P1f's question-weighted mean rows. The weights are metaqa 15,745, squad 17,729, musique 7,018, hotpotqa
  13,335 and 2wiki 18,504.
- **Pick.** The cheapest candidate that meets every target. Ties go to the earlier candidate.
- **Verdict.** TARGET_MET when every target is 0.95, CEILING_MET otherwise. NONE_MEETS when no candidate meets them
  all; the smallest worst gap is then reported.

webqsp's s1eval is graded beside and never moves the choice. The same choice also runs on today's graph, beside the
first. That shows what the hyperlinks are worth on hotpotqa and 2wiki, against the same cut. Its τ = 1 part is P1g2's
filed choice.

## 5. The ledger

For the chosen pool on the chosen graph, every s1sel gold it misses is listed (webqsp's s1eval beside). Each miss gets
one cause, the first that applies:

- **TAU_CUT:** the same rule at τ = 1 holds it, so the cut removed it.
- **BUDGET:** some candidate on the graph holds it, so a larger pool reaches it.
- **UNREACHED:** no candidate on the graph holds it.

The UNREACHED golds are counted again by five properties:
- whether the first stage's 1,000-deep lists hold the gold;
- whether it is one link from B in any family;
- whether KIND_T's order holds it;
- whether the frozen I holds it;
- whether today's graph reaches it with some candidate (a gold only the hyperlinks reach).

The questions are counted by cause too.

## 6. Checks

Any failure stops the item, and nothing is written.

**Today's graph, every question:**
- P1g's filed rows and held golds, for all 805 candidates;
- P1g's fill;
- P1f's |B|, |I_q| and depths.

On the four passage datasets, today's graph is read back from U2b's `pkg_structural.npz` and swapped in, exactly as
the chosen graph is. It must rebuild the cached graph's frozen I question for question. So these runs check the swap
itself as well as the code.

**The chosen graph, every question:**
- base and seeds are step 4d's;
- the seeds equal the cached graph's;
- |B| and the golds' layout are P1f's.

**Both graphs:** every candidate's build order agrees with P1g's rows and held golds for that question.

**Self-test** (`python outputs/mp_unified/zu2c.py --selftest`, passed before this commit):
- the CSR store equals `from_graph` field for field;
- the extra sets equal P1g's, set for set, on toy graphs;
- every candidate's build-order places equal a direct build;
- the gate, the cut and the summary equal a direct count;
- the ledger's causes add up to the misses.

**Development runs** (`--limit`, before any full run) write nothing and log no number.

## 7. Where it runs, and the outputs

The laptop holds a copy of the frozen package with the same freeze record. It runs at 2 threads, P1g's thread count,
so the cosine sums match P1g's filed arrays; P1h's development runs reproduced them there. Memory is that of P1g's
stats items: metaqa 3 GB, squad 2, musique 3, hotpotqa 6 and 2wiki 8. If the host is back first, the same commands
run there. Modal is not used, because the stage reads the whole frozen package.

    python outputs/mp_unified/zu2c.py stats DATASET --graph today [--limit Q]
    python outputs/mp_unified/zu2c.py stats DATASET --graph chosen [--limit Q]    (musique, squad, hotpotqa, 2wiki)
    python outputs/mp_unified/zu2c.py verdict

The chosen graph's runs need U2b's choice and the graph's npz, fetched from the U2b volume. The `today` runs do not.

Outputs:
- `outputs/u2c/stats/<graph>/<D>__<carve>.npz` and `.json`;
- `outputs/u2c/verdict.json` and `verdict.md`.

The arrays stay uncommitted.

**Follow-up (22 October, outputs unchanged).** The user asked for Modal on 22 October ("use modal"), so the stats
stage runs there in question shards instead of 10 to 14 hours on the laptop.
- `zu2c.py stats D --graph G --shard I/K` runs every check on the I-th of K contiguous question ranges and writes a
  part under `outputs/u2c/stats/<graph>/parts/`.
- `zu2c.py merge D --graph G --parts K` checks each part (script, file sha256, checks, its range) and joins the parts
  into the file one run over every question writes.

Each question's sources, walks, orders and sets are its own: a batch only spreads questions over threads. So a shard's
questions get the values one run gives them. On the laptop, musique's shard 1 of 3 (questions 10 to 19 of a 30-question
development run) reproduced P1g's filed arrays exactly. The self-test now covers the ranges and the merge.

`outputs/mp_unified/zu2c_modal.py` runs the shards and the merges:

    modal run --detach outputs/mp_unified/zu2c_modal.py --stage stats --datasets D1,D2,... --graph today|chosen --shards K

Each container builds the package mirror from symlinks into two volumes. Every file is the laptop's, byte for byte:
- mpr-u2-links, U2b's volume, is only read;
- mpr-u2c is new, for this stage.

The freeze check still runs, on the mirror's `CANONICAL_FREEZE.json` (58958f33...). The container pins:
- the laptop's package versions;
- P1g's 2 threads;
- OpenBLAS's Haswell kernels;
- numpy's AVX-512 paths off;
- numba compiling for Haswell.

So every container CPU takes the laptop's paths. The C library's maths functions are Linux's, not Windows'. The today
runs check the platform as well as the code: every question must reproduce P1g's filed arrays. A failed check stops
that dataset's merge and is not retried. The outputs are written on the volume under a directory named by zu2c.py's
sha256, and fetched to the paths above.

## 8. What follows (fixed now)

| outcome | the next form |
| --- | --- |
| TARGET_MET | The chosen rule on the chosen graph becomes the candidate pool for the refits (U2d, declared on its own). U2d builds it on the fit and s1eval carves of all six datasets, through this code. s1eval is graded beside. Then it builds the KT input on those pools and refits the lean pair (zrckp, zspk) and the free GNN (G6). Each model adopts the pools only through its own declared screen. |
| CEILING_MET | As TARGET_MET. The datasets whose ceiling sits below 0.955 take U2B section 7's next forms, in order: a second link from the question's best X rows, then the question-guided hop. They are aimed at the ledger's UNREACHED golds, and each is declared first. |
| NONE_MEETS | No pool goes forward. U2B section 7's next forms come first. |

**The headline graph.** docs/PROGRAM_2026_10_21.md, section 3, asks U2c to fix this rule before any number. The
chosen graph holds hotpotqa and 2wiki when its chosen pool keeps ALL ≥ 0.95 on both s1sel carves: a target of 0.95
met, not a lower ceiling. If it holds, the LLM-free graph is the headline graph on all four passage datasets, and the
hyperlinks are reported beside only. If it does not, the hyperlinks stay named as the headline graph on each dataset
it fails. The link work then stays open (section 5 of the same file), with U2B section 7's next forms.

A verdict closes only the form tested (docs/SCREENS.md, section 3b). P1h stays declared as it is, on P1g-b's pools.
