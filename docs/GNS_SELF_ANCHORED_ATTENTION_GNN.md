# GNS: GNT's learned attention with its places the GNN's own first-pass top five, in the GNN of record's encoder, on PKO's pools

Declared 3 November 2026. This is before GNT's and GNR's verdicts, and before any GNS column or fit. GNT's and GNR's
seed-0 fits have not started. GNR's WQ-musique first pass is done and passed its check.

- **Stage modules:** `outputs/mp_unified/zgns.py` (the arm) and `outputs/mp_unified/gnscols.py` (the columns).
- **Imports, none of them edited:** `zgnt.py`, `zgnr.py`, `gnrcols.py`, `zgnw.py`, `gnwcols.py`, `gnpcols.py`,
  `zpfp.py`, `zgnx.py`, `zgnp.py` and `s0stop.py`.
- **Columns:** GNS builds its own, per split and seed, under `outputs/gns/s<seed>/cols/`. GNR's first passes, GNW's
  columns and their records are read only.
- **Stack:** GNS is not a member of the stack (docs/STK_GNN_STACK.md). The stack takes it only by an amendment.

## 1. The question

- **Why now.** GNT's section 7 names one next form after a NO_GAIN, NOT_ADOPTED or ADOPT_FOR_FULL_RUN verdict: the
  same attention to the GNN's own top five after a first pass (GNR's first pass). GNS is that form.
  - It is its own arm against the same base as GNT (ML1's zg6r fits, ISO KEEP), so GNT's verdict changes none of
    GNS's reads.
  - GNR's WQ-musique first pass is done, so GNS's WQ-musique columns can be built now on the host's idle CPUs. Its
    seed-0 fit can then follow GNT's on the card's 0.42 half on 3 November. Waiting on GNT's verdict would hold it until
    4 November.
- **How GNS differs from GNT: only the places.**
  - GNT's places are rrf's top five rows of each question.
  - GNS's places are the base GNN's own five highest rows of each question in a first pass on the same pool: GNR's
    first pass, which is ML1's zg6r fit of the split and seed, read p@swa.
  - The term, its parameters, its start and everything else are GNT's.
- **Why the places can matter.**
  - A row that completes a chain gains from the attention only through the rows it attends to.
  - The base ranks its own five from the GNN's full states, the edges and the walk among them. rrf ranks its five from
    the first stage alone.
  - Section 6 reports, per carve, how many golds each of the two fives holds.
- **One round of refinement.** The first pass's ranking picks the places, and the second pass refines each row's l1
  state against them. Section 7 names a second round under ADOPT.
- **The GNN stays its own model.**
  - The places come from the base GNN's own read. The columns are made from them by GNW's rule, with no fit, before any
    of GNS's rows is scored.
  - No MLP and no MLP fit enter a GNS fit.
  - The attention passes the places' states, so it is message passing, and GNS has no MLP form.
- **So the question is:** does learned attention to the GNN's own first-pass top five raise the GNN's in-domain R@5
  over zg6r's, above all on musique?

## 2. What runs

### 2.1 The first pass (GNR's, read only)

- GNR's first pass (docs/GNR_REANCHORED_NEXT_GNN.md section 2.1) is the base of the split and seed, read p@swa on every
  carve a fit of the split loads. Each question's five highest rows are taken in `lean_gpu.top_hit`'s order (descending
  score, ties by pool position).
- Its check holds the s1eval reads to the base's own filed reads: within 0.002 in R@5.
- GNS builds no first pass. Its column builds and its fits refuse a first pass that `zgnr.first_problems` rejects.
- **WQ-musique's first pass** (`outputs/gnr/s0/first/WQ-musique.json`) passed.
  - Its s1eval R@5 equal the base's: metaqa 0.3615, squad 0.9142, musique 0.5086, hotpotqa 0.8593, 2wiki 0.9269 and
    webqsp 0.6632.
  - The golds among the five equal the base's own read for every question of the six (agreement 1.0).
  - Its training carves are read in sample: R@5 webqsp 0.8190, squad 0.9278, hotpotqa 0.8802, 2wiki 0.9470 (section 8).
- **WQ-hotpotqa's first pass** is GNR's `gnr-first-wqh`, queued.

### 2.2 The columns (gnscols.py `cols`, CPU)

- **The rule.** GNW's rule and layout (docs/GNW_PAIR_WEIGHTS_IN_GNN.md section 2), with the first pass's five rows
  a_1 ... a_5 as the places, in its order. For each row c:
  - pair_k = (c.q + a_k.q) / sqrt(2 + 2 c.a_k) and res_k = c.q - (a_k.q)(c.a_k), from the frozen encoder's unit
    vectors (`ltwdiag.pair_feats`);
  - anchor_k is 1 on a_k's own row and 0 elsewhere.

  Both scores are 0 on a place's own row and for a place the question lacks. A question of five rows or fewer has one
  place per row. Fifteen columns, float16, as GNW stores them.
- **Where.** In ltmcols' layout under `outputs/gns/s<seed>/cols/<split>/<dataset>/<carve>/`, for the ten carves a fit of
  the split loads (`ltmcols.col_carves`): its four training carves and the six s1eval carves.
- **Bound to one first pass.** Every record's fit_sha256 is one sha256 over the rule's code (gnscols.py, gnrcols.py,
  gnwcols.py, gnpcols.py, zpfp.py, ltwdiag.py and chndiag.py) and the first pass's record sha256. Columns built on
  another first pass never load as these.
- **A carve's build stops when:**
  - the carve is not one the split loads;
  - the first pass is missing, did not pass, or changes while the columns are built;
  - the first pass's questions, pool sizes, or golds among its five are not the cache's;
  - a row's c.q is not the cache's dense_cos (zpfp's identity);
  - a question does not hold exactly one row for each place it has.
- **The match** (gnscols.py `match`). With rrf's top five as the places, GNS's code must give gnwcols' own columns
  exactly, on the first 64 questions of PFS's five probe carves. The result goes to `outputs/gns/match.json` with the
  rule's sha256.
- **The check** (zgns.py `check-cols`). For every carve of the split:
  - its record and files load through `zltm.load_next` under GNS's names, sha256 checked;
  - each stands on the current first pass: its fit_sha256, first_sha256 and seed.

  zgnr.first_problems is added. The result goes to `outputs/gns/s<seed>/check_<split>.json`.

### 2.3 The arm (zgns.py)

- **GNT's term, unchanged** (docs/GNT_LEARNED_ATTENTION_GNN.md section 2, `zgnt.on_model`).
  - For row c of question q, alpha_ck is the softmax over q's valid places, other than c's own row, of
    u_c W_a u_{a_k} + wp * pair_ck + wr * res_ck + b_k.
  - m_c is the sum over k of alpha_ck * h[a_k], and l1's output becomes h + W_m m_c.
  - W_m and W_a start at zero, so a fit starts as zg6r bit for bit.
  - The term is 32,775 parameters, 212,993 in all at hidden 128.
- **The places are read from GNS's anchor_k columns,** so the attention's places are the first pass's five rows.
- **The rest is GNX's** (GNX section 2): the base fixed by ISO's filed verdict (KEEP: ML1's zg6r fits), the trainer, the
  card, the epochs, the KT input, the read cap and the p@swa read.
- **The names.** For the length of a GNS call:
  - ltmcols' root (`outputs/gns/s<seed>` of the call's seed), stage, declaration, column names and width are GNS's
    (`zgns.on_cols`), so zgnx's carve loads GNS's columns and refuses LTM's, GNW's and GNR's.
  - zgnx's fits, declaration, stage, stamp key (`gns`), output root (`outputs/gns`), fit and pair names, code, term
    record, NEXT block, IN and WIDTH (15 and 15) are GNS's, and its model patch is GNT's.
  - The NEXT block names this declaration, the seed, the seed's column root, and the first pass's record and sha256.
  - The code adds zgns.py, gnscols.py, zgnr.py, gnrcols.py, zgnt.py, zgnw.py, gnwcols.py, gnpcols.py and zpfp.py to
    GNX's.
  - All of it is put back afterwards, also on a failure.
- **The fits.** The screen is seed 0 on both splits: `scr-gns-sa-s0` (WQ-musique) and `scr-gns-sa-s0-hp`
  (WQ-hotpotqa). The compares are `scr-gns-sa-vs-base-s0` and `-s0-hp`, against the base's fit of the same seed and
  split.
- **A fit refuses to start** (`zgns.gate_problems`) unless:
  - the first pass of its split and seed passes `zgnr.first_problems`;
  - `outputs/gns/match.json` passed under the current rule's sha256;
  - `outputs/gns/s<seed>/check_<split>.json` passed on the current first pass, with its fit_sha256.

## 3. The reads

GNX's section 3, unchanged.

## 4. The rule (fixed now)

GNX's section 4, unchanged: LTG's in-domain rule (`zltg.decide_screen`, `zltg.decide_full`) and GNX's INCOMPLETE
conditions, with zgnw's fit conditions under GNS's fits and term (`zgns.fit_problems`). INCOMPLETE also holds when any
of these is true of a read fit (`zgns.gns_problems`):

- its train's or read's NEXT block does not name this declaration, its seed and that seed's column root under
  `outputs/gns`;
- its train's or read's NEXT block does not stand on the current first pass of its split and seed: that pass's record
  sha256, and gnscols' fit_sha256 of it;
- its train's term does not take GNS's fifteen columns, 15 inputs, with 32,775 term parameters and GNS's term text;
- its train's code is not this GNS's: zgns.py, gnscols.py, zgnr.py, gnrcols.py, zgnt.py, zgnw.py, gnwcols.py,
  gnpcols.py and zpfp.py;
- its split's and seed's gate does not hold (`zgns.gate_problems`, section 2.3).

## 5. The seed-0 stop

s0stop's stop, with GNS's entry patched into its stage table in the process (`zgns.py s0stop-check` and `s0stop-run`).
It decides on LTG's four in-domain WQ-musique reads:

- STOP on a LOSS among them;
- GO on any problem or missing read, and otherwise.

The WQ-hotpotqa fit's command is wrapped in `zgns.py s0stop-run --`. `zgns.py gate` is s0stop's gate under the same
entry.

## 6. Reported beside, deciding nothing

- The six beside reads, and DEP's sixteen-read rule on the same numbers.
- GNT's, GNW's, GNR's and GNX's filed screens. GNT's delta beside GNS's shows what moving the places alone does.
- The first pass of each split: its R@5 on each carve, in sample on the training carves.
- Each carve's column census:
  - its questions;
  - the share whose first-pass five are rrf's five as a set;
  - the golds among the first pass's five and among rrf's;
  - the questions with a gold.
- The parameter counts.

## 7. What follows (fixed now)

| outcome | what follows |
| --- | --- |
| ADOPT_FOR_FULL_RUN | The stack takes the attention by an amendment to docs/STK_GNN_STACK.md, since GNS is not one of its members. Of GNT, GNW and GNS, the stack takes the one that ADOPTs with the largest mean seed-0 delta over the ten deciding reads. GNS's columns are per seed: seeds 1 and 2 need their own first passes and columns, from the base fits of those seeds, before any stack fit reads them. GNS's own full run is held for the stack. The next form, in its own file: a second round, with the places re-anchored on GNS's own first-pass top five. |
| NO_GAIN or NOT_ADOPTED | Attention to the in-sample first-pass places closes for the GNN. The next form, in its own file: GNR's first pass cross-fitted (two half fits of the base per split), so that the training carves' places are out of sample as the s1eval carves' are, for GNR's columns and GNS's. |
| INCOMPLETE | What is missing is fixed in an amendment, before any number is read. |

There is no MLP form. The MLP stays free of message passing, and this term is message passing.

## 8. Risks, stated before any number

- **Seed 0 only in the screen.** A full run decides any adoption.
- **In-sample places.**
  - The base was trained on the training carves, so its five hold more golds there than at s1eval. WQ-musique's first
    pass reads R@5 0.8190 on webqsp's training carve against 0.6632 on its s1eval carve. The gaps are 0.014 on squad,
    0.021 on hotpotqa and 0.020 on 2wiki.
  - The attention learns on places better than the ones it reads at s1eval, and may lean on them too much.
  - The NO_GAIN path of section 7 names the cross-fitted first pass.
- **Self-reinforcement.** The places are the base's own top five, so the attention can confirm the base's ranking
  instead of correcting it. A gold the base ranks below fifth can gain only through its relation to the five above it.
  The census (section 6) reports how many golds each of the two fives holds.
- **One round, on l1's states.** The places' states are l1's outputs before their own messages, and the attention is
  not iterated. The layers after l1 mix it further.
- **Whole questions in a batch.** GNT's: a place missing from a batch reads as a place the question lacks, and two rows
  on one place stop the fit.
- **Columns per first pass.** A rebuilt first pass (another sha256) makes every column record of its split stale. The
  check and the gate then refuse the fit until the columns are rebuilt.
- **float16 columns.** pair_k and res_k are float16, as GNW stores them. anchor_k is exactly 0 or 1.
- **Memory and time.**
  - The columns are GNW's layout and size per carve. Each build gets GNW's measured peak for the carve plus 2 GB, on
    2 CPUs.
  - The fits ask GNT's memory: 15 GB for the seed-0 fit and 18 GB for the WQ-hotpotqa fit.
  - The card's time per epoch should be GNT's.
- **The self-tests use toys.** They show that:
  - rrf's places give gnwcols' own columns;
  - other places give the rule's columns, and bad places stop;
  - a failed, tampered or misaligned first pass stops;
  - the gate holds a fit;
  - GNS's columns load under GNS's patches only;
  - everything is put back.

  They say nothing about R@5.

## 9. How it runs

The feeder items on the host:

1. **`gns-selftest-cols`** (`gnscols.py --selftest`) and **`gns-selftest`** (`zgns.py --selftest`), 1 CPU and 4 GB
   each, at the front of the CPU queue.
2. **`gns-match`** (`gnscols.py match --host`, 2 CPUs, 8 GB), after `gns-selftest-cols`.
3. **`gns-cols-wqm-<dataset>-<carve>`** (ten, CPU), after `gnr-first-wqm` and `gns-selftest-cols`. Each takes 2 CPUs
   and GNW's measured memory for the carve plus 2 GB.
4. **`gns-check-wqm`** (`zgns.py check-cols --split WQ-musique`, 1 CPU, 4 GB), after the ten and `gns-match`.
5. **`gns-sa-s0`**: GPU share 0.42 (cap 0.40), 15 GB, right after `gnt-at-s0` in the queue. It waits on `pkoh-gate`,
   `iso-verdict`, `ml1start-check`, `readswa-check`, `gns-selftest`, `gns-match` and `gns-check-wqm`.
6. **`gns-s0stop`** (1 CPU, 4 GB), after `gns-sa-s0`.
7. **`gns-cols-wqh-<dataset>-<carve>`** (ten, CPU) after `gnr-first-wqh`, then **`gns-check-wqh`**.
8. **`gns-sa-s0-hp`**: GPU share 0.83 (cap 0.80), 18 GB, after `gns-s0stop` and `gns-check-wqh`, wrapped, right after
   `gnt-at-s0-hp` in the queue.
9. **`gns-compare`**, **`gns-calls`** and **`gns-gate`**, on the CPU. `gns-gate` runs `gate --want ADOPT_FOR_FULL_RUN`.
10. **No full-run items.** GNS's own full run is held for the stack (section 7). It is queued only by the amendment that
    takes GNS into the stack.

```
python outputs/mp_unified/gnscols.py cols --split WQ-musique --seed 0 --dataset musique --carve s1eval --host
python outputs/mp_unified/gnscols.py match --host
python outputs/mp_unified/zgns.py check-cols --split WQ-musique
python outputs/host_ops/retry_cmd.py --tries 3 --wait 90 --cuda -- python outputs/host_ops/cuda_alloc.py --frac 0.4 --gc 0.9 -- outputs/mp_unified/zgns.py fit --name scr-gns-sa-s0
python outputs/mp_unified/zgns.py s0stop-check
python outputs/mp_unified/zgns.py s0stop-run -- python outputs/host_ops/retry_cmd.py --tries 3 --wait 90 --cuda -- python outputs/host_ops/cuda_alloc.py --frac 0.8 --gc 0.9 -- outputs/mp_unified/zgns.py fit --name scr-gns-sa-s0-hp
python outputs/mp_unified/zgns.py compare-all
python outputs/mp_unified/zgns.py calls          (outputs/gns/calls.json and .md)
python outputs/mp_unified/zgns.py gate --want ADOPT_FOR_FULL_RUN
python outputs/mp_unified/gnscols.py --selftest
python outputs/mp_unified/zgns.py --selftest
```

Only `outputs/gns/calls.json`, `calls.md` and `s0stop.json` are committed, masked. The columns, their records,
`match.json` and the `check_*.json` files are not.

Both self-tests passed on the laptop's CPU before this commit.

`gnscols.py --selftest` covers:

- **The match.** rrf's top five as the places give gnwcols' own columns at every question size.
- **Other places.** They give the rule's two scores, 0 on their own rows, with one row per place. Bad places stop.
- **Toy carves.** Their float16 columns are checked question by question on the first pass's five rows.
- **Bad first passes.** A failed, tampered, misaligned or other-golds first pass stops. Each first pass gets one
  fit_sha256.
- **The loading.** Batches, and `zltm.load_next` under GNS's names only.
- **The stops** of the identity and the places.

`zgns.py --selftest` covers:

- **The names.** GNS's fits are disjoint from every other GNN and MLP stage's, GNT's, GNW's and GNR's among them.
- **The patches.** zgnx's and ltmcols' names are GNS's inside a call, and all are put back after, also on a failure.
  The model patch is GNT's, the columns root is per seed, there is no NEXT block outside a call, other stages' fits are
  refused, and s0stop's entry is GNS's.
- **The term record** is GNS's.
- **On toy carves of WQ-musique with a toy first pass:**
  - the gate (first pass, match, check) holds a fit;
  - the check passes on GNS's columns and fails on another first pass;
  - a failed first pass fails the gate;
  - GNS's columns load inside the patches and stop outside, and GNW's stop inside;
  - the NEXT block names GNS's columns on the first pass.
- **The stamp.** Its NEXT block and term are GNS's, and it refuses a fit without columns.
- **The read checks and the calls,** on made blocks and reads.
- **What follows,** as section 7 states it.
