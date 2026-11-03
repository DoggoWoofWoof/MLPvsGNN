# GNW: the pair scores as message weights from the first stage's top five, inside the GNN of record's encoder, on PKO's pools

Declared 3 November 2026, after GNP amendment A1 (c82ca5b) and PFS's section 15 (A4 and A5), and before any GNW column or
fit. Column builder: `outputs/mp_unified/gnwcols.py`. Stage module: `outputs/mp_unified/zgnw.py`. They import
`gnpcols.py`, `zpfp.py`, `ltwdiag.py`, `chndiag.py`, `ltmcols.py`, `zltm.py`, `zgnx.py`, `zgnp.py` and `s0stop.py` and edit
none of them. GNW is not a member of the stack (STK amendment A3); the stack takes it only by an amendment to
docs/STK_GNN_STACK.md.

## 1. The question

- **Why now.** GNP amendment A1 fixed GNP's next form after either verdict: the pair scores as message weights from the
  first stage's top five inside message passing. GNW is that form. It is its own arm against the same base as GNP
  (ML1's zg6r fits, ISO KEEP), so GNP's verdict changes none of GNW's reads. Declaring it now lets its seed-0 fit share the
  card with GNP's and GNA's instead of waiting on their verdicts.
- **What PFS showed.** Sections 12 and 15 of docs/PKO_FEATURE_SCREEN.md:
  - The pair scores against rrf's top five add +0.012 to +0.017 recall@5 to the lean columns on musique, hotpotqa and
    2wiki in PFS's probe, and +0.038 on webqsp.
  - More anchors (rrf's top ten) and better anchors (a first round's own top five, R2 and R3) add nothing beside them.
  - On musique's 3-hop and 4-hop questions, no anchor set read there closes the gap.
- **How GNW differs from GNP.** GNP gives every row six fixed reductions of its pair scores (max, mean and the first
  place's) as an input block. GNW keeps each place's two scores apart. A row's state takes the anchors' own states,
  weighted by a learned softmax of its scores against each of them. What a row learns from an anchor is then the
  anchor's state, not a number about it. That is message passing along the pair scores, which the probe could not read.
- **The GNN stays its own model.** The columns are made by a rule with no fit, from the first stage's order and the
  frozen vectors, before any row is scored. No MLP and no MLP fit enter a GNW fit. A message carrying other rows' states
  is message passing, so GNW has no MLP form.
- **So the question is:** does this one message family raise the GNN's in-domain R@5 over zg6r's, above all on musique?

## 2. What runs

- **The columns** (`gnwcols.py cols`, one item per carve, CPU).
  - For each question, rrf's order is `chndiag.order_of` on the cache's rrf column (descending, ties by pool position, as
    gnpcols takes it). Its first five rows a_1 ... a_5 are the places.
  - For a row c and place k, `ltwdiag.pair_feats` gives pair_k = (c.q + a_k.q) / sqrt(2 + 2 c.a_k) and
    res_k = c.q - (a_k.q)(c.a_k), from the frozen encoder's unit vectors. These are the two scores `zpfp.pair_cols`
    reduces to GNP's six columns. anchor_k is 1 on a_k's own row and 0 on every other.
  - A row's two scores against itself are 0. So are both scores against a place its question lacks: a question of five
    rows or fewer has one place per row.
  - The fifteen columns are pair_1..5, res_1..5 and anchor_1..5. They are written float16, per cache part, in ltmcols'
    layout, under `outputs/gnw/cols/<split>/<dataset>/<carve>/`, once for each split whose fits read the carve. The
    arrays are the same under both splits.
  - Every record's `fit_sha256` is one sha256 over the rule's code (gnwcols.py, gnpcols.py, zpfp.py, ltwdiag.py and
    chndiag.py).
  - The item stops and writes nothing when zpfp's identity fails (each row's c.q against the cache's dense_cos within
    0.002 on all but 1% of the carve's rows), or when a question does not hold exactly one row for each place it has.
- **The match** (`gnwcols.py match`). On the first 64 questions of PFS's five probe carves, GNP's six columns reduced
  from these fifteen (before float16) must equal gnpcols' own six exactly: maximum difference 0.0, with no bad place. It
  writes `outputs/gnw/match.json` and exits 1 otherwise, which holds the fits.
- **The arm** (`zgnw.py`). zg6r's l1 runs once on all of a batch's rows; call its output h (N x 128). A forward hook on
  l1 adds one message:
  - for row c of question q, the valid places are those q has, other than c's own row;
  - alpha_ck = softmax over the valid places of wp * pair_ck + wr * res_ck + b_k;
  - m_c = sum over k of alpha_ck * h[a_k], and zeros when c has no valid place;
  - l1's output becomes h + W_m m_c.

  W_m is 128 x 128; wp, wr and the five b_k are single numbers. All start at zero, so a fit starts as zg6r bit for bit,
  and at W_m = 0 only W_m has a gradient. The term is 16,391 parameters, 196,609 in all at hidden 128. The message uses
  index_add_, index_select, softmax and bmm, all deterministic under the trainer's settings. In training, the
  (N, 5, 128) gather and its bmm are recomputed in backward (torch's checkpoint) instead of kept. Two rows of a batch on
  one place of one question stop the fit.
- **The rest is GNX's** (GNX section 2): the base fixed by ISO's filed verdict (KEEP: ML1's zg6r fits), the trainer, the
  card, the epochs, the KT input, the read cap and the p@swa read.
- **The names.** For the length of a GNW call:
  - ltmcols' root, stage, declaration, column names and width are GNW's (`zgnw.on_cols`). zgnx's carve then loads GNW's
    columns through `zltm.load_next`, which checks each record and each part's sha256 and refuses LTM's, LTA's and GNP's.
  - zgnx's fits, declaration, stage, stamp key (`gnw`), output root (`outputs/gnw`), fit and pair names, code, term record,
    NEXT block, IN and WIDTH (15 and 15) and model patch are GNW's.
  - The NEXT block names this declaration and GNW's root. The code adds zgnw.py, gnwcols.py, gnpcols.py and zpfp.py to
    GNX's.
  - All of it is put back afterwards, also on a failure.
- **The fits.** The screen is seed 0 on both splits: `scr-gnw-mw-s0` (WQ-musique) and `scr-gnw-mw-s0-hp` (WQ-hotpotqa).
  The compares are `scr-gnw-mw-vs-base-s0` and `-s0-hp`, against the base's fit of the same seed and split.

## 3. The reads

GNX's section 3, unchanged.

## 4. The rule (fixed now)

GNX's section 4, unchanged: LTG's in-domain rule (`zltg.decide_screen`, `zltg.decide_full`) and GNX's INCOMPLETE
conditions. INCOMPLETE also holds when any of these is true of a read fit (`zgnw.gnw_problems`):

- its train's or read's NEXT block does not name this declaration;
- its train's term does not take the fifteen columns, 15 inputs, with 16,391 term parameters;
- its train's code is not this GNW's: zgnw.py, gnwcols.py, gnpcols.py and zpfp.py;
- `outputs/gnw/match.json` is missing or did not pass.

## 5. The seed-0 stop

s0stop's stop, with GNW's entry patched into its stage table in the process (`zgnw.py s0stop-check` and `s0stop-run`). It
decides on LTG's four in-domain WQ-musique reads:

- STOP on a LOSS among them;
- GO on any problem or missing read, and otherwise.

The WQ-hotpotqa fit's command is wrapped in `zgnw.py s0stop-run --`. `zgnw.py gate` is s0stop's gate under the same entry.

## 6. Reported beside, deciding nothing

- The six beside reads, and DEP's sixteen-read rule on the same numbers.
- PFS A3's five probe records.
- GNP's, GNX's and GNA's filed screens.
- The parameter counts: the term is 16,391 parameters (GNX's is 1,920 and GNP's 1,664), 196,609 in all at hidden 128.
- Each carve's column record: the gold rows' and the other rows' mean of each column, and the questions with a bad place.

## 7. What follows (fixed now)

| outcome | what follows |
| --- | --- |
| ADOPT_FOR_FULL_RUN | The stack takes GNW's message family by an amendment to docs/STK_GNN_STACK.md (GNW is not one of A3's members). GNW's own full run is held for the stack. The next form, in its own file: learned attention to the top five, with the pair scores as its bias. |
| NO_GAIN or NOT_ADOPTED | The pair scores as message weights alone (weights from the scores, not from the states) close for the GNN. The next form, in its own file: learned attention to the top five, with the pair scores as its bias. |
| INCOMPLETE | What is missing is fixed in an amendment, before any number is read. |

There is no MLP form. The MLP stays free of message passing, and this term is message passing.

## 8. Risks, stated before any number

- **Seed 0 only in the screen.** A full run decides any adoption.
- **The probe is not the GNN.** PFS's probe read the six reductions, not per-place weights over states. The GNN may
  already carry through its edges what the anchors' states add.
- **One round, on l1's states.** The anchors' states are l1's outputs before their own messages, and the message is not
  iterated. The layers after l1 mix it further.
- **The places are rrf's top five.** PFS's section 15 found that neither more anchors nor a first round's own anchors add
  anything in the probe, so GNW does not re-anchor.
- **Whole questions in a batch.** The message needs every place of a question in the same batch as its rows. zgnx's
  trainer takes all rows of each of its 32 questions per step, and its read takes whole questions under the cap. A place
  missing from a batch would read as a place the question lacks; two rows on one place stop the fit.
- **float16.** pair_k and res_k are stored as float16, whose spacing is 2^-10 below 2. anchor_k is 0 or 1 exactly.
- **The columns are not all built when this is committed.** `gnw-check-cols` waits on every `gnw-cols-*` item, on
  `gnw-match` and on `gnw-selftest`, and holds the fits until each carve both splits read loads.
- **Memory and time.**
  - The card: GNX's seed-0 fit peaked at 8.87 GB reserved under a 0.40 cap. GNW's gather is about 164 MB at 64k rows and
    is not kept for backward.
  - The host: the columns take 30 bytes per row on the CPU instead of GNX's 14, about 1.3 GB more on the eval carves.
    The seed-0 fit asks 15 GB and the WQ-hotpotqa fit 18 GB, against GNX's measured 12.87 GB.
  - The disk: about 7 GB of columns on the host.
- **The self-tests use toys.** They show that the columns are the rule's, question by question; that they load under
  GNW's names only; that the message is the loops'; that the start is zg6r's; that the term trains; and that everything
  is put back. They say nothing about R@5.

## 9. How it runs

The feeder items on the host:

1. **`gnw-selftest-cols`** (`gnwcols.py --selftest`), then the eleven **`gnw-cols-<dataset>-<carve>`** items (2 CPUs each;
   memory from GNP's measured column items) and **`gnw-match`** (2 CPUs, 8 GB). They go before the CPU queue's held head,
   so they start as memory frees.
2. **`gnw-selftest`** (`zgnw.py --selftest`), then **`gnw-check-cols`** (1 CPU and 4 GB each). `gnw-check-cols` runs after
   every `gnw-cols-*` item, `gnw-match` and `gnw-selftest`. It writes `outputs/gnw/cols.json` and exits 1 on any problem,
   which holds the fits.
3. **`gnw-mw-s0`**: GPU share 0.42 (cap 0.40), 15 GB, right after `gnp-px-s0` in the queue. It waits on `pkoh-gate`,
   `iso-verdict`, `ml1start-check`, `readswa-check`, `gnw-selftest` and `gnw-check-cols`.
4. **`gnw-s0stop`** (1 CPU, 4 GB), after `gnw-mw-s0`.
5. **`gnw-mw-s0-hp`**: GPU share 0.83 (cap 0.80), 18 GB, after `gnw-s0stop`, wrapped, right after `gnp-px-s0-hp` in the
   queue.
6. **`gnw-compare`**, **`gnw-calls`** and **`gnw-gate`**, on the CPU. `gnw-gate` runs `gate --want ADOPT_FOR_FULL_RUN`.
7. **No full-run items.** GNW's own full run is held for the stack (section 7) and is queued only by the amendment that
   takes GNW into it.

```
python outputs/mp_unified/gnwcols.py cols --dataset musique --carve s1fit --host
python outputs/mp_unified/gnwcols.py match --host
python outputs/mp_unified/zgnw.py check-cols
python outputs/host_ops/retry_cmd.py --tries 3 --wait 90 --cuda -- python outputs/host_ops/cuda_alloc.py --frac 0.4 --gc 0.9 -- outputs/mp_unified/zgnw.py fit --name scr-gnw-mw-s0
python outputs/mp_unified/zgnw.py s0stop-check
python outputs/mp_unified/zgnw.py s0stop-run -- python outputs/host_ops/retry_cmd.py --tries 3 --wait 90 --cuda -- python outputs/host_ops/cuda_alloc.py --frac 0.8 --gc 0.9 -- outputs/mp_unified/zgnw.py fit --name scr-gnw-mw-s0-hp
python outputs/mp_unified/zgnw.py compare-all
python outputs/mp_unified/zgnw.py calls          (outputs/gnw/calls.json and .md)
python outputs/mp_unified/zgnw.py gate --want ADOPT_FOR_FULL_RUN
```

Only `outputs/gnw/calls.json`, `calls.md`, `cols.json`, `match.json` and `s0stop.json` are committed, masked. The column
records and arrays are not.

Both self-tests passed on the laptop's CPU before this commit.

- **`gnwcols.py --selftest`** covers:
  - each place's two scores, 0 on its own row, and one row per place the question has;
  - GNP's six reduce from the fifteen exactly at every question size;
  - toy carves: the written columns equal the rule's, question by question, in float16; the batch size changes nothing;
    both splits' records hold the same parts and scorer;
  - `zltm.load_next` takes them under GNW's names in ltmcols only;
  - the identity and place stops, which write nothing;
  - the match's check.
- **`zgnw.py --selftest`** covers:
  - **the names**: GNW's fits are disjoint from ML1's, ISO's, DEP's, LTG's, GNX's, GNP's, GNA's, LTA's, LTM's, RTR's,
    IRA's and ISB's;
  - **the message**: against loops in float64, on questions of one to nine rows, rows out of question order, a place a
    question lacks, and two rows on one place (which stop); the checkpointed gather gives the plain one's values and
    gradients exactly;
  - **the patches**: every patched name is GNW's inside a call and put back after it, also on a failure;
  - **the columns**: gnwcols' columns load inside the patches and stop outside them; LTM's and GNP's columns stop inside;
  - **the toy term**: the four parameters start at zero; zg6r's forwards are unchanged bit for bit at W_m = 0, and only
    W_m has a gradient there; the forward's message is the loops'; the term moves the score; a batch with no NEXT block,
    a narrower one, l1 run outside the forward, and two anchors on one place each stop; the term trains in lean_gpu's
    loop; the term record is 16,391 parameters, 196,609 in all;
  - **the stamp**: its NEXT block is GNW's with GNW's term, and it refuses a fit without columns;
  - **the read checks and the calls** on made reads, INCOMPLETE without `match.json`;
  - **what follows**, as section 7 states it.
