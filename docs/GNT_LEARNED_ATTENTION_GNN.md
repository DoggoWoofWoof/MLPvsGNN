# GNT: learned attention to the first stage's top five, with the pair scores as its bias, in the GNN of record's encoder, on PKO's pools

Declared 3 November 2026. This is before GNW's verdict and before any GNT fit. GNW's seed-0 fit has not started.

- **Stage module:** `outputs/mp_unified/zgnt.py`.
- **Imports, none of them edited:** `zgnw.py`, `gnwcols.py`, `gnpcols.py`, `zpfp.py`, `zgnx.py`, `zgnp.py` and `s0stop.py`.
- **Columns:** GNT builds none. It reads GNW's fifteen columns under `outputs/gnw/cols/` as they are, under GNW's own
  names. GNW's column records, `match.json` and `cols.json` are read only.
- **Stack:** GNT is not a member of the stack (docs/STK_GNN_STACK.md). The stack takes it only by an amendment.

## 1. The question

- **Why now.** GNW's section 7 names one next form after any GNW verdict: learned attention to the top five, with the
  pair scores as its bias. GNT is that form.
  - It is its own arm against the same base as GNW (ML1's zg6r fits, ISO KEEP), so GNW's verdict changes none of GNT's
    reads.
  - Declaring it now lets its seed-0 fit take the card's half that frees at about 17:00 on 3 November. Waiting on GNW's
    verdict would hold it until about 4 November.
- **How GNT differs from GNW.**
  - GNW weights each of the five anchors' states by a softmax of fixed scores only: wp * pair_k + wr * res_k + b_k.
    Every row weights the anchors by the same two numbers.
  - GNT adds a learned term from the states themselves, u_c W_a u_k. This compares the row's own l1 state with each
    anchor's.
  - So a row can learn which anchor it completes from what the GNN's first layer knows about both: their inputs, the walk
    and the chain. The frozen encoder's pair scores alone cannot express that.
- **What the earlier forms showed.**
  - GNX (NO_GAIN, ec09b5a) gave the GNN fixed columns about rrf's top five.
  - GNA and GNP are being read now.
  - PFS's probe (docs/PKO_FEATURE_SCREEN.md sections 12 and 15) found the pair scores carry signal (+0.012 to +0.017
    recall@5 on musique, hotpotqa and 2wiki).
  - None of these lets the weights depend on learned states. GNT does.
- **The GNN stays its own model.**
  - The columns are made by a rule with no fit, before any row is scored.
  - No MLP and no MLP fit enter a GNT fit.
  - The attention passes the anchors' states, so it is message passing, and GNT has no MLP form.
- **So the question is:** does learned attention to the first stage's top five raise the GNN's in-domain R@5 over
  zg6r's, above all on musique?

## 2. What runs

- **The columns.** GNW's, read as they are. For each row they give:
  - pair_1..5 and res_1..5: the frozen encoder's two question-conditional pair scores of the row against rrf's first five
    rows a_1 ... a_5;
  - anchor_1..5: 1 on a_k's own row and 0 elsewhere.

  Both scores are 0 on a place's own row and for a place the question lacks (gnwcols.py, docs/GNW_PAIR_WEIGHTS_IN_GNN.md
  section 2). They load through `zltm.load_next` under GNW's names. Each record and each part's sha256 is checked, and
  LTM's, LTA's and GNP's columns are refused.
- **The arm** (`zgnt.py`). zg6r's l1 runs once on all of a batch's rows. Call its output h (N x 128) and
  u_c = h_c / max(|h_c|, 1e-6). A forward hook on l1 adds one message:
  - For row c of question q, the valid places are those q has, other than c's own row.
  - alpha_ck is the softmax over the valid places of u_c W_a u_{a_k} + wp * pair_ck + wr * res_ck + b_k.
  - m_c is the sum over k of alpha_ck * h[a_k]. It is zero when c has no valid place.
  - l1's output becomes h + W_m m_c.
- **The parameters.** W_m and W_a are 128 x 128 each. wp, wr and the five b_k are single numbers.
  - All start at zero, so a fit starts as zg6r bit for bit.
  - At W_m = 0, only W_m has a gradient. W_a has one once W_m moves. W_a is one matrix, not a product of two, so a zero
    start is not a saddle, and nothing is drawn.
  - The term is 32,775 parameters, 212,993 in all at hidden 128 (GNW's is 16,391).
- **How the message is computed.** One place at a time: index_select of the place's row per question and then per row,
  a row-wise dot product, softmax, and a weighted sum in a fixed order.
  - All of these are deterministic under the trainer's settings.
  - Memory does not grow with the number of questions in a batch.
  - In training the message is recomputed in backward (torch's checkpoint) instead of kept.
  - Two rows of a batch on one place of one question stop the fit.
- **The rest is GNX's** (GNX section 2): the base fixed by ISO's filed verdict (KEEP: ML1's zg6r fits), the trainer, the
  card, the epochs, the KT input, the read cap and the p@swa read.
- **The names.** For the length of a GNT call:
  - ltmcols' root, stage, declaration, column names and width are GNW's (`zgnw.on_cols`), so zgnx's carve loads GNW's
    columns.
  - zgnx's fits, declaration, stage, stamp key (`gnt`), output root (`outputs/gnt`), fit and pair names, code, term
    record, NEXT block, IN and WIDTH (15 and 15) and model patch are GNT's.
  - The NEXT block names this declaration, GNW's column root and GNW's declaration.
  - The code adds zgnt.py, zgnw.py, gnwcols.py, gnpcols.py and zpfp.py to GNX's.
  - All of it is put back afterwards, also on a failure.
- **The fits.** The screen is seed 0 on both splits: `scr-gnt-at-s0` (WQ-musique) and `scr-gnt-at-s0-hp` (WQ-hotpotqa).
  The compares are `scr-gnt-at-vs-base-s0` and `-s0-hp`, against the base's fit of the same seed and split.
- **A fit refuses to start** unless GNW's `match.json` and `cols.json` exist and passed.

## 3. The reads

GNX's section 3, unchanged.

## 4. The rule (fixed now)

GNX's section 4, unchanged: LTG's in-domain rule (`zltg.decide_screen`, `zltg.decide_full`) and GNX's INCOMPLETE
conditions. INCOMPLETE also holds when any of these is true of a read fit (`zgnt.gnt_problems`):

- its train's or read's NEXT block does not name this declaration, GNW's column root (`outputs/gnw`) and GNW's
  declaration;
- its train's term does not take GNW's fifteen columns, 15 inputs, with 32,775 term parameters;
- its train's code is not this GNT's: zgnt.py, zgnw.py, gnwcols.py, gnpcols.py and zpfp.py;
- GNW's `outputs/gnw/match.json` or `outputs/gnw/cols.json` is missing or did not pass.

## 5. The seed-0 stop

s0stop's stop, with GNT's entry patched into its stage table in the process (`zgnt.py s0stop-check` and `s0stop-run`).
It decides on LTG's four in-domain WQ-musique reads:

- STOP on a LOSS among them;
- GO on any problem or missing read, and otherwise.

The WQ-hotpotqa fit's command is wrapped in `zgnt.py s0stop-run --`. `zgnt.py gate` is s0stop's gate under the same
entry.

## 6. Reported beside, deciding nothing

- The six beside reads, and DEP's sixteen-read rule on the same numbers.
- PFS A3's five probe records.
- GNW's, GNP's, GNX's and GNA's filed screens.
- The parameter counts.

## 7. What follows (fixed now)

| outcome | what follows |
| --- | --- |
| ADOPT_FOR_FULL_RUN | The stack takes GNT's attention by an amendment to docs/STK_GNN_STACK.md, since GNT is not one of its members. If GNW also passes, the stack takes the one with the larger mean seed-0 delta over the ten deciding reads. GNT's own full run is held for the stack. The next form, in its own file: the same attention to the GNN's own top five after a first pass (GNR's first pass). |
| NO_GAIN or NOT_ADOPTED | Learned attention to the first stage's top five closes for the GNN. The next form, in its own file: the same attention to the GNN's own top five after a first pass (GNR's first pass). |
| INCOMPLETE | What is missing is fixed in an amendment, before any number is read. |

There is no MLP form. The MLP stays free of message passing, and this term is message passing.

## 8. Risks, stated before any number

- **Seed 0 only in the screen.** A full run decides any adoption.
- **The GNN may already carry it.** GNX's fixed columns about the same five anchors gave NO_GAIN. The edges may already
  carry what the anchors' states add.
- **One round, on l1's states.** The anchors' states are l1's outputs before their own messages, and the attention is not
  iterated. The layers after l1 mix it further.
- **The places are rrf's top five.** The GNN's own top five are GNR's subject and section 7's next form.
- **Whole questions in a batch.** The message needs every place of a question in the same batch as its rows. zgnx's
  trainer takes all rows of each of its 32 questions per step, and its read takes whole questions under the cap. A place
  missing from a batch reads as a place the question lacks. Two rows on one place stop the fit.
- **float16 columns.** pair_k and res_k are float16, as GNW stores them. anchor_k is exactly 0 or 1.
- **Memory and time.**
  - The card: the base's read peaked at 9.34 GB (8,904 MiB) against a 0.40 cap of 9,828 MiB on the 24,570 MiB card.
  - The message holds about three N x 128 float32 arrays at once in the read: about 50 MB at the read's 32,768 rows. In
    training it is recomputed in backward.
  - The host asks GNW's memory: 15 GB for the seed-0 fit and 18 GB for the WQ-hotpotqa fit.
  - The card's time per epoch should be close to GNW's. The message is five index_selects and dot products per batch on
    top of zg6r's layers.
- **The self-test uses toys.** It shows that:
  - the message is the loops';
  - the start is zg6r's;
  - the term trains;
  - GNW's columns load under GNT's patches only;
  - everything is put back.

  It says nothing about R@5.

## 9. How it runs

The feeder items on the host:

1. **`gnt-selftest`** (`zgnt.py --selftest`, 1 CPU, 4 GB), at the front of the CPU queue.
2. **`gnt-at-s0`**: GPU share 0.42 (cap 0.40), 15 GB, right after `gnr-nx-s0` in the queue. It waits on `pkoh-gate`,
   `iso-verdict`, `ml1start-check`, `readswa-check`, `gnw-check-cols`, `gnw-match` and `gnt-selftest`. In the GPU's order
   it comes before the 0.83 phase, so it takes the half of the card that GNW's seed-0 fit frees.
3. **`gnt-s0stop`** (1 CPU, 4 GB), after `gnt-at-s0`.
4. **`gnt-at-s0-hp`**: GPU share 0.83 (cap 0.80), 18 GB, after `gnt-s0stop`, wrapped, right after `gnr-nx-s0-hp` in the
   queue.
5. **`gnt-compare`**, **`gnt-calls`** and **`gnt-gate`**, on the CPU. `gnt-gate` runs `gate --want ADOPT_FOR_FULL_RUN`.
6. **No full-run items.** GNT's own full run is held for the stack (section 7). It is queued only by the amendment that
   takes GNT into the stack.

```
python outputs/host_ops/retry_cmd.py --tries 3 --wait 90 --cuda -- python outputs/host_ops/cuda_alloc.py --frac 0.4 --gc 0.9 -- outputs/mp_unified/zgnt.py fit --name scr-gnt-at-s0
python outputs/mp_unified/zgnt.py s0stop-check
python outputs/mp_unified/zgnt.py s0stop-run -- python outputs/host_ops/retry_cmd.py --tries 3 --wait 90 --cuda -- python outputs/host_ops/cuda_alloc.py --frac 0.8 --gc 0.9 -- outputs/mp_unified/zgnt.py fit --name scr-gnt-at-s0-hp
python outputs/mp_unified/zgnt.py compare-all
python outputs/mp_unified/zgnt.py calls          (outputs/gnt/calls.json and .md)
python outputs/mp_unified/zgnt.py gate --want ADOPT_FOR_FULL_RUN
python outputs/mp_unified/zgnt.py --selftest
```

Only `outputs/gnt/calls.json`, `calls.md` and `s0stop.json` are committed, masked.

`zgnt.py --selftest` passed on the laptop's CPU before this commit. It covers:

- **The names.** GNT's fits are disjoint from every other GNN and MLP stage's, GNW's and GNR's among them.
- **The message.** It is checked against loops in float64 on:
  - questions of one to nine rows;
  - rows out of question order;
  - a place a question lacks;
  - W_a, wp, wr and b all set.

  Two rows on one place stop. The checkpointed message gives the plain one's values and gradients exactly. A two-row
  question's rows each take the other's state whole.
- **The patches.** Every patched name is GNT's inside a call, ltmcols' names are GNW's there, and all are put back after,
  also on a failure.
- **The columns.**
  - gnwcols' columns load inside the patches, and the NEXT block names GNT's declaration, GNW's root and GNW's
    declaration.
  - They stop outside the patches.
  - LTM's and GNP's columns stop inside.
- **The toy term.**
  - The five parameters start at zero.
  - zg6r's forwards are unchanged bit for bit at W_m = 0, and only W_m has a gradient there.
  - With the term set, the forward's message is the loops' and every term parameter has a gradient.
  - The term moves the score.
  - A batch with no NEXT block, a narrower one, l1 run outside the forward, and two anchors on one place each stop.
  - The term trains in lean_gpu's loop, and both W_m and W_a move.
  - The term record is 32,775 parameters, 212,993 in all.
- **The stamp.** Its NEXT block is GNT's with GNT's term, and it refuses a fit without columns.
- **The read checks and the calls,** on made reads. They are INCOMPLETE without GNW's `match.json` or `cols.json`.
- **What follows,** as section 7 states it.
