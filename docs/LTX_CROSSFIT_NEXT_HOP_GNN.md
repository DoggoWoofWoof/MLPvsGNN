# LTX: LTG's next-hop term cross-fitted inside the GNN of record, on PKO's pools

Declared 3 November 2026, after LTG's screen verdict (NOT_ADOPTED, 538e467) and LTG amendment A5
(docs/LTG_NEXT_HOP_GNN.md section 13, in the same commit), and before any LTX column or fit. Column builder:
`outputs/mp_unified/ltxcols.py`. Stage module: `outputs/mp_unified/zltx.py`. They import `ltmcols.py`, `lean_cache.py`,
`zltm.py`, `zltg.py`, `zgnx.py` and `s0stop.py` and edit none of them. LTX is not a member of the stack (STK amendment
A3); the stack takes it only by an amendment to docs/STK_GNN_STACK.md.

## 1. The question

- **Why now.** LTG's term lost two in-domain reads: the WQ-hotpotqa fit's musique R@5 (-0.0243) and its 3-hop slice
  (-0.0224). LTG's section 6 named the cumulative residual as the next form. A5 puts this form first, because of what the
  fits' own records show.
- **What the fits show** (A5):
  - LTG fit its training questions better than the base did. Its listwise loss over the last epoch was 1.6017 against
    1.7635 on WQ-hotpotqa, and 1.3592 against 1.4189 on WQ-musique.
  - Held-out questions of the same datasets did no better, and musique's did worse.
  - The scorer that weights every message was trained on the golds of the questions the layers train on. In training, an
    edge's score already knows which rows are the question's next hops; at the read it does not. So the layers learn to
    lean on a score that is worth less on held-out questions.
  - musique's training questions share golds. LTM's census of musique s1fit has 731 components of questions joined by a
    shared gold node over 4,601 questions; the largest holds 455. So a scorer fit on one question has seen golds of others.
  - The MLP's next-hop columns (LTM, ADOPT, 7874e4d) are cross-fitted by those components. LTG's scorer was not.
- **What LTX changes.** Only who trains which scorer, and which scorer a question's message passing reads:
  - two scorers, one per fold;
  - on a training carve the folds are components, so no gold node is in both;
  - each question's layers read the scorer of the fold it is not in, detached;
  - the next-hop loss trains the scorer of the question's own fold;
  - the main loss trains no scorer.

  So no scorer a question's message passing reads has seen that question's golds, or any gold node of its component, in
  training as at the read.
- **The GNN stays its own model.** The folds are a rule with no fit: components of the carve's gold lists on a training
  carve, a draw on any other, made before any row is scored. No MLP and no MLP fit enter an LTX fit.
- **So the question is:** does LTG's next-hop term, cross-fitted, raise the GNN's in-domain R@5 over zg6r's, above all
  on musique and its 3- and 4-hop questions?

## 2. What runs

- **The folds** (`ltxcols.py cols --host`, one item, CPU).
  - **A training carve** is one a split trains on (`ltmcols.train_carves`). Its folds are `ltmcols.comp_folds` with two
    folds: the components of the questions joined by a shared gold node (a question without a gold alone), dealt largest
    first to the fold with fewer questions. The components must equal LTM's own question for question (Q_comp in LTM's
    lists file, `outputs/ltm/work/lists/`), and no gold node may sit in both folds. Otherwise nothing is written.
  - **Any other carve** (the s1eval carves): each question's fold is drawn, `default_rng([0, ltmcols.DS_K[ds],
    11]).integers(0, 2)`.
  - Every row of a question holds its fold, 0.0 or 1.0. The column is written float16, per cache part, in ltmcols'
    layout, under `outputs/ltx/cols/<split>/<dataset>/<carve>/`, once for each split whose fits read the carve. The
    rule fits nothing, so the arrays are the same under both splits.
  - Every record's `fit_sha256` is one sha256 over the rule's code (ltxcols.py and ltmcols.py). A carve's record also
    holds its components (count, largest, singletons), each fold's rows and gold rows, and the identity check.
- **The arm** (`zltx.py`): LTG's term (docs/LTG_NEXT_HOP_GNN.md section 2) with two scorers in place of one.
  - `ltx_V` and `ltx_U` hold two of LTG's scorers, each V and U of 3 x 256 x 32. Scorer 0's V is LTG's draw (generator
    seed + 4101). Scorer 1's V is the next draw from the same generator. Both U start at 0.
  - A question's fold f(q) comes from its rows' fold column. A value other than 0 and 1, or a question with rows of both
    folds, stops the fit.
  - Every layer's edge score on an edge of question q reads the P and Q of scorer 1 - f(q), detached.
  - The next-hop loss (LTG's, λ = 1, on the same gold senders and receivers) reads the P and Q of scorer f(q), with
    their gradients.
  - The layers' W_k, gains and summaries are LTG's. They start at zero and are trained by the main loss. With U = 0 every
    edge score is 0 and every layer's term is 0, so a fit starts as zg6r bit for bit.
  - An edge between two questions' rows stops the fit. zlink builds none; the check guards the fold selection.
  - The term is 117,897 parameters: two scorers of 49,152 each, and LTG's three layer terms of 6,531 each (LTG's term is
    68,745). That is 298,115 in all at hidden 128.
- **The rest is GNX's and LTG's** (GNX section 2, LTG section 2): the base fixed by ISO's filed verdict (KEEP: ML1's zg6r
  fits), the trainer, the card, the epochs, the KT input, the read cap and the p@swa read. LTG's store terms, layers,
  edge products and next-hop loss run unedited.
- **The names.** For the length of an LTX call:
  - ltmcols' root, stage, declaration, column names and width are LTX's (`zltx.on_cols`). zgnx's carve then loads LTX's
    folds through `zltm.load_next`, which checks each record and each part's sha256 and refuses LTM's, GNP's and GNW's.
  - zgnx's fits, declaration, stage, stamp key (`ltx`), output root (`outputs/ltx`), fit and pair names, code, term
    record, NEXT block, IN and WIDTH (1 and 1), model patch, arm import and stamp are LTX's.
  - For the length of an LTX model, zltg's layer names are LTX's, so zltg's own layers run on LTX's parameters.
  - The NEXT block names this declaration, LTX's root and ltxcols' rule. The code adds zltx.py and ltxcols.py to GNX's.
    A train's stamp also holds the next-hop curve.
  - All of it is put back afterwards, also on a failure.
- **The fits.** The screen is seed 0 on both splits: `scr-ltx-xf-s0` (WQ-musique) and `scr-ltx-xf-s0-hp` (WQ-hotpotqa).
  The compares are `scr-ltx-xf-vs-base-s0` and `-s0-hp`, against the base's fit of the same seed and split.

## 3. The reads

GNX's section 3, unchanged.

## 4. The rule (fixed now)

GNX's section 4, unchanged: LTG's in-domain rule (`zltg.decide_screen`, `zltg.decide_full`) and GNX's INCOMPLETE
conditions, with LTX's term in the stamp check. A read fit's train and read must name the term (117,897 parameters, two
folds, rank 32, 256 axes, basis sha `0ca4797b...`, λ = 1, three layers, the one column `fold`), its base fit and ISO's
verdict, and its train must record a next-hop step. INCOMPLETE also holds when any of these is true of a read fit
(`zltx.ltx_problems`):

- its train's or read's NEXT block does not name this declaration, or its folds come from another rule than ltxcols';
- its train's term does not take the one fold column, 1 input, with 117,897 term parameters over two folds;
- its train's code is not this LTX's: zltx.py and ltxcols.py;
- `outputs/ltx/cols.json` is missing, did not pass, or counts a NaN in any carve.

## 5. The seed-0 stop

s0stop's stop, with LTX's entry patched into its stage table in the process (`zltx.py s0stop-check` and `s0stop-run`). It
decides on LTG's four in-domain WQ-musique reads:

- STOP on a LOSS among them;
- GO on any problem or missing read, and otherwise.

The WQ-hotpotqa fit's command is wrapped in `zltx.py s0stop-run --`. `zltx.py gate` is s0stop's gate under the same entry.

## 6. Reported beside, deciding nothing

- The six beside reads, and DEP's sixteen-read rule on the same numbers.
- LTG's, GNX's and GNW's filed screens.
- Each fit's next-hop curve (listwise and next-hop loss and gold senders, per 500 steps) in its train stamp, and its last
  window in `outputs/ltx/calls.json`, beside A5's table.
- The parameter counts: the term is 117,897 parameters (LTG's is 68,745), 298,115 in all at hidden 128.
- Each carve's fold record: its components, and each fold's questions, rows and gold rows.

## 7. What follows (fixed now)

| outcome | what follows |
| --- | --- |
| ADOPT_FOR_FULL_RUN | The stack takes LTX's cross-fitted term by an amendment to docs/STK_GNN_STACK.md (LTX is not one of A3's members). LTX's own full run is held for the stack. |
| NO_GAIN or NOT_ADOPTED | The cross-fitted next-hop term closes for the GNN. The next form, in its own file: LTG's cumulative residual (docs/LTG_NEXT_HOP_GNN.md section 6). |
| INCOMPLETE | What is missing is fixed in an amendment, before any number is read. |

There is no MLP form here. The MLP's next-hop columns are LTM's, cross-fitted already (ADOPT, 7874e4d).

## 8. Risks, stated before any number

- **Seed 0 only in the screen.** A full run decides any adoption.
- **A5 is a reading, not a measurement.** The lower training loss and the held-out LOSS fit a leak through the scorer.
  They fit a term that is simply worse on musique too. This screen tells the two apart only in part.
- **Each scorer sees half the golds.** Each trains on about half of the training questions, so each is weaker than LTG's.
- **The folds keep gold nodes apart, not every shared row.** As in LTM (its section 9), a shared non-gold passage or
  entity can still cross folds.
- **Drawn folds on the eval carves.** An eval question's layers read the scorer of the fold it was not drawn into. That
  scorer is fit on about half the training questions, as for a training question. The read is p@swa, which averages
  both scorers with every other weight.
- **Both scorers train inside one fit.** In training, a question's layers read a scorer that is still being fit on the
  other fold; at the read, both are final.
- **Memory and time.**
  - The card: LTG's WQ-musique fit peaked at 7.95 GB under a 0.40 cap, and its WQ-hotpotqa fit at 18.73 GB (17.45 GiB)
    under the 19.19 GiB of a 0.80 cap. The second scorer adds one (N, 256) x (256, 160) product a step, and the fold
    selection four N x 96 tensors: at most about 0.3 GB at 64k rows.
  - Time: LTG's fits took 4,161 s and 10,098 s. The second scorer's products add a few percent.
  - The host: the column takes 2 bytes a row on the CPU (GNX's take 14). The fits ask 15 GB and 18 GB, as GNW's do.
- **The columns are not built when this is committed.** `ltx-check-cols` waits on `ltx-cols` and `ltx-selftest`, and
  holds the fits until each carve both splits read loads.
- **The self-tests use toys.** They show that the folds are the rule's, row by row; that they load under LTX's names
  only; that each question's layers read the other fold's scorer and its next-hop loss trains its own; that the start
  is zg6r's; that the term trains; and that everything is put back. They say nothing about R@5.

## 9. How it runs

The feeder items on the host:

1. **`ltx-selftest-cols`** (`ltxcols.py --selftest`), then **`ltx-cols`** (`ltxcols.py cols --host`, every carve either
   split reads; 2 CPUs, 6 GB). They go before the CPU queue's held head, so they start as memory frees.
2. **`ltx-selftest`** (`zltx.py --selftest`), then **`ltx-check-cols`** (1 CPU and 4 GB each). `ltx-check-cols` runs
   after `ltx-cols` and `ltx-selftest`. It writes `outputs/ltx/cols.json` and exits 1 on any problem, which holds the
   fits.
3. **`ltx-xf-s0`**: GPU share 0.42 (cap 0.40), 15 GB, right after `gnw-mw-s0` in the queue. It waits on `pkoh-gate`,
   `iso-verdict`, `ml1start-check`, `readswa-check`, `ltx-selftest` and `ltx-check-cols`.
4. **`ltx-s0stop`** (1 CPU, 4 GB), after `ltx-xf-s0`.
5. **`ltx-xf-s0-hp`**: GPU share 0.83 (cap 0.80), 18 GB, after `ltx-s0stop`, wrapped, right after `gnw-mw-s0-hp` in the
   queue.
6. **`ltx-compare`**, **`ltx-calls`** and **`ltx-gate`**, on the CPU. `ltx-gate` runs `gate --want ADOPT_FOR_FULL_RUN`.
7. **No full-run items.** LTX's own full run is held for the stack (section 7) and is queued only by the amendment that
   takes LTX into it.

```
python outputs/mp_unified/ltxcols.py cols --host
python outputs/mp_unified/zltx.py check-cols
python outputs/host_ops/retry_cmd.py --tries 3 --wait 90 --cuda -- python outputs/host_ops/cuda_alloc.py --frac 0.4 --gc 0.9 -- outputs/mp_unified/zltx.py fit --name scr-ltx-xf-s0
python outputs/mp_unified/zltx.py s0stop-check
python outputs/mp_unified/zltx.py s0stop-run -- python outputs/host_ops/retry_cmd.py --tries 3 --wait 90 --cuda -- python outputs/host_ops/cuda_alloc.py --frac 0.8 --gc 0.9 -- outputs/mp_unified/zltx.py fit --name scr-ltx-xf-s0-hp
python outputs/mp_unified/zltx.py compare-all
python outputs/mp_unified/zltx.py calls          (outputs/ltx/calls.json and .md)
python outputs/mp_unified/zltx.py gate --want ADOPT_FOR_FULL_RUN
```

Only `outputs/ltx/calls.json`, `calls.md`, `cols.json` and `s0stop.json` are committed, masked. The column records and
arrays are not.

Both self-tests passed on the laptop's CPU before this commit.

- **`ltxcols.py --selftest`** covers:
  - two folds by components: questions that share a gold node share a fold, no gold node sits in both, and components
    are dealt largest first;
  - the draw on a carve no split trains on;
  - toy carves: each row holds its question's fold, float16 0 or 1, and both splits' records hold the same parts and
    rule;
  - LTM's components gate a training carve: another labelling, a lists file made on a limit, a missing file and a split
    component each stop;
  - `zltm.load_next` takes the folds under LTX's names in ltmcols only.
- **`zltx.py --selftest`** covers:
  - **the names**: LTX's fits are disjoint from ML1's, ISO's, DEP's, LTG's, GNX's, RTR's, IRA's, ISB's, LTM's, GNA's,
    LTA's, GNP's and GNW's; the constants are ltxcols' own;
  - **the folds of a batch**: each question's from its rows; a value other than 0 and 1, a question of both folds and an
    edge between two questions stop; a question without rows is in fold 0;
  - **the patches**: every patched name is LTX's inside a call and put back after it, also on a failure;
  - **the columns**: ltxcols' folds load inside the patches and stop outside them; LTM's and GNW's columns stop inside;
  - **the toy term**:
    - V's two draws (scorer 0's is LTG's) and the zeros; zg6r's eval and training forwards bit for bit at the start;
    - each scorer's P and Q are zltg's on a view of it;
    - the layers read the other fold's scorer, detached, and the next-hop loss the question's own;
    - with the two scorers equal, the forwards are LTG's model's bit for bit;
    - the main loss gives no scorer a gradient and every layer term one;
    - the next-hop loss on one fold's golds moves that fold's scorer alone, the other's gradient exactly 0;
    - seven stops;
    - checkpointing on and off agree;
    - the strict load with the patch, and its refusal without;
    - the term trains in lean_gpu's loop;
    - the term record is 117,897 parameters, 298,115 in all;
  - **the stamp**: its NEXT block is LTX's, with LTX's term and the next-hop curve on the train alone, and it refuses a
    fit without columns;
  - **the read checks and the calls** on made reads, INCOMPLETE without `cols.json`;
  - **what follows**, as section 7 states it.
