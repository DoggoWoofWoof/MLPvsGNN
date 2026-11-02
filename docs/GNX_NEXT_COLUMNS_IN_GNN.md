# GNX: the GNN of record with LTM's NEXT columns as one more input of its encoder, on PKO's pools

Declared 3 November 2026, before any GNX fit. Stage module: `outputs/mp_unified/zgnx.py`. It imports `zml1s.py`,
`ziso.py`, `zdep.py`, `zltg.py`, `zltm.py`, `ltmcols.py`, `ml1start.py`, `readswa.py` and `zg6r.py` and edits none of
them.

## 1. The question

- **What the MLP gained.** LTM's full run (docs/LTM_NEXT_HOP_MLP.md section 12, ADOPT, 7874e4d) gave zrckp seven NEXT
  columns per pool row: LTW's learned next-hop score of the row against the first stage's top five rows, cross-fitted on
  the split's training carves. Over three seeds:
  - GAIN on 2wiki under both splits (mean +0.0083 under WQ-musique, +0.0101 under WQ-hotpotqa);
  - GAIN on musique's 3-hop slice under WQ-hotpotqa (+0.0079);
  - Claim 1's two reads rose at every seed, within their floors: hotpotqa under WQ-musique +0.0066, musique under
    WQ-hotpotqa +0.0056.
- **What the GNN lacks.** The GNN of record (zg6r after ISO) does not read these columns. LTG (docs/LTG_NEXT_HOP_GNN.md)
  puts a learned next-hop term inside the GNN's message passing, trained end to end with the GNN's weights. GNX takes the
  other route: the same columns the MLP reads, as one more input of the GNN's encoder.
- **The GNN stays its own model.** The columns are a fixed input of each pool row, made by LTW's scorer before any row is
  scored. Neither the MLP nor any of its fits enters a GNX fit.
- **Equal inputs.** zrckpn is the MLP candidate. While the GNN of record does not read the NEXT columns, the two models
  read different inputs, and Claim 3's timing must not compare unequal inputs (F1). GNX gives the GNN the same input set.
- **So the question is:** do LTM's NEXT columns, as one more input block of zg6r's encoder, raise the GNN's in-domain R@5,
  above all on musique and its 3- and 4-hop questions?

## 2. What runs

- **The arm.** The GNN of record after ISO, with one more input. Everything else is the base's:
  - the trainer and its loss, the card and its epochs;
  - PKO's roots and typed links, the KT input, the 64-wide state, the seed offsets;
  - the read cap of 32,768, and the candidate p@swa (readswa, LTG amendment A2);
  - on ISO's KEEP, ML1's zspk start read under ML1's names (ml1start, LTG amendment A1).
- **The columns.** They are LTM's files, `outputs/ltm/cols/<split>/<dataset>/<carve>/`:
  - seven per row (`next_1` to `next_5`, `next_max`, `next_n`), float16, from one scorer per split (`fit_sha256`);
  - a carve built in a GNX fit loads its split's columns with `zltm.load_next`. That checks the record (stage,
    declaration, split, carve, columns, training flag, identity) and each cache part's record sha256, rows and scorer. With
    verify on, it also checks each file's sha256;
  - the columns stay on the CPU (at most 1.15 GB, the six s1eval carves of a split). Each batch's rows go to the device as
    float32, with NaN and infinities set to 0, as zrckpn's carve hands them;
  - the training carves' columns are cross-fitted: each row's score comes from a fold fit that did not see its question.
    The s1eval carves' columns come from the split's scorer, fit on the split's training carves. WQ-musique's scorer never
    saw musique, and WQ-hotpotqa's never saw hotpotqa.
- **The input.** Each row's [raw, z, 1], 15 numbers, formed as zg6's encode forms any block:
  - raw is the seven columns;
  - z is their z-score against the question's retrieved rows (`lean_screen3.seg_zscore_ref`);
  - 1 is the mask (lean_gpu's keep is all ones).

  This is how zrckpn's encoder takes the block.
- **The term.** gnx_w (hidden x 15, zeros at the start) is applied to the input. Its output is added to the output of
  zg6r's l1, through a forward hook, before zg6r's GELU and dropout. So the encoder's first layer reads the NEXT block as
  one more set of input columns. zg6r's l1.weight stays the one it copies from zspk (on KEEP) or ISO's own draw (on
  REPLACE).
- **The start is zg6r's, bit for bit.** gnx_w is zeros and takes no draw, so zg6r's own draws and the training's random
  stream are the base fit's. Unlike LTG's term, this start is not a saddle: the listwise loss gives gnx_w a gradient above
  0 at the first step (the self-test shows it).
- **Parameters.** 1,920 at hidden 128 (128 x 15). The stack and readout go from 180,218 to 182,138.
- **How it is set.** For the length of a GNX fit zgnx patches, in-process:
  - ZG6R's `__init__`, which adds gnx_w and the hook after zg6r's own;
  - ZG6R's `forward`, which forms the batch's NEXT input for the hook and clears it after;
  - the arm's carve (zg6r's G6rLinkCarve): its `__init__` loads the carve's columns, and its `batch` adds the batch's
    rows of them under `NEXT`;
  - zml1s's train and read, which run with the fit's split set. The stamp refuses a train that did not load every
    training carve's columns, and a read that did not load every s1eval carve's.

  Every patched name is put back afterwards, also on a failure. Any later read of these fits must run under the same
  patch; a plain zg6r refuses the state (the strict load fails on gnx_w).
- **The base, fixed by ISO's filed verdict, as DEP's** (`zdep.iso_call`, `zdep.base_fit`):
  - **KEEP:** zg6r, ML1's fits `scr-ml1-zg6r-s<k>` (with `-hp` on WQ-hotpotqa), from the same zspk seed-0 encoder;
  - **REPLACE:** zg6r-m, ISO's fits `scr-iso-zg6rm-s<k>`, with its own draw through ISO's own-start patches;
  - **anything else:** no GNX fit, compare or call runs.
- **The fits.** The screen is seed 0 on both splits: `scr-gnx-nx-s0` (WQ-musique) and `scr-gnx-nx-s0-hp` (WQ-hotpotqa).
  Each fit is stamped under the key `gnx` with:
  - the term's record (inputs, columns, parameter counts, layers);
  - its base fit and ISO's verdict;
  - the NEXT block (split, columns, scorer, and each carve's column record sha256);
  - zml1s's stamp (device, environment, gate, code).
- **The compares.** Each GNX fit against the base's fit of its seed and split, through zwq's `run_compare`, as DEP's.

## 3. The reads

DEP's section 3, unchanged: ML1's twelve reads (two splits by six datasets, p@swa on the s1eval carves) and musique's
3-hop (760 of 2,417) and 4-hop (405) slices on both splits. Each slice comes from the fit's musique read file
(`zdep.slice_r5`), whose R@5 over all questions must equal its compare's within 0.00006.

## 4. The rule (fixed now)

LTG's in-domain rule (docs/LTG_NEXT_HOP_GNN.md section 4), unchanged. zgnx calls `zltg.decide_screen` and
`zltg.decide_full`.

- **Deciding (ten):** the eight reads of datasets their split trains on, and the WQ-hotpotqa fits' musique 3-hop and
  4-hop slices.
- **Beside (six):** the four zero-shot reads and the WQ-musique fits' musique slices.
- **The screen (seed 0):** the floor is max(0.0075, 2 x the base's spread over its three seeds). LOSS is a delta at or
  below minus the floor; GAIN is at or above the floor. **NOT_ADOPTED:** any LOSS. **ADOPT_FOR_FULL_RUN:** no LOSS and at
  least one GAIN. **NO_GAIN:** every read WITHIN.
- **The full run (seeds 1 and 2):** LTG's full-run rule.

**INCOMPLETE** holds when:

- a fit, compare or read file is missing, or ISO's verdict is not REPLACE or KEEP;
- a fit's train or read stamp does not name the term (1,920 parameters, 15 inputs, the seven columns, three layers), its
  base fit and ISO's verdict;
- its NEXT block is not its split's seven columns, its train lacks a training carve's columns, or its read lacks an
  s1eval carve's;
- its train and read read columns of different scorers;
- its encoder's start is not its base's, or it ran on another device than its base;
- its trainer's code differs from its base's (stage modules aside);
- the base's seed-0 R@5 in GNX's compare differs from its own stage's, or a read file's R@5 differs from its compare's.

A verdict closes only the form it tested.

## 5. The seed-0 stop, from the start

DEP's seed-0 stop (docs/DEP_GNN_DEPTH.md section 12, `outputs/mp_unified/s0stop.py --stage gnx`):

- `gnx-s0stop` runs after `scr-gnx-nx-s0`. It runs `zgnx.py compare-all` and `zgnx.py calls`, then decides on the four
  in-domain reads of WQ-musique (`zltg.deciding_keys()`): STOP on a LOSS among them; GO on any problem, a missing read, or
  otherwise.
- `scr-gnx-nx-s0-hp` waits on `gnx-s0stop`, and its command is wrapped in `s0stop.py run --stage gnx --`.
- On a STOP the wrapper writes `outputs/gnx/s0stop_skip.json`, and `gnx-calls` gives INCOMPLETE (the WQ-hotpotqa reads are
  missing). The screen's verdict is the NOT_ADOPTED in `outputs/gnx/s0stop.json`.

No number changes: a LOSS among the four gives NOT_ADOPTED whatever the WQ-hotpotqa reads are.

## 6. Reported beside, deciding nothing

- The six beside reads, called the same way.
- DEP's sixteen-read rule on the same numbers (`dep_rule_beside` in `outputs/gnx/calls.json`).
- Each fit's parameter counts, seconds and GPU peak.
- The floor, printed beside each delta.

## 7. What follows (fixed now)

| outcome | what follows |
| --- | --- |
| ADOPT_FOR_FULL_RUN | GNX's input goes into the stack: zg6r with every GNN form whose screen asks for a full run (DEP, LTG, RTR, IRA, GNX), declared in its own file before it runs. The stack's full run decides adoption. No GNX-alone full run is queued before the 6-7 November freeze; GNX's own full run (seeds 1 and 2 on both splits, section 4's full-run rule) may follow after the freeze. |
| NO_GAIN or NOT_ADOPTED | The input as declared closes. The next form, in its own file: the NEXT columns against the GNN's own top five rows. A first pass of the GNN ranks the pool, LTM's split scorer scores each row against that pass's top five, and a second pass reads those columns (an iterative refinement of the anchors). |
| INCOMPLETE | What is missing is fixed in an amendment, before any number is read. |

## 8. Risks, stated before any number

- **Seed 0 only in the screen.** It is a one-fit screen per split; a full run decides any adoption.
- **The GNN may already hold the gain.** The MLP gained without message passing. zg6r's attention already moves mass from
  rrf's top rows to their neighbours, which is what NEXT scores. A NO_GAIN would say the GNN's own layers already find it.
- **The anchors are rrf's.** The columns score rows against the first stage's top five, not against the GNN's ranking,
  which is stronger. Section 7's next form re-anchors them.
- **Train and eval columns differ in kind.** The training rows' columns are out of fold; the eval rows' come from the
  split's full scorer. LTM's MLP read the same difference and adopted.
- **Zero-shot reads may fall.** The rule does not stop on them; they are reported beside, and so is DEP's rule.
- **Memory.** The base's hp fits peaked at 17.17 GiB against the 17.27 GiB cap of `--frac 0.72`. The input adds, per
  encode, the N x 15 input and the hook's sum with l1's output. So the hp fits take LTG's `--frac 0.80` (GPU share 0.83)
  and run alone on the card; the musique-split fits keep 0.40 (share 0.42). On the host, the columns add at most 1.15 GB,
  so the items ask the base's 7 and 12 GB plus 2. A fit that runs out of memory stops INCOMPLETE, and the caps change in
  an amendment before it runs again.
- **Time.** One more input block in the encoder: within a few percent of the base's 4,300 s (WQ-musique) and 6,500 s
  (WQ-hotpotqa) of training.
- **The self-test's model is a toy.** It shows that the input builds, starts as zg6r, trains and restores. It says nothing
  about R@5.

## 9. How it runs

Feeder items on the host:

1. `gnx-selftest` and then `gnx-check-cols`, on the CPU (1 CPU, 4 GB). `check-cols` loads LTM's columns of every carve
   both splits read (each split's training carves and the six s1eval carves), against each carve's cache parts, with one
   scorer per split. It writes `outputs/gnx/cols.json` and exits 1 on any problem, which holds the fits.
2. `gnx-nx-s0`: GPU share 0.42, cap 0.40, 9 GB, after `pkoh-gate`, `iso-verdict`, `ml1start-check`, `readswa-check`,
   `gnx-selftest` and `gnx-check-cols`. It sits ahead of RTR's and IRA's seed-0 fits in the items file. When DEP's
   WQ-hotpotqa fit ends it shares the card with the next ready musique-split fit; RTR's and IRA's wait on DEP's call.
3. `gnx-s0stop` (1 CPU, 4 GB), after `gnx-nx-s0`.
4. `gnx-nx-s0-hp`: GPU share 0.83, cap 0.80, 14 GB, after `gnx-s0stop` (section 5), wrapped. It sits right after LTG's
   WQ-hotpotqa fit.
5. `gnx-compare` and then `gnx-calls`, on the CPU.

```
python outputs/mp_unified/zgnx.py check-cols
python outputs/host_ops/retry_cmd.py --tries 3 --wait 90 --cuda -- python outputs/host_ops/cuda_alloc.py --frac 0.4 --gc 0.9 -- outputs/mp_unified/zgnx.py fit --name scr-gnx-nx-s0
python outputs/mp_unified/s0stop.py check --stage gnx
python outputs/mp_unified/s0stop.py run --stage gnx -- python outputs/host_ops/retry_cmd.py --tries 3 --wait 90 --cuda -- python outputs/host_ops/cuda_alloc.py --frac 0.8 --gc 0.9 -- outputs/mp_unified/zgnx.py fit --name scr-gnx-nx-s0-hp
python outputs/mp_unified/zgnx.py compare-all
python outputs/mp_unified/zgnx.py calls          (outputs/gnx/calls.json and .md)
```

Only `outputs/gnx/calls.json`, `calls.md`, `cols.json` and `s0stop.json` are committed, masked.

The self-test, `python outputs/mp_unified/zgnx.py --selftest`, passed on the laptop's CPU before this commit. It covers:

- the names and pairs, disjoint from ML1's, ISO's, DEP's, LTG's, RTR's, IRA's, ISB's and LTM's; the training and s1eval
  carves each split loads;
- zg6r's toy carve with LTM-format toy columns (zltm's own `toy_cols`):
  - the carve's columns are the files, float16 on the CPU, and the batch's NEXT rows are its questions' rows of them
    (questions out of order, NaN set to 0);
  - the carve refuses no split, another split's columns, another scorer's, and a batch that already holds NEXT;
  - gnx_w is hidden x 15 zeros. At that start the eval forward and the training forward (the same draws) are zg6r's bit
    for bit, with zg6r's parameters drawn at random;
  - gnx_w gets a gradient above 0 from the listwise loss; drawn, it moves the score; the hook's input is cleared after
    each forward;
  - l1 run outside the forward stops, and so does a batch without NEXT;
  - the state loads strictly under the patch, and a plain zg6r refuses it;
  - one epoch of lean_gpu's own `fit_variant` on the patched carve and model, with a finite loss and gnx_w moving;
- the patches (names, stamp with the term's record and the NEXT block, ISO's own start on REPLACE), and that they are put
  back, also on a failure;
- the stamp's refusals: a train without every training carve's columns, and a read without every s1eval carve's;
- that the rule's keys are LTG's (its calls are tested in zltg's self-test).
