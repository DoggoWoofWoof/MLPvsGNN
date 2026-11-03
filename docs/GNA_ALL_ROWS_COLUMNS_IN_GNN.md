# GNA: GNX's arm on LTA's NEXT columns, which score every pool row, on PKO's pools

Declared 3 November 2026, after LTA's two musique column records were read and before any GNA fit. Stage module:
`outputs/mp_unified/zgna.py`. It imports `zgnx.py`, `zlta.py`, `ltacols.py` and `s0stop.py` and edits none of them. STK's
amendment A2 (docs/STK_GNN_STACK.md section 11), committed with this file, lets the stack's gnx slot read these columns.

## 1. The question

- **Where GNX's columns stop.** GNX (docs/GNX_NEXT_COLUMNS_IN_GNN.md) gives zg6r's encoder LTM's seven NEXT columns. LTM
  scores a pool row only when it is linked to one of the first stage's top five rows. On musique that is 0.045 to 0.047 of
  the rows; every other row's columns are empty.
- **What LTA scores.** LTA (docs/LTA_ALL_ROWS_NEXT_MLP.md) refits LTW's bil scorer per split on A lists over every row
  outside the top five, linked or not, with LTM's folds. Its columns have LTM's seven names and format and score every
  such row.
- **What LTA's musique records show** (WQ-hotpotqa, the split musique trains under; `outputs/lta/cols`, not filed). The
  A case is the first top-five gold's row, ranked among every row outside the top five. T2 is the share of A cases
  ranked in the top two:

  | carve | A cases | bil T2 | rrf T2 | bil T2, unlinked targets | rrf T2, unlinked targets |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | s1eval | 2,547 | 0.179 | 0.067 | 0.084 | 0.070 |
  | s1fit | 3,798 | 0.174 | 0.066 | 0.096 | 0.079 |

- **The condition for writing this file.** It was set after the s1eval record was read and before the s1fit record was:
  on both carves, bil's T2 on unlinked targets is at least rrf's plus 0.01. Both carves meet it (0.084 against 0.080, and
  0.096 against 0.089).
- **The GNN stays its own model.** The columns are a fixed input of each pool row, made by LTA's scorer before any row is
  scored. No MLP and no MLP fit enters a GNA fit.
- **So the question is:** do LTA's columns, as GNX's input block, raise the GNN's in-domain R@5 over zg6r's, above all on
  musique and its 3-hop and 4-hop questions?

## 2. What runs

- **The arm is GNX's** (GNX section 2), unchanged:
  - the input, each row's [raw, z, 1], 15 numbers;
  - the term, gnx_w (hidden x 15, zeros at the start), added to the output of zg6r's l1 through a forward hook;
  - the start, zg6r's bit for bit;
  - the base, fixed by ISO's filed verdict (KEEP: ML1's zg6r fits), with the trainer, card, epochs, KT input, read cap
    and p@swa read.
- **The columns are LTA's**, from `outputs/lta/cols/<split>/<dataset>/<carve>/`:
  - For the length of a GNA call, ltmcols' root, stage and declaration are LTA's (`zlta.on_cols`). zgnx's carve then
    loads them through `zltm.load_next`, which checks each record and each part's sha256 (GNX section 2) and refuses
    LTM's records.
  - The training carves' columns are cross-fitted with LTM's folds. The s1eval carves' columns come from the split's
    scorer, fit on its training carves.
- **The names.** zgnx's fits, declaration, stage, stamp key (`gna`), output root (`outputs/gna`), fit and pair names, code,
  term record and NEXT block are GNA's for the call. They are all put back afterwards, also on a failure.
  - The NEXT block names LTA's declaration and root.
  - The code adds this file's module, `zlta.py` and `ltacols.py` to GNX's.
- **The fits.** The screen is seed 0 on both splits: `scr-gna-nx-s0` (WQ-musique) and `scr-gna-nx-s0-hp` (WQ-hotpotqa).
  The compares are `scr-gna-nx-vs-base-s0` and `-s0-hp`, against the base's fit of the same seed and split.

## 3. The reads

GNX's section 3, unchanged.

## 4. The rule (fixed now)

GNX's section 4, unchanged: LTG's in-domain rule (`zltg.decide_screen`, `zltg.decide_full`) and GNX's INCOMPLETE
conditions. INCOMPLETE also holds when either of these is true of a read fit:

- its train's or read's NEXT block does not name LTA's declaration;
- its train's code is not this GNA's: zgna.py, zlta.py and ltacols.py (`zgna.gna_problems`).

## 5. The seed-0 stop

This is s0stop's stop, with GNA's entry patched into its stage table in the process (`zgna.py s0stop-check` and
`s0stop-run`). It decides on LTG's four in-domain WQ-musique reads:

- STOP on a LOSS among them;
- GO on any problem or missing read, and otherwise.

The WQ-hotpotqa fit's command is wrapped in `zgna.py s0stop-run --`. `zgna.py gate` is s0stop's gate under the same
entry.

## 6. Reported beside, deciding nothing

- The six beside reads, and DEP's sixteen-read rule on the same numbers.
- LTA's report (each carve's A-case T2 by scorer, linked and unlinked apart).
- GNX's filed screen.
- The parameter counts.

## 7. What follows (fixed now)

| outcome | what follows |
| --- | --- |
| ADOPT_FOR_FULL_RUN | The stack's gnx slot passes (STK section 11). It reads LTA's columns when GNX's screen does not pass, or when GNA's mean seed-0 delta over STK's six WQ-hotpotqa reads is larger than GNX's; LTM's otherwise. GNA's own full run (seeds 1 and 2 on both splits, section 4's full-run rule) is released by `zstk.py own-gate --stage gna`, under STK's sections 3 and 6. |
| NO_GAIN or NOT_ADOPTED | LTA's columns close as an encoder input for the GNN. The next form, in its own file: the next-hop term inside message passing over every pool row against the first stage's top five, which is LTG's term on unlinked rows too. |
| INCOMPLETE | What is missing is fixed in an amendment, before any number is read. |

## 8. Risks, stated before any number

- **Seed 0 only in the screen.** A full run decides any adoption.
- **The learned scorer is not the strongest on unlinked rows.** On musique's A cases with unlinked targets, ltwdiag's free
  scores rank higher than bil:

  | carve | pair | res | bil |
  | --- | ---: | ---: | ---: |
  | s1eval | 0.113 | 0.104 | 0.084 |
  | s1fit | 0.144 | 0.128 | 0.096 |

  GNA tests bil's columns. The free scores, against rrf's top five with no gold anchor, are PFS amendment A3's question
  (docs/PKO_FEATURE_SCREEN.md section 11).
- **What was seen when this file was written:**
  - **GNX's WQ-musique seed-0 fit** (08:08 IST, `outputs/gnx/s0stop.json`) gave GO. Its four deciding reads are WITHIN,
    from +0.0003 (squad) to +0.0015 (2wiki). Beside them, zero-shot musique is +0.0147 (GAIN) and its 4-hop slice +0.0173
    (GAIN). No WQ-hotpotqa read of GNX had been seen.
  - **PFS A3's first two probe records** (musique and hotpotqa, not filed) are PAIR_ADDS.
- **LTA's columns are not all built.** `gna-check-cols` waits on every one of LTA's column items and holds the fits until
  each carve both splits read loads.
- **Memory and time** are GNX's (GNX section 8). The columns are the same width and type, and only the values of the
  rows LTM left empty change.
- **The self-test's model is a toy.** It shows that LTA's columns load, the input builds, the start is zg6r's, the term
  trains and everything is put back. It says nothing about R@5.

## 9. How it runs

The feeder items on the host go right after GNX's WQ-hotpotqa fit, so GNX's screen is read first:

1. **`gna-selftest`**, then **`gna-check-cols`** (1 CPU and 4 GB each). `gna-check-cols` runs after every `lta-cols-*`
   item. It writes `outputs/gna/cols.json` and exits 1 on any problem, which holds the fits.
2. **`gna-nx-s0`**: GPU share 0.42 (cap 0.40), 9 GB. It waits on `pkoh-gate`, `iso-verdict`, `ml1start-check`,
   `readswa-check`, `gna-selftest` and `gna-check-cols`.
3. **`gna-s0stop`** (1 CPU, 4 GB), after `gna-nx-s0`.
4. **`gna-nx-s0-hp`**: GPU share 0.83 (cap 0.80), 14 GB, after `gna-s0stop`, wrapped.
5. **`gna-compare`**, **`gna-calls`** and **`gna-gate`**, on the CPU. `gna-gate` runs `gate --want ADOPT_FOR_FULL_RUN`.
6. **The full run**, held by `stk-own-gna` (STK section 11):
   - `gna-nx-s1`, `gna-nx-s2`, `gna-nx-s1-hp` and `gna-nx-s2-hp`, each with its seed-0 fit's size and command, after
     `gna-gate` and `stk-own-gna`;
   - then `gna-compare-full` and `gna-calls-full`.

```
python outputs/mp_unified/zgna.py check-cols
python outputs/host_ops/retry_cmd.py --tries 3 --wait 90 --cuda -- python outputs/host_ops/cuda_alloc.py --frac 0.4 --gc 0.9 -- outputs/mp_unified/zgna.py fit --name scr-gna-nx-s0
python outputs/mp_unified/zgna.py s0stop-check
python outputs/mp_unified/zgna.py s0stop-run -- python outputs/host_ops/retry_cmd.py --tries 3 --wait 90 --cuda -- python outputs/host_ops/cuda_alloc.py --frac 0.8 --gc 0.9 -- outputs/mp_unified/zgna.py fit --name scr-gna-nx-s0-hp
python outputs/mp_unified/zgna.py compare-all
python outputs/mp_unified/zgna.py calls          (outputs/gna/calls.json and .md)
python outputs/mp_unified/zgna.py gate --want ADOPT_FOR_FULL_RUN
```

Only `outputs/gna/calls.json`, `calls.md`, `cols.json` and `s0stop.json` are committed, masked.

The self-test, `python outputs/mp_unified/zgna.py --selftest`, passed on the laptop's CPU before this commit. It covers:

- **The names.** GNA's fits are disjoint from ML1's, ISO's, DEP's, LTG's, GNX's and LTA's.
- **The patches.** Every patched name is GNA's inside a call and put back after it, also on a failure.
- **The columns.**
  - Toy columns stamped as LTA's load inside the patches, from ltmcols' root there, and stop outside them.
  - LTM-stamped columns stop inside.
  - The NEXT block names LTA's declaration.
- **GNX's toy term under GNA's patches.**
  - At gnx_w = 0, zg6r's forwards are unchanged bit for bit.
  - The term trains in lean_gpu's loop.
  - The stamp's NEXT block is LTA's with GNA's term, and the stamp refuses a fit without columns.
- **What follows**, as section 7 states it.
