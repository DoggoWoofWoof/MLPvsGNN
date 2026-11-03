# GNP: GNX's arm on PFS A3's pair columns, which score every pool row against the first stage's top five, on PKO's pools

Declared 3 November 2026, after PFS amendment A3's five probe records were read (docs/PKO_FEATURE_SCREEN.md section 12)
and before any GNP column or fit. Column builder: `outputs/mp_unified/gnpcols.py`. Stage module:
`outputs/mp_unified/zgnp.py`. They import `zpfp.py`, `ltwdiag.py`, `chndiag.py`, `ltmcols.py`, `zltm.py`, `zgnx.py` and
`s0stop.py` and edit none of them. What the stack does with GNP's verdict is STK amendment A3's, committed before GNP's
screen is read.

## 1. The question

- **What PFS A3 showed.** Section 11 of PFS fixed six free scores of each pool row against rrf's top five rows, from the
  frozen encoder's unit vectors, with no fit and no gold anchor. Section 12 read them on the first questions of each fit
  carve with section 9's probe, the 172 lean columns as the baseline:

  | dataset | lean recall@5 | + pair | d | call |
  | --- | ---: | ---: | ---: | --- |
  | musique (s1fit, 400, all 2-hop) | 0.6500 | 0.6663 | +0.0162 | PAIR_ADDS |
  | hotpotqa (fit, 1,000) | 0.8053 | 0.8223 | +0.0170 | PAIR_ADDS |
  | 2wiki (fit, 1,000) | 0.8085 | 0.8203 | +0.0118 | PAIR_ADDS |
  | squad (fit, 1,000) | 0.9518 | 0.9558 | +0.0040 | PAIR_NOTHING |
  | webqsp (fit, 600) | 0.2919 | 0.3300 | +0.0380 | PAIR_ADDS |

- **The rule that declares GNP.** By PFS section 11, PAIR_ADDS on musique with PAIR_HURTS on none of the five, or
  PAIR_ADDS on at least three of the five, declares the six columns as an input block on the GNN first: GNP, "an arm of
  the GNN of record (zg6r) with its own base and null, read by LTG's in-domain rule". The MLP follows second, on the CPU,
  in its own file. Both conditions hold: musique is PAIR_ADDS, four of the five are, and none is PAIR_HURTS.
- **How these columns differ from GNX's and GNA's.** LTM's columns (GNX) score only the rows linked to one of the top five.
  LTA's (GNA) score every row with a learned scorer. These score every row with no fit at all: the same rule on every carve
  and under both splits.
- **The GNN stays its own model.** The columns are a fixed input of each pool row, made by a rule with no fit from the
  first stage's order and the frozen vectors, before any row is scored. No MLP and no MLP fit enters a GNP fit. A new
  input is a model change under the 13 October rule, so GNP is its own file.
- **So the question is:** do the six pair columns, as GNX's input block, raise the GNN's in-domain R@5 over zg6r's, above
  all on musique?

## 2. What runs

- **The columns** (`gnpcols.py cols`, one item per carve, CPU):
  - For each question, rrf's order is `chndiag.order_of` on the cache's rrf column (descending, ties by pool position),
    and its first five rows are the anchors.
  - Each row's six columns are `zpfp.pair_cols`'s, in its order: `pair_max`, `res_max`, `pair_mean`, `res_mean`, `pair_1`,
    `res_1` (PFS section 11's definitions; the first row reads its second).
  - A question of five rows or fewer takes every row as an anchor, and a row with no other row reads zeros.
  - Each row's c.q must match the cache's dense_cos within 0.002 on all but 1% of the carve's rows (zpfp's identity), or
    the item stops and writes nothing.
  - They are written float16, per cache part, in ltmcols' layout, under `outputs/gnp/cols/<split>/<dataset>/<carve>/`,
    once for each split whose fits read the carve. The arrays are the same under both splits.
  - Every record's `fit_sha256` is one sha256 over the rule's code (gnpcols.py, zpfp.py, ltwdiag.py and chndiag.py), so
    zgnx's one-scorer check holds across a fit's carves only when every carve was written by the same code.
- **The match** (`gnpcols.py match`): on the first 64 questions of PFS's five probe carves, these columns must equal
  zpfp's own batch within 1e-4 before float16 and 2e-3 after, on the same cache records, with both identities passing.
  zpfp takes its rows from lean_gpu's carve and its order from zlink's pool_rank on the within-question z-score of rrf.
  That is rrf's order unless the z-score merges two nearly equal values, which the match would show as a failure. It writes `outputs/gnp/match.json` and exits 1 otherwise, which holds
  the fits.
- **The arm is GNX's** (GNX section 2), with the block six columns wide:
  - the input, each row's [raw, z, 1], 13 numbers;
  - the term, gnx_w (hidden x 13, zeros at the start), added to the output of zg6r's l1 through a forward hook;
  - the start, zg6r's bit for bit;
  - the base, fixed by ISO's filed verdict (KEEP: ML1's zg6r fits), with the trainer, card, epochs, KT input, read cap
    and p@swa read.
- **The names.** For the length of a GNP call:
  - ltmcols' root, stage, declaration, column names and width are GNP's (`zgnp.on_cols`). zgnx's carve then loads GNP's
    columns through `zltm.load_next`, which checks each record and each part's sha256 and refuses LTM's and LTA's.
  - zgnx's fits, declaration, stage, stamp key (`gnp`), output root (`outputs/gnp`), fit and pair names, code, term record,
    NEXT block, IN (13) and WIDTH (6) are GNP's.
  - The NEXT block names this declaration and GNP's root. The code adds zgnp.py, gnpcols.py and zpfp.py to GNX's.
  - All of it is put back afterwards, also on a failure.
- **The fits.** The screen is seed 0 on both splits: `scr-gnp-px-s0` (WQ-musique) and `scr-gnp-px-s0-hp` (WQ-hotpotqa).
  The compares are `scr-gnp-px-vs-base-s0` and `-s0-hp`, against the base's fit of the same seed and split.

## 3. The reads

GNX's section 3, unchanged.

## 4. The rule (fixed now)

GNX's section 4, unchanged: LTG's in-domain rule (`zltg.decide_screen`, `zltg.decide_full`) and GNX's INCOMPLETE
conditions. INCOMPLETE also holds when any of these is true of a read fit (`zgnp.gnp_problems`):

- its train's or read's NEXT block does not name this declaration;
- its train's term does not take 13 inputs of the six columns;
- its train's code is not this GNP's: zgnp.py, gnpcols.py and zpfp.py;
- `outputs/gnp/match.json` is missing or did not pass.

## 5. The seed-0 stop

s0stop's stop, with GNP's entry patched into its stage table in the process (`zgnp.py s0stop-check` and `s0stop-run`). It
decides on LTG's four in-domain WQ-musique reads:

- STOP on a LOSS among them;
- GO on any problem or missing read, and otherwise.

The WQ-hotpotqa fit's command is wrapped in `zgnp.py s0stop-run --`. `zgnp.py gate` is s0stop's gate under the same entry.

## 6. Reported beside, deciding nothing

- The six beside reads, and DEP's sixteen-read rule on the same numbers.
- PFS A3's five probe records (each recall@5 difference and call).
- GNX's and GNA's filed screens.
- The parameter counts: the term is 128 x 13 = 1,664 parameters (GNX's is 1,920), 181,882 in all at hidden 128.
- Each carve's column record: the gold rows' and the other rows' mean of each column, and the questions of five rows or
  fewer.

## 7. What follows (fixed now)

| outcome | what follows |
| --- | --- |
| ADOPT_FOR_FULL_RUN | The stack takes the pair columns as STK amendment A3 states. GNP's own full run (seeds 1 and 2 on both splits, section 4's full-run rule) is held by `stk-own-gnp`. The next form, in its own file: the same six columns against the GNN's own top five after a first pass (two-pass refinement). |
| NO_GAIN or NOT_ADOPTED | The pair columns close as an encoder input for the GNN. The next form, in its own file: the pair scores as message weights from the first stage's top five inside message passing. |
| INCOMPLETE | What is missing is fixed in an amendment, before any number is read. |

In every case the MLP's form is its own file, on the CPU: zrckpn's NEXT input with the six pair columns beside it.

## 8. Risks, stated before any number

- **Seed 0 only in the screen.** A full run decides any adoption.
- **The probe is not the GNN.** PFS's logistic probe has no interactions and no message passing. The GNN may already carry
  what the columns add, through its edges. squad's probe gain was +0.004 (PAIR_NOTHING).
- **musique's probe read was 2-hop only.** Its 3-hop and 4-hop questions were not in PFS's 400.
- **2wiki's single columns point the wrong way beyond the top five** (z-scored AUC 0.37 to 0.48). The probe gained there
  only beside the lean columns.
- **float16.** The columns are stored as float16, whose spacing is 2^-10 below 2. The match bounds the difference from
  zpfp's float32 columns at 2e-3.
- **The columns are not all built when this is committed.** `gnp-check-cols` waits on every `gnp-cols-*` item, on
  `gnp-match` and on `gnp-selftest`, and holds the fits until each carve both splits read loads.
- **Memory and time** are GNX's (GNX section 8) with a narrower block: 12 bytes per row on the CPU instead of 14.
- **The self-tests use toys.** They show that the columns are the rule's, question by question, that they load under GNP's
  names only, that the input builds 13 wide, that the start is zg6r's, that the term trains and that everything is put
  back. They say nothing about R@5.

## 9. How it runs

The feeder items on the host:

1. **`gnp-selftest-cols`** (`gnpcols.py --selftest`), then the eleven **`gnp-cols-<dataset>-<carve>`** items (2 CPUs each;
   memory from LTA's and LTM's measured column items), and **`gnp-match`** (2 CPUs, 10 GB). They go before the CPU queue's
   held head, so they start as memory frees.
2. **`gnp-selftest`** (`zgnp.py --selftest`), then **`gnp-check-cols`** (1 CPU and 4 GB each). `gnp-check-cols` runs after
   every `gnp-cols-*` item, `gnp-match` and `gnp-selftest`. It writes `outputs/gnp/cols.json` and exits 1 on any problem,
   which holds the fits.
3. **`gnp-px-s0`**: GPU share 0.42 (cap 0.40), 9 GB, right after `gna-nx-s0` in the queue so the two share the card. It
   waits on `pkoh-gate`, `iso-verdict`, `ml1start-check`, `readswa-check`, `gnp-selftest` and `gnp-check-cols`.
4. **`gnp-s0stop`** (1 CPU, 4 GB), after `gnp-px-s0`.
5. **`gnp-px-s0-hp`**: GPU share 0.83 (cap 0.80), 14 GB, after `gnp-s0stop`, wrapped, right after `gna-nx-s0-hp` in the
   queue.
6. **`gnp-compare`**, **`gnp-calls`** and **`gnp-gate`**, on the CPU. `gnp-gate` runs `gate --want ADOPT_FOR_FULL_RUN`.
7. **The full run**, held by `stk-own-gnp` (STK amendment A3):
   - `gnp-px-s1`, `gnp-px-s2`, `gnp-px-s1-hp` and `gnp-px-s2-hp`, each with its seed-0 fit's size and command, after
     `gnp-gate` and `stk-own-gnp`;
   - then `gnp-compare-full` and `gnp-calls-full`.

```
python outputs/mp_unified/gnpcols.py cols --dataset musique --carve s1fit --host
python outputs/mp_unified/gnpcols.py match --host
python outputs/mp_unified/zgnp.py check-cols
python outputs/host_ops/retry_cmd.py --tries 3 --wait 90 --cuda -- python outputs/host_ops/cuda_alloc.py --frac 0.4 --gc 0.9 -- outputs/mp_unified/zgnp.py fit --name scr-gnp-px-s0
python outputs/mp_unified/zgnp.py s0stop-check
python outputs/mp_unified/zgnp.py s0stop-run -- python outputs/host_ops/retry_cmd.py --tries 3 --wait 90 --cuda -- python outputs/host_ops/cuda_alloc.py --frac 0.8 --gc 0.9 -- outputs/mp_unified/zgnp.py fit --name scr-gnp-px-s0-hp
python outputs/mp_unified/zgnp.py compare-all
python outputs/mp_unified/zgnp.py calls          (outputs/gnp/calls.json and .md)
python outputs/mp_unified/zgnp.py gate --want ADOPT_FOR_FULL_RUN
```

Only `outputs/gnp/calls.json`, `calls.md`, `cols.json`, `match.json` and `s0stop.json` are committed, masked. The column
records and arrays are not.

Both self-tests passed on the laptop's CPU before this commit.

- **`gnpcols.py --selftest`** covers:
  - the carves each split reads, and the rule's code sha;
  - one question: zpfp's rule above five rows, every row an anchor at five or fewer, zeros alone;
  - toy carves: the written columns equal the rule's, question by question, in float16; the batch size changes nothing;
    both splits' records hold the same parts and scorer, with the right training flags;
  - `zltm.load_next` takes them under GNP's names in ltmcols and refuses them under LTM's;
  - the identity stops a node store that is not the cache's, and writes nothing;
  - the match's comparison.
- **`zgnp.py --selftest`** covers:
  - **the names**: GNP's fits are disjoint from ML1's, ISO's, DEP's, LTG's, GNX's, GNA's, LTA's, LTM's, RTR's, IRA's and
    ISB's;
  - **the patches**: every patched name, IN and WIDTH included, is GNP's inside a call and put back after it, also on a
    failure;
  - **the columns**: gnpcols' columns load inside the patches, from ltmcols' root there, and stop outside them; LTM's
    columns stop inside; the NEXT block names this declaration and the rule's sha;
  - **GNX's toy term under GNP's patches**, with six-column toy files: gnx_w is hidden x 13; at gnx_w = 0, zg6r's forwards
    are unchanged bit for bit; the term trains in lean_gpu's loop; the term record is 1,664 parameters, 181,882 in all;
  - **the stamp**: its NEXT block is GNP's with GNP's term, and it refuses a fit without columns;
  - **what follows**, as section 7 states it.

## 10. Amendment A1, 3 November 2026, after PFS amendment A4's calls and before any GNP fit

**Why.** PFS amendment A4 (docs/PKO_FEATURE_SCREEN.md section 13) tested the next form named in section 7 for ADOPT_FOR_FULL_RUN. That form uses the same six columns against the top five of a first pass. A4 made the first pass as a fixed probe: the logistic probe on lean + P5, crossfit by question parity. It then read R2, the six columns against that pass's own top five, beside P5. Its rule, fixed before any number: R2_NOTHING on at least three of the five closes re-anchoring on a first round for the pair columns. GNP's two-pass next form is then withdrawn by a GNP amendment before GNP's verdict, and after ADOPT_FOR_FULL_RUN GNP's next form is its message-weight form. Three reads already give R2_NOTHING. The other two cannot change the count:

| dataset | questions | recall@5, lean + P5 | lean + P5 + R2 | d2 | call | first round's top five kept from rrf's |
| --- | --- | --- | --- | --- | --- | --- |
| hotpotqa | 1,000 | 0.8223 | 0.8203 | -0.0020 | R2_NOTHING | 0.595 |
| squad | 1,000 | 0.9558 | 0.9568 | +0.0010 | R2_NOTHING | 0.757 |
| 2wiki | 1,000 | 0.8203 | 0.8244 | +0.0041 | R2_NOTHING | 0.426 |
| musique (first 400, all 2-hop) | 400 | 0.6663 | 0.6675 | +0.0012 | R2_NOTHING; counted under section 14's rule once its 3-hop and 4-hop read is in | 0.706 |

Every read's machinery check is exact. Each read reproduces A3's lean and lean + P5 recall@5 to 0.0. The first round changes between a quarter (squad) and more than half (2wiki) of the anchors, and the six columns read against the new anchors add nothing beside P5. Section 13's full table, with musique's 3-hop and 4-hop read (section 14) and webqsp, is filed in its own section of that file.

**What changes.**
- **Section 7, the ADOPT_FOR_FULL_RUN row.** Its next form is now the pair scores as message weights from the first stage's top five inside message passing (the NO_GAIN row's form), in its own file. The two-pass form is withdrawn.
- **Section 7, the NO_GAIN or NOT_ADOPTED row,** is unchanged. So after either verdict, GNP's next form is the message-weight form.
- **Nothing else changes:** not zgnp.py, not gnpcols.py, not GNP's items, its rule or its reads, and not STK amendment A3's use of the columns.
