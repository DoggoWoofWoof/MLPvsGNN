# CAP: zrckpn's hidden width and epochs, on PKO's pools, on the host's CPU

Declared 4 November 2026 (about 09:20 IST), before any CAP fit. Module: `outputs/mp_unified/zcap.py`. Records:
`outputs/dob/cap/`.

## 1. The question

DOB (docs/DOB_DROP_ONE_BLOCK_CPU.md) found no input block to drop: section 13 keeps topo_STRUCT, and section 6 names the next
CPU form for that case, zrckp's capacity (hidden width and epochs), in its own file. The MLP of record has been zrckpn since
LTM (docs/LTM_NEXT_HOP_MLP.md): zrckp's model on zp1c3's chain carve plus LTM's seven NEXT columns, hidden width 128,
trained for 8 epochs with SWA from epoch 2 (`2e-3:1e-4:0.1:8:2`). The form runs on zrckpn.

**Does a wider zrckpn, or one trained for twice the epochs, rank better than zrckpn by more than the noise of its own seeds,
on the deciding reads?** Nothing else changes.

Round 33 (docs/SCREENS.md, zep; docs/FULL_ROUND33.md) trained zrc, the MLP's base on the old pools, for twice the epochs. Its
screen moved the in-domain reads by under 0.003, and its full run was NOT_ADOPTED (12 October). PKO's pools and LTM's
columns are other inputs, so that result does not close this form. It does make the epoch cell the likelier flat one
(section 7).

## 2. What runs

zrckpn is the base: hidden 128, `2e-3:1e-4:0.1:8:2` (lr : weight decay : dropout : epochs : SWA start). Its three seeds on
both splits (`scr-ltm-zrckpn-s0`, `-s1`, `-s2`, and the same with `-hp`) are fitted and read already, and are not fitted
again. Three cells change the width, the epochs, or both, and nothing else:

| cell | hidden | config | what changes |
| --- | ---: | --- | --- |
| e16 | 128 | `2e-3:1e-4:0.1:16:4` | epochs 8 to 16 and SWA start 2 to 4 (Round 33's doubling) |
| w256 | 256 | `2e-3:1e-4:0.1:8:2` | width 128 to 256 |
| w256e16 | 256 | `2e-3:1e-4:0.1:16:4` | both |

- **What stays zrckpn's.** The arm (zp1c3p's model on the chain carve, with NEXT), PKO's pools under WQ, LTM's NEXT columns
  (the scorer is not fitted again: each fit must have read the split's present scorer), the read (p@swa on the six s1eval
  carves at zrckp's read cap of 65,536 rows), the device (the host's CPU at 6 threads) and the code (every file the base
  fit's stamp names, by sha).
- **How a cell gets in.** `zcap.py` imports zltm, zdob, zdoba3, zdep, zwww, zml1 and zml1s and edits none of them. For the
  length of a CAP call, zltm's fit table holds the 18 cell names (each is zrckpn, a seed and a split); `rmatch.train` takes
  the active cell's `--config` and `--hidden` on its argv (an argv that names either already is refused, as ISB's is);
  zml1s's stamp adds a `cap` block (cell, config, width, base fit) to the train's and the read's records. Every patch is put
  back after, also on a failure. lean_gpu's own defaults must be the base's (checked).
- **Names.** Fits `scr-cap-<cell>-s<seed>` (WQ-musique) and `scr-cap-<cell>-s<seed>-hp` (WQ-hotpotqa). Compares
  `scr-cap-<cell>-vs-zrckpn-s<seed>[-hp]`, each against zrckpn's fit of the same seed and split.
- **Which fits.** The screen: seed 0 of the three cells on both splits (6 fits). The full run: seeds 1 and 2 on both splits
  of the one chosen cell (4 fits). The other cells' seeds 1 and 2 never run: the module refuses them.

## 3. The reads

As DOB's. The own rule reads the twelve reads of `zml1` (WQ-musique and WQ-hotpotqa, each on the six evaluation datasets,
p@swa on the s1eval carves). The six-read rule (6cb8f39, docs/WWW_SCOPE_RULE.md section 2) reads WQ-musique hotpotqa and
2wiki, WQ-hotpotqa musique and 2wiki, and musique's 3-hop and 4-hop slices on WQ-hotpotqa (`zdep.slice_r5`, from the fits'
musique read files, as DEP, LTM and DOB's A3 read them). The base is zrckpn's three seeds, from LTM's compares
(`outputs/screen/scr-ltm-zrckpn-vs-zrckp-s<k>[-hp].json`). A CAP compare must hold the same zrckpn R@5, or every verdict is
INCOMPLETE.

## 4. The rule (fixed now)

**The floor and the call at seed 0.** Per read, the floor is DOB's: `zml1.pair_floor` of zrckpn's spread on the read (the RMS
of its seed-1 and seed-2 differences from its seed 0) on both sides, never below 0.0075. A cell's seed-0 delta (its R@5 minus
zrckpn's seed-0 R@5) is a LOSS at or below minus the floor, a GAIN at or above the floor, and WITHIN otherwise.

**The stop.** After each cell's WQ-musique fit, `zcap.py stop` calls that fit's WQ-musique hotpotqa and 2wiki reads. A LOSS on
either stops the cell: its WQ-hotpotqa fit does not run, and its screen is NOT_ADOPTED under both rules, since a LOSS
decides. A stale record, a problem or a missing read is GO (zwww's seed-0 stop).

**The screen, per cell, under both rules.**
- Own (ML1's rule, twelve reads): a LOSS on any read is NOT_ADOPTED; else a missing read is INCOMPLETE; else a GAIN is
  ADOPT_FOR_FULL_RUN; else NO_GAIN.
- Six-read (`zwww.screen_verdict` on the six reads): the same, on the six.
- Any problem (section 8's checks, or a read file that does not give its compare's R@5) makes every verdict INCOMPLETE and
  chooses nothing.

**The choice.** The six-read rule decides the queue until the freeze. Among the cells whose six-read screen is
ADOPT_FOR_FULL_RUN, the one with the largest mean seed-0 delta on the six reads goes to the full run; a tie goes to the
cheaper cell (e16, then w256, then w256e16). One cell only: a full run costs four fits, and the pick among three is made on
seed 0 alone, so seeds 1 and 2 stay out of the choice and confirm it. When no cell passes, the stage is NO_FULL_RUN, both
rules are NOT_ADOPTED, and the full-run items end at once. A cell that passes the own screen only is filed beside and goes
no further.

**The full run** (`zcap.py calls`, through `zdoba3.full_decide`; `outputs/dob/cap/full.json` and `.md`):
- own: ML1's rule (`zml1.verdict`), the cell's three seeds against zrckpn's three on the twelve reads;
- six-read: `zwww.full_verdict` on the six reads, each called by `zml1.call` at three seeds with `zml1.pair_floor`'s floor.
  ADOPT needs no LOSS among the six and a GAIN on one of the four dataset reads, so a slice GAIN alone cannot adopt;
- INCOMPLETE on any problem.

## 5. Reported beside, deciding nothing

- Each CAP fit's train and read seconds (`ml1_timing.json`) beside zrckpn's: a cost line from a shared host, not a timing
  claim.
- The cells' seed-0 deltas on all twelve reads and the two slices (`screen.md`), and the per-seed deltas of the chosen cell
  (`full.md`). webqsp, metaqa and squad are in the own rule's twelve and outside the six.

## 6. What follows (fixed now)

| outcome | what follows |
| --- | --- |
| six-read ADOPT (full run) | The cell is filed beside zrckpn, and zrckpn stays the MLP of record until the user decides (D2). The GNN gets the matching capacity first, so the MLP holds none that the GNN did not get: ISB (zg6r at 16 epochs, docs/ISB_ZG6RM_16_EPOCHS.md, held until after the freeze) is the epoch rung, and a width form for zg6r is declared in its own file, on the GPU. The cell's forward is timed in the cold path before any latency claim. The next rung (width 512, or 32 epochs) is its own file, after the GNN's. |
| six-read NOT_ADOPTED or NO_FULL_RUN | Capacity is closed for zrckpn on these pools: width 256 and 16 epochs do not move its ranking by more than its seeds' noise on the six reads. Width 512 is not run. The next CPU form is named in its own file. |
| INCOMPLETE | What is missing is supplied; nothing else changes. |

The own verdict is filed beside the six-read verdict and decides after the freeze, with DOB's other full runs.

**The freeze.** Results freeze on 6-7 November. A full run not read by then does not change the MLP of record for the
frozen results; its reads are filed when they land.

A verdict closes only the form it tested.

## 7. Risks, stated before any number

- **The prior.** Round 33's doubling of the epochs was flat in-domain on the old pools. e16 is the likelier flat cell; the
  width cells are the open question.
- **The screen is permissive, and the full run is strict.** A3's one seed-0 GAIN (+0.0154) did not hold at seeds 1 and 2.
  The screen here only picks the cell that is tested again. It adopts nothing.
- **Choosing among three at seed 0.** The chosen cell's seed-0 delta is the largest of three, so it is optimistic. Seeds 1
  and 2 are fresh fits outside the choice, and the floors do not depend on it.
- **Memory and time.** The sizes in section 8 are zrckpn's measured peaks with a margin. Width 256 adds weights, and the main
  MLP's multiply-adds per row become 2.6 times zrckpn's (324, 256, 256, 1 against 324, 128, 128, 1); at 65,536 rows that adds
  little to a peak that the pools and columns set. The first fits measure the cost, and a fit that is killed is retried by
  the wrapper.
- **A wider MLP serves slower.** An adoption changes no latency claim until the cell's forward is timed (section 6).
- **A shared host.** The CPU is not exclusive; seconds beside are not timing numbers.

## 8. How it runs

**What a read fit must be** (`zcap.fit_problems`, in every verdict): zltm's checks on zrckpn's six fits and the cells' (the
device, the threads, the pick, the gate, the NEXT columns and their scorer, the code against DOB's base), and for each cell
its train's config and width, its `cap` block in the train's and the read's stamps, and its trainer's code against zrckpn's
fit of the same seed and split. Any miss makes the verdicts INCOMPLETE.

**Sizes, from zrckpn's measured peaks** (6 CPUs each): 26.77, 26.82 and 27.29 GB on WQ-musique, 30.98, 31.01 and 30.91 GB on
WQ-hotpotqa. The items take 29 GB and 33 GB. Training alone peaked at 11.7 GB (seed 0).

**The queue** (the block sits at the end of the feeder's file, after every pending GPU fit, as A3's did):

| item | CPUs | GB | after | runs |
| --- | ---: | ---: | --- | --- |
| `dob-cap-selftest` | 1 | 3 | | `zcap.py --selftest` |
| `dob-cap-dry` | 1 | 8 | | `zcap.py dry`: zrckpn's six fits through the checks, its floors on every read and slice; writes nothing; no item waits on it |
| `dob-cap-<cell>-s0` | 6 | 29 | selftest | `zcap.py fit --name scr-cap-<cell>-s0` |
| `dob-cap-stop-<cell>` | 1 | 8 | `dob-cap-<cell>-s0` | `zcap.py stop --cell <cell>` |
| `dob-cap-<cell>-s0-hp` | 6 | 33 | `dob-cap-stop-<cell>` | `zcap.py fit --name scr-cap-<cell>-s0-hp` (not run after a stop) |
| `dob-cap-compare` | 1 | 8 | the three `-s0-hp` | `zcap.py compare-all` |
| `dob-cap-screen` | 1 | 8 | compare | `zcap.py screen` |
| `dob-cap-full-s<1,2>` and `-hp` | 6 | 29, 33 | screen | `zcap.py fit-full --seed <k> --split <split>` (nothing when no cell is chosen) |
| `dob-cap-compare-full` | 1 | 8 | the four full fits | `zcap.py compare-all` |
| `dob-cap-calls` | 1 | 8 | compare-full | `zcap.py calls` |

The cells go in the cheaper-first order e16, w256, w256e16. The fits run through `retry_cmd.py --tries 3 --wait 90`, as
LTM's do; a fit that trained and was not read is only read on a retry.

**Cost model (an estimate; the first fits' `ml1_timing.json` replaces it).** zrckpn's fits took 6,236 to 7,072 s on
WQ-musique (seed 0: 2,714 s to train, 3,518 s to read) and 8,835 to 8,949 s on WQ-hotpotqa. e16 doubles only the training.
Width 256 costs at most 2.6 times the multiply-adds of training and read, and how much of those 6,000 s is multiply-adds is
unknown (the read is mostly assembly), so a cell's fit is between 1 and 3 times zrckpn's. At two fits at a time (memory
allows two, and three once FSV's L2 ends) the screen takes about 7 to 20 hours, the chosen cell's full run 4 to 13 hours more,
and the verdict falls between about 23:00 IST on 4 November and 20:00 IST on 5 November, at least 28 hours before the
freeze, if the block is not held behind other items.

**The self-test**, `python outputs/mp_unified/zcap.py --selftest`, passed on the laptop's CPU before this commit. It covers:
- the tables (18 fits, 18 compares, three cells in the cheaper-first order), zltm's fit table put back also on a failure, and
  the refusal of every other fit;
- the argv (the active cell's config and width, once; a named config or width, and another fit, refused), the stamp, and
  rmatch's train and the stamp put back; the fit's whole context as `fit()` enters it (on zltm's fakes: zml1s's table holds
  the cells, the train and read that run are zltm's, rmatch's train takes the cell's config), put back after;
- the seed-0 stop (a LOSS on WQ-musique stops; a stale record is GO);
- the screen under both rules on made-up compares and read files (a slice GAIN alone, a tie to the cheaper cell, no cell
  chosen, a compare off LTM's zrckpn) and the full run (no cell chosen ends it; a chosen cell is called by `zdoba3.full_decide`);
- the dry run (`zcap.py dry`) on made-up compares and read files, with a base seed missing;
- a read fit's records: the cell's config and width, its `cap` block, and the code against the base's.

```
python outputs/mp_unified/zcap.py dry
python outputs/host_ops/retry_cmd.py --tries 3 --wait 90 -- python outputs/mp_unified/zcap.py fit --name scr-cap-e16-s0
python outputs/mp_unified/zcap.py stop --cell e16
python outputs/mp_unified/zcap.py compare-all
python outputs/mp_unified/zcap.py screen
python outputs/mp_unified/zcap.py fit-full --seed 1 --split WQ-musique
python outputs/mp_unified/zcap.py calls
```

## 9. Decisions for the user (the defaults are in force)

- **D1, the host beside the timing track.** Default: CAP's block runs at the feeder's own concurrency, as the standing rule
  asks for work to be queued at all times. Otherwise: hold the block behind the timing track's timed items.
- **D2, what an ADOPT means for the claims.** Default: the cell is filed beside zrckpn, the MLP of record stays, and the GNN's
  matching form runs first (section 6). Otherwise: the user moves the MLP of record to the cell.

## 10. Results of CAP, 5 November 2026 (00:50 IST)

`zcap.py compare-all`, `screen` and `calls` ran on the host on 5 November between 00:48 and 00:56 IST, after the six
seed-0 fits. The records are `outputs/dob/cap/screen.json` and `.md` and `outputs/dob/cap/full.json` and `.md`, with no
problems. All three cells passed the seed-0 stop (no LOSS on WQ-musique's hotpotqa and 2wiki reads), so all six fits ran.

| cell | hidden | config | own (twelve reads) | six-read rule | mean seed-0 delta, six reads | GAIN / LOSS (six) |
| --- | ---: | --- | --- | --- | ---: | --- |
| e16 | 128 | 2e-3:1e-4:0.1:16:4 | NOT_ADOPTED | NO_GAIN | +0.0032 | 0 / 0 |
| w256 | 256 | 2e-3:1e-4:0.1:8:2 | NOT_ADOPTED | NOT_ADOPTED | -0.0089 | 0 / 3 |
| w256e16 | 256 | 2e-3:1e-4:0.1:16:4 | NOT_ADOPTED | NO_GAIN | -0.0029 | 0 / 0 |

- The screen chose **no cell**, so the stage is NO_FULL_RUN (section 4): the four full-run items ended at once, and
  `full.json` files both rules as NOT_ADOPTED.
- The six reads are WQ-musique's hotpotqa and 2wiki and WQ-hotpotqa's musique, 2wiki, musique-3hop and musique-4hop. No
  cell has a GAIN on any of them. w256 has three LOSSes (WQ-hotpotqa musique -0.0117, musique-3hop -0.0167,
  musique-4hop -0.0093); e16 and w256e16 have none.
- On the own rule's twelve reads each cell has one or two LOSSes: e16 on WQ-musique metaqa (-0.0385) and WQ-hotpotqa
  hotpotqa (-0.0085), w256 on WQ-hotpotqa musique (-0.0117), w256e16 on WQ-musique metaqa (-0.0462). w256 also has one
  GAIN (WQ-musique metaqa, +0.0573), which decides nothing.

**What follows (section 6, fixed before the fits).**
- Capacity is closed for zrckpn on these pools: no cell lifts a read of the six, and width 256 lowers three of them.
  Width 512 is not run. zrckpn stays the MLP of record, and no cell's forward is timed.
- No next CPU form is declared before the freeze. The host's CPU goes to the hyperlink carve's reads first, and a form is
  declared in its own file after the freeze.
- A verdict closes only the form it tested: the epochs and the width of this MLP, on these pools, at these seeds.
