# STK: the GNN forms that pass their screens, stacked on the GNN of record, on PKO's pools

Declared 3 November 2026, before any DEP, LTG, RTR, IRA or GNX verdict. This file fixes the members, the fits, the rule and
what follows. The stage module, its self-test, its items and how the members' code composes in one model come in an
amendment of this file (section 8), committed before any stack fit runs.

## 1. The question

On 3 November the user asked to "finish these any faster, and iterate much faster", and to "close the gaps that we have no
matter what". Results freeze on 6-7 November.

Five GNN forms are screened at seed 0 on both WQ splits, on zg6r, the GNN of record (ISO's verdict KEEP):

| form | what it adds to zg6r | file |
| --- | --- | --- |
| DEP | a fourth message-passing layer | docs/DEP_GNN_DEPTH.md |
| LTG | LTW's learned next-hop term inside each layer's message passing | docs/LTG_NEXT_HOP_GNN.md |
| RTR | a second round of its layers that goes back to round one's top candidates | docs/RTR_GO_BACK_ROUNDS.md |
| IRA | RTR's two rounds, with four alignment slots between them | docs/IRA_ITERATIVE_ALIGNMENT.md |
| GNX | LTM's seven NEXT columns as one more input of its encoder | docs/GNX_NEXT_COLUMNS_IN_GNN.md |

A form's own full run is four fits, seeds 1 and 2 on both splits. The two WQ-musique fits share the card for about 3 hours,
and each WQ-hotpotqa fit has it alone for about 4.5 hours. That is about 12 hours of the card per form. The screens still to
run need about a day of the card. Three or four passing forms' own full runs, one after another, would then need one and a
half to two days more. They would end at the freeze, with no two forms tried on one GNN.

STK asks one question in their place: do the forms that pass add up on one GNN? Its full run is the GNN's one full run
before the freeze.

## 2. The members (fixed now)

- **The base** is zg6r, by ISO's verdict, with ML1's zg6r fits on PKO's pools as its three seeds. ISB's zg6r-m16 waits until
  after the freeze (docs/ISB_ZG6RM_16_EPOCHS.md section 9), so the base does not change before then.
- **The members** are the forms among DEP, LTG, RTR, IRA and GNX whose screen verdict is ADOPT_FOR_FULL_RUN. The verdict is
  the one `python outputs/mp_unified/s0stop.py gate --stage <stage>` gives, so a seed-0 STOP counts as NOT_ADOPTED.
- **RTR and IRA.** IRA's arm holds RTR's two rounds. When both pass, IRA is the member and RTR's rounds come in it. RTR and
  IRA run only when DEP's screen is NO_GAIN or NOT_ADOPTED, so DEP is never in a stack with either.
- **Each member enters as it was screened:** its own arm, settings and code, unchanged.
- **When the members are fixed:** once every screen has a verdict, or will not run (RTR and IRA when DEP's screen passes). A
  screen with no verdict at 12:00 IST on 5 November is left out.

## 3. What runs (fixed now)

| members | what runs before the freeze |
| --- | --- |
| none | Nothing. zg6r stays the GNN of record, and each form's NO_GAIN or NOT_ADOPTED row applies as its file declares. |
| one | That form's own full run: its held items (seeds 1 and 2 on both splits) are released, and its own full-run rule decides. GNX queued none, so section 8's amendment adds GNX's four. |
| two or more | The stack, zg6r with every member. First its screen: seed 0 on both splits. Then, on ADOPT_FOR_FULL_RUN, seeds 1 and 2 on both splits. |

The stack's fits keep everything else from the members' screens:

- PKO's pools and the WQ splits' carves (zwq);
- ML1's zg6r fits as the base;
- the host's card and the trainer;
- the read of p@swa only (readswa).

Its WQ-hotpotqa seed-0 fit has the seed-0 stop: it does not run when the WQ-musique seed-0 fit already decides NOT_ADOPTED.

## 4. The rule (fixed now)

The five forms use two rules:

- DEP, RTR and IRA use DEP's sixteen-read rule (docs/DEP_GNN_DEPTH.md section 4).
- LTG and GNX use LTG's in-domain rule (docs/LTG_NEXT_HOP_GNN.md section 4).

The stack is read under every rule its members use, so each member keeps the rule it was screened under.

- **Screen:** ADOPT_FOR_FULL_RUN when each of those rules gives ADOPT_FOR_FULL_RUN. NOT_ADOPTED when one gives NOT_ADOPTED.
  NO_GAIN otherwise.
- **Full run:** ADOPT when each of those rules' full-run calls is ADOPT. NOT_ADOPTED otherwise.
- **INCOMPLETE:** any member's INCOMPLETE condition, applied to the stack's fits, or a fit whose train or read stamp does not
  name every member and the base. Where two members' conditions disagree (LTG's stamp names three layers, DEP's four), section
  8's amendment says which holds, before any stack fit runs.

## 5. Beside, deciding nothing

- The calls of a rule that no member uses.
- Each member's own seed-0 deltas, from its screen, beside the stack's.

## 6. What follows (fixed now)

| outcome | what follows |
| --- | --- |
| full run ADOPT | The stack is the GNN of record for the frozen results. Claim 1 (C1R's intervals) and Claim 3 (cold timing) are measured again on it. |
| screen NO_GAIN or NOT_ADOPTED, or full run NOT_ADOPTED | One passing form gets its own full run: its held items are released. It is the form with the largest mean seed-0 delta over the six WQ-hotpotqa reads that LTG's rule decides on: webqsp, squad, musique and 2wiki, and musique's 3-hop and 4-hop slices. A tie goes to the earlier form in the order DEP, LTG, RTR, IRA, GNX. |
| INCOMPLETE | What is missing is fixed in an amendment, before any number is read. |

The pick reads only WQ-hotpotqa reads, where musique trains. No WQ-hotpotqa read of any form had been seen when this file
was written (section 7). With one member, the outcomes are that form's own, as its file declares.

After the freeze, in amendments that add or release items only:

- each member's own full run;
- the stack without each member in turn, to show what each one adds;
- each member's next form, as its file names it.

**The freeze.** A full run not read by 6-7 November does not change the GNN of record for the frozen results. Its reads are
filed when they land.

## 7. What was seen when this file was written

`dep-s0stop` and `ltg-s0stop` read DEP's and LTG's WQ-musique seed-0 fits at 02:40 IST on 3 November. Both gave GO. In each,
every deciding read is WITHIN:

- DEP's eight, with deltas from -0.0081 (webqsp) to +0.0070 (musique's 3-hop slice);
- LTG's four, from +0.0016 (hotpotqa) to +0.0070 (webqsp).

The records are `outputs/dep/s0stop.json` and `outputs/ltg/s0stop.json`. No WQ-hotpotqa read of any form had been seen, and
no read of RTR, IRA or GNX.

## 8. Not yet declared

These come in an amendment of this file, committed before any stack fit runs, and change nothing in sections 1 to 7:

- how the members' code composes in one model, including where two members touch the same part;
- the stage module, its self-test and its items;
- the gate that fixes the members by section 2.

Amendment A1 (section 10) declares all three.

## 9. The queue now (host)

- The members' own full-run fits wait on `hold-full-for-stack`, a name no job has. The feeder keeps an item whose dependency
  it does not know waiting, and does not drop it.
  - The fits held are `-s1`, `-s2`, `-s1-hp` and `-s2-hp` of DEP (`dep-l4-*`), LTG (`ltg-nh-*`), RTR (`rtr-r2-*`) and IRA
    (`ira-k4-*`).
  - Their `-compare-full` and `-calls-full` items wait on those fits.
  - GNX queued no full run.
- `iso-zg6z-s0-hp`, ISO's report-only fit, waits on `hold-isoz-hp-after-stack` (docs/ISO_ZG6R_OWN_START.md section 14), so it
  does not take the card ahead of the stack's fits.
- A hold is released by an amendment that changes only those items' dependencies.

Amendment A1 (section 10.4) replaces `hold-full-for-stack` with each stage's own-run gate.

## 10. Amendment A1 (3 November)

Committed before any stack fit runs and before any member's screen verdict. It declares what section 8 lists and changes
nothing in sections 1 to 7.

The stage module is `outputs/mp_unified/zstk.py`. It imports zml1s, zg6r, zdep, zltg, zrtr, zira, zgnx and s0stop and edits
none of them. Every patch it makes is in-process, and it is put back afterwards, also on a failure.

### 10.1 How the members compose

The model is zg6r's. Each member enters through its own module's patch. The patches are entered in this order, each one
wrapping what the one before it left:

1. IRA's model (`zira.on_model`), or RTR's rounds (`zrtr.on_rounds`) when IRA is not a member;
2. LTG's term in each layer, and its next-hop loss (`zltg.on_term`, `zltg.on_aux`);
3. GNX's model and carve (`zgnx.on_model`, `zgnx.on_carve`).

DEP is not a patch. With DEP, zg6r is imported with four layers (`zdep.import_deep`, in a process of its own). RTR and IRA are
never in a stack with DEP (section 2), and zstk refuses that set.

Where two members touch the same part:

| members | part | how it composes |
| --- | --- | --- |
| DEP and LTG | the layers | LTG's term is in each of the stack's four layers. For the fit, zltg's per-layer names and counts are set to zg6r's layer count, and the first three layers' names are zltg's own. The term has 75,276 parameters at four layers and 68,745 at three. |
| LTG with RTR or IRA | the rounds | The rounds call zg6r's layers once a round. In the stack each call is LTG's term layers, so the term is in each layer of both rounds, with the same parameters in both, like the layers' own. The term's inputs (P, Q, the out-degrees, the flags) are taken once a forward, on the kept edges, because none of them depends on the state. In training, the next-hop loss reads the last round's score, as zltg's does after its one round. |
| GNX with any | the encoder | GNX's NEXT input enters zg6r's encoder at its first linear layer. No other member touches the encoder. |

Each member's added parameters start as they did in its own screen: LTG's `ltg_U`, RTR's `rtr_e`, IRA's `ira_E` and GNX's
`gnx_w` at zero. So a stack starts as the same set without LTG and GNX, bit for bit (section 10.5).

### 10.2 Where the members' conditions disagree (section 4)

- **The layer count.** The stack's count holds: four with DEP, three otherwise. Each member's condition on its own record is
  applied to the stack's stamp at that count:
  - DEP's four layers;
  - LTG's term, with its parameters at that count (75,276 with DEP), its rank, axes, basis and lambda;
  - RTR's and IRA's rounds, with IRA's slots and Sinkhorn steps;
  - GNX's input and columns.

  The stamp must name the fixed members, the stack's layer count and one part for each member. This answers section 4's
  example: with DEP and LTG, a stack fit whose stamp names three layers is INCOMPLETE.
- **A read that two rules both decide.** DEP's sixteen-read rule and LTG's in-domain rule share ten reads. For each shared
  read both rules must give the same row: the same values and the same call. They should, because the per-read call is one
  formula in zdep and zltg. A shared read where they differ makes the screen INCOMPLETE.
- **"As it was screened" (section 2).** Each member's module in a stack fit's train stamp must be the one named by the train
  stamps of both of that member's seed-0 fits, and both of those fits must be read.

  One fit is exempt: LTG's WQ-musique seed-0 fit (`scr-ltg-nh-s0`, item `ltg-nh-s0-b`). It started before commit 24385d8
  changed zltg.py, so it names zltg.py's earlier sha256,
  `4b1657818d6f6641085821e6e7f4323d13ff240751dee6c9d0c4c63ed562e2d9`. That commit's change touches the read alone: a fit's
  read keeps p@swa only (readswa), and training is unchanged. That one fit may name either sha. Every other fit must name
  the stack's.
- **The other conditions** are each member's own, applied to the stack's fits:
  - the base fit and ISO's verdict;
  - the encoder's start;
  - one device, with zg6r's rows cap;
  - the trainer's code, against the base's fit of the same seed and split;
  - with LTG, at least one next-hop loss step in training;
  - with GNX, the NEXT block: the split, the columns, one scorer for train and read, every training carve in the train and
    every s1eval carve in the read;
  - with RTR or IRA, DEP's call under RTR's condition.

### 10.3 The gates and their deadlines

- **The members.** `zstk.py members --wait` fixes the members by section 2. It writes `outputs/stk/members.json` once and
  never rewrites it, and it looks again every 10 minutes until then.
  - A stage's verdict is the one `s0stop.py gate` reads: NOT_ADOPTED on a current seed-0 STOP, otherwise the screen verdict in
    the stage's own calls.json.
  - RTR and IRA will not run once DEP's verdict is ADOPT_FOR_FULL_RUN.
  - At 12:00 IST on 5 November, a stage still without a verdict is left out.
  - The same record fixes the stack's layer count, its rules and section 6's pick, with each passing form's mean delta.
- **The stack.** `zstk.py stack-gate` exits 0 with two or more members, so the stack's items run. Otherwise it exits 1, and
  they drop.
- **Each stage's own full run.** `zstk.py own-gate --stage <stage> --wait` looks every 10 minutes.
  - It exits 0, and the stage's held items are released, when section 3 or 6 gives the stage its own full run. That happens
    in two cases: the stage is the one member, or it is the pick and the stack's screen is NO_GAIN or NOT_ADOPTED, or its
    full run is NOT_ADOPTED.
  - It exits 1, and the held items drop, in three cases: the stage is neither the one member nor the pick; the stack's full
    run is ADOPT; or the stack is still undecided at 00:00 IST on 7 November, the freeze (section 6).
- **The stack's seed-0 stop (section 3)** is s0stop's, run through `zstk.py s0stop-check` and `zstk.py s0stop-run`.
  - Its deciding reads are the union of the WQ-musique deciding reads of the rules its members use. That is DEP's eight when
    DEP, RTR or IRA is a member (LTG's four are among them), and LTG's four otherwise.
  - `zstk.py gate` is s0stop's gate for the stack.

### 10.4 The items (host)

The stack's block goes right after `ira-k4-s0-hp`, so its fits take the card ahead of every held full run:

| item | after | size | runs (`zstk.py ...`) |
| --- | --- | --- | --- |
| `stk-selftest` | (none) | 1 CPU, 4 GB | `--selftest` |
| `stk-members` | `stk-selftest` | 0.2 CPU, 1 GB | `members --wait` |
| `stk-stack-gate` | `stk-members` | 0.2 CPU, 1 GB | `stack-gate` |
| `stk-own-dep`, `-ltg`, `-rtr`, `-ira`, `-gnx` | `stk-members` | 0.2 CPU, 1 GB each | `own-gate --stage <stage> --wait` |
| `stk-st-s0` | `stk-stack-gate`, and the checks the members' fits need (`pkoh-gate`, `iso-verdict`, `ml1start-check`, `readswa-check`, `gnx-check-cols`) | 1.1 CPU, 10 GB, 0.6 of the card | `fit --name scr-stk-st-s0` |
| `stk-s0stop` | `stk-st-s0` | 1 CPU, 4 GB | `s0stop-check` |
| `stk-st-s0-hp` | as `stk-st-s0`, and `stk-s0stop` | 1.1 CPU, 16 GB, 0.9 of the card | `s0stop-run --` the fit |
| `stk-compare`, `stk-calls`, `stk-gate` | the two seed-0 fits, in turn | 1 CPU; 8, 4 and 1 GB | `compare-all`, `calls`, `gate --want ADOPT_FOR_FULL_RUN` |
| `stk-st-s1`, `stk-st-s2` | as `stk-st-s0`, and `stk-gate` | as `stk-st-s0` | the full run's WQ-musique fits |
| `stk-st-s1-hp`, `stk-st-s2-hp` | as `stk-st-s0`, and `stk-gate` | as `stk-st-s0-hp` | the full run's WQ-hotpotqa fits |
| `stk-compare-full`, `stk-calls-full` | the four full-run fits, in turn | 1 CPU; 8 and 4 GB | `compare-all`, `calls` |

- **Card shares.** The stack's fits get more of the card than any member's fits (0.42 and 0.83), because the stack holds
  several members at once and no stack fit has been measured. Each fit runs under `retry_cmd.py --tries 3`.
- **The holds.** They are released by changing dependencies only (section 9). In the sixteen held fits of DEP, LTG, RTR and
  IRA, `hold-full-for-stack` becomes `stk-own-<stage>`.
- **GNX's own full run**, which section 3 says this amendment adds, goes before `iso-zg6z-s0-hp`:
  - `gnx-gate`, after `gnx-calls`: `s0stop.py gate --stage gnx --want ADOPT_FOR_FULL_RUN`;
  - `gnx-nx-s1`, `gnx-nx-s2`, `gnx-nx-s1-hp` and `gnx-nx-s2-hp`: each with its seed-0 fit's size and command, after
    `gnx-gate` and `stk-own-gnx`;
  - `gnx-compare-full` and `gnx-calls-full`: `zgnx.py compare-all` and `zgnx.py calls`. zgnx's calls already gives the full
    run's verdict once all six fits are read.
- **ISO.** `iso-zg6z-s0-hp` keeps its own hold.

### 10.5 The self-test

`zstk.py --selftest` runs on the CPU, with no fit: on the laptop, and on the host as `stk-selftest`. It checks:

- the stack's fit and compare names, which no other stage's fits share, the pick's six reads and the two deadlines;
- the rules' combination on made rows;
- zdep's and zltg's own screen calls on one made input, which must give one row for each of the ten reads both call;
- the member-code rule on made stamps:
  - one sha in both of a member's fits passes, and any other fails;
  - the earlier zltg sha passes on LTG's WQ-musique fit only;
  - an unread fit fails;
- the gates on made files:
  - the members are fixed once and never rewritten;
  - a STOP is read as NOT_ADOPTED;
  - IRA stands for RTR;
  - the pick's tie, and the late leave-out;
  - the stack gate and every own-gate outcome;
  - the stack's seed-0 stop and its deciding reads;
- on zg6r's toy carve, eight stacks without DEP and three with DEP (in a process of its own, with four layers):
  - each starts as the same set without LTG and GNX, bit for bit, in eval and training forwards;
  - every member's parameters get the gradient of the listwise and next-hop losses;
  - the strict load passes under the patches and is refused without them;
  - a one-epoch fit moves each zero-started parameter;
  - the stamp's record is checked, and LTG's term in the rounds is refused without the rounds;
  - every patch is put back, also on a failure.
