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

## 11. Amendment A2 (3 November): the gnx slot on LTA's columns (GNA)

Committed with docs/GNA_ALL_ROWS_COLUMNS_IN_GNN.md, before any GNA fit or stack fit and before the members are fixed.

**What had been seen when it was written:**

- DEP's screen was filed ADOPT_FOR_FULL_RUN (d9c74d6).
- GNX's WQ-musique seed-0 stop gave GO at 08:08 IST, with its four deciding reads WITHIN.
- No WQ-hotpotqa read of GNX or LTG had been seen, and no GNA fit had run.

**The change.**

- **The members wait on GNA's verdict too.** It is read as s0stop's gate reads a verdict, under GNA's entry: NOT_ADOPTED on a
  current seed-0 STOP, otherwise the screen verdict in `outputs/gna/calls.json`. A GNA with no verdict at 12:00 IST on 5
  November is left out, as any stage is.
- **The gnx slot passes when GNX's screen or GNA's passes.** Its columns are LTA's when GNA's screen passes and GNX's does
  not, or when both pass and GNA's mean seed-0 delta over section 6's six reads is larger than GNX's. They are LTM's
  otherwise, including on a tie. The members record names the columns (`gnx_columns`) and both means.
- **The pick (section 6)** takes the gnx slot's mean from its columns' stage: GNA's calls when the columns are LTA's.
- **A stack fit on LTA's columns** runs under `zlta.on_cols`, with zgnx's NEXT block and term record set to GNA's
  (`zstk.on_columns`). The NEXT block in its stamp must name its columns' declaration, LTA's or LTM's. That is one more
  INCOMPLETE condition (section 10.2).
- **"As it was screened" (section 10.2).** For the gnx slot on LTA's columns, the member's own seed-0 fits are GNA's, and
  the sha256 of zgna, zlta and ltacols count beside zgnx's. In every member's check, an own fit whose stamp names no sha for
  a key now fails. Before, it passed when EARLIER had no entry for that fit.
- **Each stage's own full run (sections 3 and 6).** The gnx slot's is GNX's on LTM's columns and GNA's on LTA's
  (`own-gate --stage gna`). The other one is not released.
- **The code.** zstk.py imports zgna, and through it zlta and ltacols, and edits none of them. Their sha256 join the stage's
  code record.
- **LTA.** STK's members now wait on GNA, and GNA's fits wait on LTA's column items. LTA's section 7 did not foresee this.
  LTA's own items and rule are unchanged.

**What does not change.** Everything else in sections 1 to 10: the five forms and their order, the rules, the deadlines,
and the stack's fits and their sizes.

**The items (host):**

- **`stk-selftest-a2`** (1 CPU, 4 GB), then **`stk-members-a2`** (0.2 CPU, 1 GB, `members --wait`), right after
  `stk-members`.
- **`stk-stack-gate` and the `stk-own-*` items** wait on `stk-members-a2` in `stk-members`'s place.
- **`stk-members`** ran A1's code, which does not wait on GNA. It is cancelled once no item depends on it. It had written no
  `members.json`.
- **`stk-own-gna`** (0.2 CPU, 1 GB): `own-gate --stage gna --wait`, after `stk-members-a2`. It holds GNA's full run.
- **Inputs.** Every stk item lists zgna.py, zlta.py, ltacols.py and GNA's file.

**The self-test** passed on the laptop's CPU before this commit. On made files, it adds these cases:

- **GNA's verdict.**
  - It is waited on.
  - A calls file declared by another file is not a verdict.
  - GNA's seed-0 STOP is read as NOT_ADOPTED.
- **GNA passing alone.** The gnx slot passes on LTA's columns. GNA's mean is the slot's in the pick, and GNA's own run is
  released, not GNX's.
- **Both passing.** The columns are LTA's when GNA's mean is larger, and LTM's on a tie or a smaller one. Every own-gate
  outcome is checked.
- **An unknown column set** is not a members record.

It also adds these cases outside the made files:

- **`on_columns`.**
  - On LTM's columns, nothing is patched.
  - On LTA's, the patches are made and put back, also on a failure.
  - An unknown set is refused.
- **`member_code` on GNA's fits** and their four sha256: a mismatch, a missing sha and unread fits each fail.

## 12. Amendment A3 (3 November): GNP's pair columns in the gnx slot

Committed after GNP's declaration (docs/GNP_PAIR_COLUMNS_IN_GNN.md, 2f6aaf7), before any GNP fit or stack fit and before
the members are fixed.

**What had been seen when it was written:**

- PFS A3 was filed PAIR_ADDS on four of its five reads, and GNP was declared on its six pair columns (2f6aaf7).
- On the host, GNP's eleven column items, its match against zpfp's batch and its column check were done by 09:30 IST. No
  GNP fit had run.
- No GNA fit had run, and no WQ-hotpotqa read of GNX or LTG had been seen.

**The change.**

- **GNP is a form of the stack.** The forms are section 2's five and GNP, in the order DEP, LTG, RTR, IRA, GNX, GNP.
  - GNP's form enters through the gnx slot, so the members stay section 2's five.
  - The gnx slot is a member when GNX's, GNA's or GNP's screen passes.
  - The members record lists both the passing forms (`forms`) and the members they make (`members`).
- **The members wait on GNP's verdict too.** It is read as s0stop's gate reads a verdict, under GNP's entry (zgnp's fit
  names): NOT_ADOPTED on a current seed-0 STOP, otherwise the screen verdict in `outputs/gnp/calls.json` when that file is
  declared by GNP's file. A GNP with no verdict at 12:00 IST on 5 November is left out, as any stage is.
- **The gnx slot's columns.** The NEXT set is chosen as in section 11, LTM's or LTA's. When GNP's screen passes, the six pair
  columns are added:
  - with GNX's or GNA's screen passing too, the slot reads the NEXT set's seven columns, then the six pair columns, through
    one shared constant: 27 inputs (`LTM+PAIR` or `LTA+PAIR`);
  - with neither, it reads the six pair columns alone, as GNP's fits do: 13 inputs (`PAIR`).
  - The members record names the set (`gnx_columns`). The gnx term has 128 parameters for each input.
- **The pick (section 6)** is taken over the forms.
  - GNP's mean seed-0 delta over the six reads comes from GNP's calls. The gnx slot's comes from its NEXT set's stage.
  - A tie goes to the earlier form, so GNX's or GNA's goes before GNP's.
- **The stack runs with two or more forms** (`stack-gate`). With one form, that form's own full run goes ahead and the
  stack's items drop. GNX and GNP passing alone still make a stack: one member, the gnx slot on both column sets.
- **Each form's own full run (sections 3 and 6).**
  - GNP's is released by `own-gate --stage gnp` in two cases: GNP is the one form; or it is the pick, and the stack's screen
    is NO_GAIN or NOT_ADOPTED or its full run is NOT_ADOPTED.
  - The gnx slot's own run stays GNX's or GNA's, by its NEXT set (section 11).
- **A stack fit on two column sets.**
  - Each set loads under its own stage's names. They join row by row, the NEXT set first.
  - Both sets must give the same rows and the same training flag.
  - The NEXT block names both declarations (a list) and both column roots. Its fit sha256 is the two sets' joined by "+".
  - zgnx's names for LTM's columns stay set while the sets load.
  - A set in another order, two NEXT sets, or an unknown set is refused.
  - A stack fit whose NEXT block names other columns or other declarations is INCOMPLETE (section 10.2), as in section 11.
- **"As it was screened" (section 10.2).** For GNP's form, the member's own seed-0 fits are GNP's (`scr-gnp-px-s0` and
  `scr-gnp-px-s0-hp`). The sha256 of zgnp, gnpcols and zpfp count beside zgnx's, and the stack's code record adds the three.
- **The code.** zstk.py imports zgnp on first use, and through it gnpcols and zpfp. It edits none of them.
  - zgnp keeps zgnx's own functions when it is imported. So zstk imports it outside every patch, and stops if what zgnp
    kept is not zgnx's own.
  - The import keeps the thread variables as they were.
- **A members record that disagrees with itself is refused.** That covers three cases:
  - forms that do not make the members;
  - a pair set without GNP's form, or GNP's form without a pair set;
  - GNP's form with a NEXT set but without the gnx slot's form, or the reverse.
  A record written before this amendment, with no forms, reads its members as its forms.
- **GNP.** Its own items and rule are unchanged. Its full run (`gnp-px-s1`, `-s2`, `-s1-hp`, `-s2-hp`) already waited on
  `stk-own-gnp`, which this amendment adds.

**What does not change.** Everything else in sections 1 to 11 stays: the five members and their order, the rules, the
deadlines, GNA's part, and the stack's fits and their sizes.

On the sizes: the NEXT columns are float16 on the CPU. GNX's WQ-musique seed-0 fit held its seven columns, for 16.6
million training rows, in 0.23 GB. Six more columns add about 0.2 GB.

**The items (host):**

- **`stk-selftest-a3`** (1 CPU, 4 GB), then **`stk-members-a3`** (0.2 CPU, 1 GB, `members --wait`), right after
  `stk-members-a2`.
- **`stk-stack-gate` and the `stk-own-*` items** wait on `stk-members-a3` in `stk-members-a2`'s place.
- **`stk-members-a2`** ran A2's code, which does not wait on GNP. It is cancelled once no item depends on it. It had written
  no `members.json`.
- **`stk-own-gnp`** (0.2 CPU, 1 GB): `own-gate --stage gnp --wait`, after `stk-members-a3`. It holds GNP's full run.
- **Inputs.** Every stk item lists zgnp.py, gnpcols.py, zpfp.py, zpfs.py and GNP's file, along with PFS's file.

**The self-test** passed on the laptop's CPU before this commit. On made files, it adds these cases:

- **GNP's verdict.**
  - It is waited on.
  - GNP's seed-0 STOP, under GNP's entry, is read as NOT_ADOPTED, and zgnp's own stop record agrees.
- **GNP beside DEP.**
  - The members are DEP and the gnx slot, on the pair columns alone.
  - The pick is GNP, on the larger mean.
  - After the stack's full run is NOT_ADOPTED, only GNP's own run is released.
- **GNX and GNP.** The columns are `LTM+PAIR`. The pick goes to GNX on a tie, and to GNP on a larger mean.
- **GNA and GNP.** The columns are `LTA+PAIR`. Only GNA's own run is released.
- **GNP alone.** It is one form, so the stack gate exits 1 and GNP's own run is released.
- **Members records.** Seven that disagree are refused, and a valid `LTA+PAIR` record passes. An A2 record with no forms
  reads its members as its forms.

It also adds these cases outside the made files:

- **zgnp's import** keeps the thread variables. zgnp's names, columns, code keys and fits are A3's.
- **`on_columns`** on the pair columns alone and on both sets.
  - The patches are made and put back, also on a failure.
  - On both sets, LTM's names stay set.
  - A set in another order, two NEXT sets and an unknown set are refused.
- **The joined load** on made LTM, LTA and GNP columns.
  - It gives 13 columns, each half equal to its own set's.
  - It carries the joined fit sha256 and both declarations.
  - Wrong roots are refused.
- **`member_code` on GNP's fits** and their sha256.
- **Two more stacks on zg6r's toy carve**, without DEP: one on `LTM+PAIR` and one on `PAIR`. Each passes section 10.5's
  checks.

## 13. Amendment A4 (3 November): GNW and LTX as forms; how they compose is A5's

Committed before any GNW or LTX fit and before any stack fit. The members are not fixed yet: `stk-members-a3` is still
waiting on GNX, GNA and GNP.

**What had been seen when it was written:**

- LTG's screen was filed NOT_ADOPTED (538e467). Its amendment A5 (docs/LTG_NEXT_HOP_GNN.md section 13) reads the loss as
  the next-hop scorer fitting its own training questions' golds. LTX (docs/LTX_CROSSFIT_NEXT_HOP_GNN.md, 6afca09) is the
  same term with the scorer cross-fitted by gold components.
- GNW (docs/GNW_PAIR_WEIGHTS_IN_GNN.md, 4c89736) puts the pair scores into zg6r's encoder as message weights.
- On the host:
  - GNW's and LTX's column items and their column checks were done by 12:58 IST, and no GNW or LTX fit had run.
  - GNX's WQ-hotpotqa seed-0 fit was training. GNA's and GNP's seed-0 fits were queued behind it.
  - No GNA, GNP or stack fit had run.

**Why.** The stack is the only GNN full run before the freeze (sections 6 and 10.3). GNW and LTX are GNN screens aimed at
musique's bridge hop, and each one's own file holds its full run for the stack. Without this amendment, a passing GNW or
LTX would reach no full run before the freeze.

On the card, the stack's fits come after GNW's and LTX's seed-0 fits in the feeder's order. So waiting on their verdicts
costs the stack only their compares and calls.

**The change.**

- **The members wait on GNW's and LTX's verdicts too.** Each is read as s0stop's gate reads a verdict, under the stage's
  own entry (its module, root, two seed-0 fits, LTG's rule and declaration, as `zgnw.on_s0stop` and `zltx.on_s0stop`
  patch them in):
  - NOT_ADOPTED on a current seed-0 STOP;
  - otherwise the screen verdict in `outputs/gnw/calls.json` or `outputs/ltx/calls.json`, when that file is declared by
    the stage's file.

  A stage with no verdict at 12:00 IST on 5 November is left out, as any stage is. The members record names both verdicts
  (`gnw_verdict`, `ltx_verdict`).
- **The forms** are DEP, LTG, RTR, IRA, GNX, GNP, GNW and LTX, in that order. The order settles the pick's tie (section 6).
  - A passing GNW or LTX is a form and a member of its own. The members are listed in section 2's order, then GNW, then LTX.
  - LTG's screen is filed NOT_ADOPTED, so LTG and LTX are never both members.
- **The rules (section 4).** GNW and LTX are each read under LTG's in-domain rule, as their own files read them. A stack
  with either reads LTG's rule, and its seed-0 stop's deciding reads include LTG's four.
- **The pick (section 6)** takes GNW's and LTX's mean seed-0 deltas over the six reads from their own calls.
- **Each form's own full run (sections 3 and 6).** GNW's and LTX's are released by `own-gate --stage gnw` and
  `own-gate --stage ltx` in the same cases as any form's:
  - the stage is the one form;
  - or it is the pick, and the stack's screen is NO_GAIN or NOT_ADOPTED, or its full run is NOT_ADOPTED.

  Their full-run items are added here, held as every other form's are.
- **How they compose.** That is amendment A5's.
  - A5 is committed before any stack fit runs, and only for a form whose seed-0 stop gave GO.
  - It changes how the members compose and nothing else: not the members, the rules, the pick or any gate.
  - Until A5, the stage module composes section 2's five alone (`COMPOSED`), and a stack fit refuses members it does not
    compose.
- **The stack gate waits for that code.** Suppose the members hold a form the code does not compose.
  - `stack-gate --wait` looks every 10 minutes whether the zstk.py on disk composes them. It checks in a process of its
    own (`zstk.py composes`), because a waiting gate runs the code it started with. The stack runs once it does.
  - If the code still does not compose them at 12:00 IST on 5 November, the stack does not run.
  - The decision is written once (`outputs/stk/stack_gate.json`).
  - The own gates read a stack that does not run as the stack's screen NOT_ADOPTED, so the pick's own full run goes
    ahead.
  - With members this code composes, the gate is as before. It decides at once, and writes no record.

**What does not change.** Everything else in sections 1 to 12: section 2's five members and their composition, the
rules, the deadlines, GNA's part, GNP's part, and the stack's fits and their sizes.

**The items (host):**

- **`stk-selftest-a4`** (1 CPU, 4 GB), then **`stk-members-a4`** (0.2 CPU, 1 GB, `members --wait`), right after
  `stk-members-a3`.
- **`stk-stack-gate`** waits on `stk-members-a4` and runs `stack-gate --wait`. It had not been sent.
- **The `stk-own-*` items** wait on `stk-members-a4` in `stk-members-a3`'s place.
- **`stk-own-gnw` and `stk-own-ltx`** (0.2 CPU, 1 GB each) run `own-gate --stage gnw --wait` and
  `own-gate --stage ltx --wait`, after `stk-members-a4`.
- **`stk-members-a3`** ran A3's code, which does not wait on GNW or LTX. It is cancelled once no item depends on it. It
  had written no `members.json`.
- **The stack's fits** also wait on `gnw-check-cols` and `ltx-check-cols`.
- **GNW's and LTX's full runs** go after GNX's.
  - `gnw-mw-s1`, `gnw-mw-s2`, `gnw-mw-s1-hp` and `gnw-mw-s2-hp`: each with its seed-0 fit's size and command, after
    `gnw-gate` and `stk-own-gnw`.
  - `gnw-compare-full` and `gnw-calls-full`: `zgnw.py compare-all` and `zgnw.py calls`.
  - LTX's are the same, with `ltx-xf-*`, `ltx-gate` and `stk-own-ltx`.
- **Inputs.** Every stk item lists zgnw.py, gnwcols.py, zltx.py, ltxcols.py and both files beside A3's.

**The self-test** passed on the laptop's CPU before this commit.

On made files, it adds these cases:

- **GNW's and LTX's verdicts.**
  - Both are waited on.
  - A calls file declared by another file is not a verdict.
  - Each one's seed-0 STOP, under its module's own entry, is read as NOT_ADOPTED. The module's own stop record agrees,
    and s0stop's table is put back.
- **DEP and LTX passing.**
  - They make two forms and two members, at four layers, under both rules, and the pick is LTX on its larger mean.
  - The gate returns 2 without wait, with one look a call.
  - At MEMBERS_BY it decides once that the stack does not run, and that decision stands when the code composes later.
  - The own gates then release LTX's own full run alone.
  - Composed before MEMBERS_BY, after one look that it is not: the stack runs (one POLL slept), and the own gates wait on
    the stack.
- **GNX, GNW and LTX passing with DEP failing.** Three forms and three members at three layers, under LTG's rule. The
  pick's tie goes to GNX.
- **GNW alone.** One form, so there is no stack. Composition is never looked at, and only GNW's own full run is released.
- **`composes`.**
  - It returns 2 with no members fixed, 0 on section 2's members, and 1 on GNW's or LTX's.
  - On disk it runs in a process of its own.
- **Members records.** A record with GNW's or LTX's forms is read when its forms make its members, and refused otherwise.

It also adds these cases outside the made files:

- zgnw and zltx are imported after zgnp and outside every patch. Their names, fits and stop entries are A4's, and the
  thread variables are kept.
- Every earlier case runs as written, with GNW and LTX not passing.

## 14. Amendment A5 (3 November): GNR, GNT and GNS as forms, one attention form, and RTR and IRA reopened by a second pass

Committed before any GNR, GNT or GNS screen verdict, before any RTR or IRA fit and before any stack fit. The members are
not fixed yet: `stk-members-a4` is still waiting on GNA, GNP and LTX.

**What had been seen when it was written:**

- DEP's screen was filed ADOPT_FOR_FULL_RUN (d9c74d6). By section 2, RTR and IRA then do not run. `rtr-open` and
  `ira-open` refused, the feeder dropped every RTR and IRA fit item after them, and no RTR or IRA fit has run.
- GNW's seed-0 stop was STOP, so its screen is NOT_ADOPTED (5f8da44). Under WQ-musique, webqsp fell 0.0124 against a
  0.0085 floor. Beside, held-out musique rose 0.0165, its 3-hop slice 0.0197 and its 4-hop slice 0.0247, all GAIN.
- Three GNN forms followed:
  - **GNR** (docs/GNR_REANCHORED_NEXT_GNN.md, dbc8a43) re-anchors GNX's NEXT columns on the base GNN's own top five.
    Its seed-0 stop gave GO (24a330c), and its WQ-hotpotqa seed-0 fit waits in the GPU queue.
  - **GNT** (docs/GNT_LEARNED_ATTENTION_GNN.md, 213eed7) learns attention to the first stage's top five on GNW's
    columns. This is the next form that GNW's file names.
  - **GNS** (docs/GNS_SELF_ANCHORED_ATTENTION_GNN.md, ba6fca0) is GNT's attention on the GNN's own first-pass top five,
    which is GNR's first pass.

  GNT's and GNS's seed-0 WQ-musique fits were training. Each of the three files holds its own full run for the stack, which
  takes it only by an amendment of this file.
- On the host, GNA's, GNP's and LTX's WQ-hotpotqa seed-0 fits waited in the GPU queue, and no stack fit had run.

**Why.**

- GNR, GNT and GNS are GNN screens aimed at musique's bridge hop, as GNW and LTX are. Without this amendment, a passing one
  would reach no full run before the freeze.
- GNR and GNS each read the GNN's own top five again after a first pass. A pass by either is a second pass helping, and
  that is what RTR's rounds do inside one model and IRA's rounds with their alignment slots.
- Section 2 shuts RTR and IRA on DEP's passing screen to keep DEP and the rounds out of one stack. It says nothing against
  the rounds. So a pass by GNR or GNS reopens them, and the stack then takes DEP or the rounds, not both.

**The change.**

- **The members wait on GNR's, GNT's and GNS's verdicts too.** Each is read as A4 reads GNW's and LTX's: s0stop's gate
  under the stage's own entry, so NOT_ADOPTED on a current seed-0 STOP, otherwise the screen verdict in the stage's
  `calls.json` when that file is declared by the stage's file.
  - A stage with no verdict at 12:00 IST on 5 November is left out, as any stage is.
  - The members record names the three verdicts (`gnr_verdict`, `gnt_verdict`, `gns_verdict`).
- **The forms** are A4's, then GNR, GNT and GNS. Each passing one is a form and a member of its own, listed after LTX, and
  each is read under LTG's in-domain rule, as its own file reads it.
- **One attention form at most.** GNW, GNT and GNS are three versions of one input, weights or attention over a top five.
  When two or more pass, the stack keeps one:
  - the one with the largest mean seed-0 delta over LTG's ten deciding reads (its calls file's rows);
  - on a tie, the earlier of GNW, GNT and GNS;
  - a form missing one of the ten reads only when none has all ten, and then the earliest.

  GNT's and GNS's files fixed this choice (their section 7 rows). The record names each passing one's mean
  (`attention_means`).
- **RTR and IRA reopen on a second pass.** With DEP's screen passing, RTR and IRA also run when GNR's or GNS's screen is
  ADOPT_FOR_FULL_RUN:
  - `reopen-gate --wait` reads GNR's and GNS's verdicts as the members do.
    - It opens once either is ADOPT_FOR_FULL_RUN.
    - It closes once both are other verdicts, or at 12:00 IST on 5 November, since RTR's and IRA's screens could not be
      read by then.
  - It writes its call once (`outputs/stk/reopen.json`, `open` true or false). It exits 0 when open and 1 when closed.
  - RTR's and IRA's fits read that record through `zrtr.dep_call` (RTR A5, docs/RTR_GO_BACK_ROUNDS.md section 13; IRA A5,
    docs/IRA_ITERATIVE_ALIGNMENT.md section 14). DEP's screen ADOPT_FOR_FULL_RUN with no full run, and an open record
    declared by this file, let them run.
  - While the call is open or not yet made, the members wait on RTR's and IRA's verdicts. Once it is closed, RTR and IRA
    will not run, as before (`will_not_run`). The record says whether the rounds were reopened (`rounds_reopened`).
- **DEP or the rounds, not both.** Section 2 keeps DEP out of any stack with RTR or IRA, and that stays.
  - The rounds' form is IRA, or RTR when IRA does not pass.
  - When DEP and the rounds' form both pass, the stack keeps the one with the larger mean seed-0 delta over section 6's six
    reads. A tie, or a missing mean, goes to DEP.
  - The record names both means (`dep_or_rounds_means`).
- **The pick (section 6)** is over the passing forms, except GNR and GNS.
  - Their columns are per seed. Seeds 1 and 2 need their own first passes and columns before any fit reads them (GNR's
    file, section 6; GNS's file, section 7).
  - So neither has an own full run before the freeze, even as the one form (`NO_OWN`).
  - GNT's own full run is released by `own-gate --stage gnt` in the same cases as any form's. Its items are added here.
- **How they compose.** As A4 says for its later forms. That is amendment A6, A4's "A5" under its new name:
  - committed before any stack fit runs, and only for forms whose seed-0 stop gave GO;
  - it also states how the stack's fits read DEP's call when the rounds were reopened.

  Until A6, the stage module composes section 2's five alone, and `stack-gate --wait` waits for that code as A4 says.

**What does not change.** Everything else in sections 1 to 13:

- section 2's five members and their composition;
- the rules, the deadlines and the gates;
- GNA's, GNP's, GNW's and LTX's parts;
- the stack's fits and their sizes;
- DEP's own full run, held for the stack.

**The items (host):**

- **`stk-selftest-a5`** (1 CPU, 4 GB), then **`stk-members-a5`** (0.2 CPU, 1 GB, `members --wait`), next to
  `stk-members-a4`.
- **`stk-reopen`** (0.2 CPU, 1 GB, `reopen-gate --wait`), after `stk-selftest-a5`.
- **`stk-stack-gate` and the `stk-own-*` items** wait on `stk-members-a5` in `stk-members-a4`'s place.
- **`stk-own-gnt`** (0.2 CPU, 1 GB, `own-gate --stage gnt --wait`), after `stk-members-a5`. There is no `stk-own-gnr` or
  `stk-own-gns`.
- **GNT's full run**, after GNP's in the queue, each fit with its seed-0 fit's size and command:
  - `gnt-at-s1` and `gnt-at-s2` (GPU 0.42, cap 0.40, 15 GB);
  - `gnt-at-s1-hp` and `gnt-at-s2-hp` (GPU 0.83, cap 0.80, 18 GB);
  - all four after `gnt-gate` and `stk-own-gnt`;
  - then `gnt-compare-full` and `gnt-calls-full` (`zgnt.py compare-all` and `zgnt.py calls`).
- **RTR's and IRA's items, under new names.** The feeder dropped the old names when `rtr-open` and `ira-open` refused, and
  a dropped name does not come back. Each new item has its old item's command, size and fit name:
  - `rtr-re-selftest` and `ira-re-selftest`;
  - the screens: `rtr-re-r2-s0`, `rtr-re-s0stop`, `rtr-re-r2-s0-hp`, `rtr-re-compare`, `rtr-re-calls` and `rtr-re-gate`;
  - the full runs: `rtr-re-r2-s1`, `rtr-re-r2-s2`, `rtr-re-r2-s1-hp`, `rtr-re-r2-s2-hp`, `rtr-re-compare-full` and
    `rtr-re-calls-full`;
  - IRA's the same, with `ira-re-` and `k4`.

  The WQ-hotpotqa fits change in one way: they take GPU 0.83 with a cap of 0.80, as LTG's and GNX's do, in place of
  0.75 and 0.72. DEP's WQ-hotpotqa fit ran out of memory under 0.72 (it asked for 0.743 of the card) and finished
  under 0.77, and RTR and IRA run zg6r's layers twice.

  The screen fits wait on `stk-reopen` in `rtr-open`'s and `ira-open`'s place. The full-run fits wait on
  `stk-own-rtr` and `stk-own-ira`, as before. In the GPU order the screen fits come after GNS's WQ-hotpotqa seed-0 fit and
  before the stack's.
- **`stk-members-a4`** ran A4's code, which does not wait on GNR, GNT or GNS. It is cancelled once no item depends on it.
  It had written no `members.json`.
- **Inputs.** Each new stk item lists zgnr.py, gnrcols.py, zgnt.py, zgns.py and gnscols.py and their three files, beside
  A4's inputs.

**The self-test** passed on the laptop's CPU before this commit.

On made files, it adds these cases:

- **DEP's screen passing, GNR and GNS pending.**
  - The members wait on RTR and IRA, and on GNR, GNT and GNS. `reopen-gate` without wait returns 2.
  - With GNT passing and GNR at NO_GAIN, the call still waits on GNS. At 12:00 IST on 5 November it would close.
  - GNS's NOT_ADOPTED closes it: the waiting gate exits 1 at once, and `reopen.json` has `open` false.
  - The members are then DEP and GNT, RTR and IRA will not run, the pick is DEP, and RTR's reader finds no open record.
- **The waiting gate.** With GNS's calls INCOMPLETE it looks again after one POLL, finds GNS at ADOPT_FOR_FULL_RUN and
  exits 0. `zrtr.dep_call` and `zira.dep_call` then both answer "screen ADOPT_FOR_FULL_RUN, reopened by STK A5".
- **Reopened beside DEP.**
  - With GNS passing, the members wait on RTR and IRA, and nothing is marked as not running.
  - RTR and IRA then pass. IRA stands for RTR, and IRA is kept over DEP on the larger six-read mean (0.002 against 0.001).
    On a tie (0.001 each), DEP is kept.
  - The members are the kept one and GNS. The pick is RTR on its larger mean, and GNS is not among the pick's means.
- **One attention form**, with GNW, GNT and GNS all passing:
  - means of 0.004, 0.006 and 0.005 keep GNT;
  - a tie at 0.006 keeps GNW;
  - with GNT's calls missing one of the ten reads, GNS is kept on the larger mean of the two with all ten (0.009
    against GNW's 0.002);
  - with every one missing a read, GNW is kept.

  Each case releases the kept form's own full run, and none when it is GNS.
- **GNR alone.** One form, so there is no stack and no own full run, and nothing is released.
- **Members records.** Records with DEP and GNR, IRA and GNT, or GNR and GNS are read. Records with GNW and GNT, GNT and
  GNS, DEP and IRA, or DEP, RTR and GNR are refused.

It also adds these cases outside the made files:

- zgnr, zgnt and zgns are imported after A4's modules and outside every patch. Their names, fits and stop entries are
  A5's, and the thread variables are kept.
- Each one's seed-0 STOP, under its module's own entry, is read as NOT_ADOPTED.
- Every earlier case runs as written, with GNR, GNT and GNS not passing.
