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
