# RTR: the GNN of record with a second round of its layers that goes back to round one's top candidates, on PKO's pools

Declared 2 November 2026, before any RTR fit and before DEP's verdict. Stage module: `outputs/mp_unified/zrtr.py`. It imports
`zml1s.py`, `ziso.py`, `zdep.py`, `zg6.py` and `zg6r.py` and edits none of them.

## 1. The question

- **What the program names.** MUSIQUE_DIAGNOSIS step 3 has two depth forms: "three or four message-passing layers, or read-time
  rounds that go back to the neighbours of the top candidates". DEP (docs/DEP_GNN_DEPTH.md) is the first. DEP's section 6 says that
  on NO_GAIN or NOT_ADOPTED three layers stay and the next depth form is the second, in its own file. This is that file. It is
  declared and queued now, behind DEP's verdict, so the GPU has work when DEP's screen ends.
- **What going back means here.** The user ruled on 5 October that going back is neighbour summarisation by message passing, not
  EM rounds. Step 3 links it to S6's idea and to HippoRAG's personalised PageRank, which spreads from the question's entities. In
  zg6r, the question's context in every layer is a weighted sum of its rows' states. The weights come from one linear score of the
  first state, `s0 = W_ctx h0`, and stay fixed for all three layers. So the message passing is steered by a guess made before any
  message arrives.
- **The form.** A second round of the same three layers. Its context is weighted by round one's own score, so round two's messages
  are steered by the rows round one ranked high, and by their neighbours.
  - The rounds are trained as well as read: a round used only at reading would score with layers never trained on its context.
  - Nothing outside the model changes: no EM, no change to the pool, no second model.
- **So the question is:** does a second round, steered by round one's top candidates, raise R@5 on the twelve reads and on musique's
  3-hop and 4-hop questions?

## 2. What runs

- **The arm: the GNN of record after ISO, with two rounds of its three message-passing layers.**
  - **Round one** is zg6r's stack, op for op:
    - h0 from the encoder's states and the boundary;
    - the context weights w1, the softmax over the question's rows of s0's clipped z-score;
    - three layers;
    - s1 = W_out h + W_skip h0 + b.
  - **Round two:**
    - z1 is s1's z-score over the question's rows, clipped at zg6's ±ZCLIP;
    - the context weights w2 are the softmax of z1 over the question's rows;
    - each row starts from h0 + z1 · e, where e is a learned 64-vector that starts at 0;
    - the same three layers run again, with w2 as the question's context;
    - s2 = W_out h + W_skip (h0 + z1 · e) + b.
  - **The score is s2,** in training and at reading. The family dropout's one draw holds for both rounds. The layers are checkpointed
    in training, as zg6r's are.
  - **Parameters.** The message-passing stack and readout go from 180,218 to 180,282 at hidden 128 (e's 64).
  - **At the start** e is 0, so round two differs from zg6r only in its context weights.
  - **Everything else is the base's:** the trainer, the card and its epochs, PKO's roots and typed links, the KT input, the 64-wide
    state, the seed offsets, and the read cap of 32,768.
- **How it is set.** For the length of an RTR fit, zrtr patches two names in-process:
  - zg6r's `g6r_stack`, which zg6r's forward calls by name;
  - ZG6R's `__init__`, which adds e after zg6r's own `__init__`. e takes no random draw, so every other parameter starts as the
    base's.

  The class stays lean_screen's, so zg6r's arm check (by identity) holds. Every patched name is put back afterwards, also on a
  failure. Any later read of these fits must run under the same patch. A plain zg6r refuses the state, because the strict load
  fails on e.
- **The base, fixed by ISO's filed verdict, as DEP's** (`zdep.iso_call`, `zdep.base_fit`):
  - **KEEP:** zg6r, ML1's fits `scr-ml1-zg6r-s<k>` (with `-hp` on WQ-hotpotqa). RTR starts from the same zspk seed-0 encoder.
  - **REPLACE:** zg6r-m, ISO's fits `scr-iso-zg6rm-s<k>`. RTR takes its own draw, through ISO's own-start patches.
  - **Anything else:** no RTR fit, compare or call runs.
- **When it runs: only when DEP's filed calls keep three layers** (`zrtr.dep_call`). That means DEP's screen is NO_GAIN or
  NOT_ADOPTED, or its full run is NOT_ADOPTED. A fit refuses on anything else.
  - **DEP's screen ADOPT_FOR_FULL_RUN:** RTR's queued items drop with their opening gate. If DEP's full run is then NOT_ADOPTED, the
    same items are queued again, in an amendment that adds items only.
  - **DEP ADOPT:** the four-layer GNN is the GNN of record, and DEP's section 6 names its next form.
- **The fits.**
  - **Screen:** seed 0 on both WQ splits, `scr-rtr-r2-s0` (WQ-musique) and `scr-rtr-r2-s0-hp` (WQ-hotpotqa), on the host's GPU as
    the base's fits were.
  - **Full run:** seeds 1 and 2 on both splits, queued now behind the screen's gate.
  - Each fit is stamped under the key `rtr` with:
    - its rounds, layers and parameter counts;
    - its base fit, ISO's verdict and DEP's call;
    - zml1s's stamp (device, environment, gate, code).
- **The compares.** Each RTR fit against the base's fit of its seed and split, through zwq's `run_compare`, as DEP's.

## 3. The reads

DEP's section 3, unchanged:

- **ML1's twelve reads:** two splits by six datasets, p@swa on the s1eval carves. The zero-shot reads are among them.
- **Four slices:** musique's 3-hop questions (760 of 2,417) and 4-hop questions (405), on both splits.
  - Each slice comes from the fit's musique s1eval read file, by `zdep.slice_r5`.
  - Each file's R@5 over all its questions must equal its compare's musique R@5 within 0.00006.

## 4. The rule (fixed now)

DEP's section 4, through zdep's own functions (`decide_screen`, `decide_full`).

**The screen (seed 0)** covers sixteen reads: the twelve and the four slices.

- Per read, the floor is max(0.0075, 2 x the base's spread).
- **LOSS:** the delta is at or below minus the floor. **GAIN:** at or above the floor.
- **ADOPT_FOR_FULL_RUN:** at least one GAIN and no LOSS. **NO_GAIN:** every read WITHIN. **NOT_ADOPTED:** any LOSS.

**The full run** (seeds 1 and 2):

- ML1's rule (`zml1.verdict`) on the twelve reads against the base's three seeds.
- Each slice is called by `zml1.call`, with the pair floor.
- **ADOPT:** zml1 says ADOPT and no slice is a LOSS. **NOT_ADOPTED** otherwise.

**INCOMPLETE** is DEP's list, with one line changed. It holds when:

- a fit, compare or read file is missing;
- ISO's verdict is not REPLACE or KEEP;
- a fit's train or read stamp does not name two rounds of three layers, its base fit, ISO's verdict and a DEP call that keeps
  three layers;
- its encoder's start is not its base's;
- it ran on another device than its base;
- its trainer's code differs from its base's (stage modules aside);
- the base's seed-0 R@5 in RTR's compare differs from its own stage's;
- a read file's R@5 differs from its compare's.

A verdict closes only the form it tested.

## 5. Reported beside, deciding nothing

- The mean delta over the twelve reads.
- The parameter counts (each fit's stamp), the seconds per fit and the GPU peaks (`ml1_timing.json`).
- DEP's seed-0 deltas, in `outputs/dep/calls.json`. They are against the same base, so the two depth forms read side by side; they
  are not recomputed here.
- The floor, printed beside each delta.

## 6. What follows (fixed now)

| outcome | what follows |
| --- | --- |
| ADOPT_FOR_FULL_RUN | The full run, queued now behind `rtr-gate`, then section 4's full-run rule. On ADOPT, zg6r with two rounds is the GNN of record. Claim 1 (C1R's intervals) and Claim 3 (cold timing) are measured again on it, and the next form is three rounds, in its own file. |
| NO_GAIN or NOT_ADOPTED | Step 3's two declared forms are closed as tested, and one round (zg6r) stays. The GNN's next form is the per-layer context, in its own file: each layer's question context weighted by the scores of that layer's own input states, where zg6r fixes the weights from the first state for every layer. |
| INCOMPLETE | What is missing is fixed in an amendment, before any number is read. |

Claim 3's cold re-time (ML1 section 13, item 4) follows DEP's verdict and does not wait for RTR's.

## 7. Risks, stated before any number

- **Seed 0 only in the screen.** It is a one-fit screen. The full run decides any adoption.
- **RTR waits on DEP.** dep-calls is expected about 08:00 on 3 November. The GPU is held by ISO's and DEP's fits until then, so the
  wait costs no GPU time.
- **Second passes have added little before.** S6 part 14's extra read-time EM round added about nothing. RTR's round is message
  passing inside the model, trained end to end, but the risk is of the same kind: a second pass over the same pool may add nothing.
- **Zero-shot reads may fall.** S6 found that static neighbour summaries hurt the KBs' zero-shot reads beyond one step. A round-two
  state sees round one's scores three edges out, and each of those scores saw three edges further. The rule's LOSS catches a fall.
- **Weight tying.** The same three layers serve both rounds' contexts. Only s2 is scored, so round one's score is shaped only through
  round two's context and start. Round one's score may drift away from a good ranking.
- **Early training.** At the start, round two's context comes from round one's untrained score. The base's card and epochs may be too
  few for two rounds. The card is not changed in this stage.
- **Time.** The layers run twice. The encoder and the batch build run once, so a fit takes about 1.5 to 1.9 times the base's. ML1's
  zg6r read took 77 minutes on WQ-musique.
- **Memory.** Round two adds three stored layer inputs per batch (N x 64 floats each). The items keep DEP's caps. A fit that runs out
  of memory stops INCOMPLETE, and the caps change in an amendment before it runs again.
- **The self-test's model is a toy.** It shows that the rounds build, match zg6r and zg6 by hand, and train. It says nothing about
  R@5.

## 8. How it runs

Feeder items on the host:

1. `rtr-selftest`, on the CPU, ahead of DOB's held base fits.
2. `rtr-open`, after `dep-calls`: `verdict_gate.py` on `outputs/dep/calls.json`, `screen.verdict` in NO_GAIN or NOT_ADOPTED. On any
   other verdict it stops, and every RTR item after it drops.
3. The screen's fits, both after `pkoh-gate`, `iso-verdict`, `rtr-open` and `rtr-selftest`:
   - `rtr-r2-s0`: GPU share 0.42, cap 0.40, 7 GB;
   - `rtr-r2-s0-hp`: GPU share 0.75, cap 0.72, 12 GB.
4. `rtr-compare`, then `rtr-calls`, then `rtr-gate` (`outputs/rtr/calls.json`, `screen.verdict` ADOPT_FOR_FULL_RUN).
5. The full run, after `pkoh-gate` and `rtr-gate`:
   - `rtr-r2-s1`, `rtr-r2-s2`, `rtr-r2-s1-hp`, `rtr-r2-s2-hp`, with the screen's sizes;
   - then `rtr-compare-full` and `rtr-calls-full`.

```
python outputs/host_ops/retry_cmd.py --tries 3 --wait 90 --cuda -- python outputs/host_ops/cuda_alloc.py --frac 0.4 --gc 0.9 -- outputs/mp_unified/zrtr.py fit --name scr-rtr-r2-s0
python outputs/mp_unified/zrtr.py compare-all
python outputs/mp_unified/zrtr.py calls          (outputs/rtr/calls.json and .md)
```

Only `outputs/rtr/calls.json` and `calls.md` are committed. `rtr-calls-full` rewrites them with both blocks, so the screen's are
committed before the full run ends.

The self-test, `python outputs/mp_unified/zrtr.py --selftest`, passed on the laptop before this commit. It covers:

- the names and pairs;
- DEP's filed calls opening RTR only when three layers stay; any other DEP verdict, a file that is not DEP's, or no file stops;
- zg6r's toy carve (106 typed of 164 edges), parameters drawn at random:
  - one round equals zg6r's stack bit for bit;
  - the forward is two rounds;
  - with the typed edges taken out, two rounds equal zg6's own layers by hand (difference 0.0);
  - with e at 0, round two still moves the score;
  - a checkpointed training forward and backward gives e and every last-layer parameter a gradient;
  - 173,114 parameters at hidden 16 (zg6's, G6r's and e's);
  - the state loads strictly under the patch, and a plain zg6r refuses it;
- the patches (names, stamp, declaration, ISO's own start on REPLACE, the rounds), and that they are put back, also on a failure;
- DEP's rule on sixteen made-up reads.

## 9. Amendment A1, 2 November 2026, before any RTR fit: ML1's start read under ML1's names

ISO's verdict is KEEP (docs/ISO_ZG6R_OWN_START.md, section 11), so RTR's base is ML1's zg6r and RTR's fits start from
ML1's seed-0 zspk encoder. That start could not be read under RTR's names: LTG's first fit stopped on it 2 s in, before any
train, and docs/LTG_NEXT_HOP_GNN.md (amendment A1) gives the cause. No RTR fit had started.

**The change.**

- The fit context in `outputs/mp_unified/zrtr.py` installs `ml1start.ml1_load` in zml1s on KEEP, where it left ML1's loader. That is ML1's loader
  with ML1's declaration, stage and key in force for the length of the load only. The tensors and the record are the loader's,
  bit for bit, and a zspk fit without ML1's read stamp is still refused. REPLACE is unchanged (ISO's `own_load`).
- Each fit's code block also names `ml1start_sha256`. DEP's CODE_SKIP, which RTR uses, skips it in the code comparison with the base.

**What does not change.** Every earlier section: the arm, the fits, the base, the reads, the rule and what follows.

**The queue.** RTR's fits also wait on `ml1start-check` (LTG's A1), which runs this stage's self-test on the host among
others.

**Self-test.** `python outputs/mp_unified/zrtr.py --selftest` passed on the laptop's CPU before this commit. It now checks that the KEEP context
installs `ml1_load` and that the code block names ml1start.

## 10. Amendment A2, 3 November 2026, before any RTR fit: a fit's read keeps p@swa only

**What happened.** A GNN fit's read takes about as long as its train: ML1's zg6r at seed 0 trained in 4,320 s and read in
4,618 s. lean_gpu's read scores every candidate a fit saved, that is each epoch's state and the SWA state (nine at 8 epochs).
RTR's rule reads p@swa alone, and so do the compares, DEP's slices, C1R's intervals and the knee. Each finds p@swa by its name.

**The change.**

- `outputs/mp_unified/readswa.py` (new). `on_read()` wraps `lean_gpu.load_models` so that it cuts the fit's candidates to
  p@swa before it builds the models. lean_gpu's read then writes read.json and each carve's npz from that list. The wrap is
  undone after, also on a failure, and entered twice it wraps once. A fit without p@swa, or without its state, is refused.
- `zrtr.py`'s fit context enters `on_read()` for the whole fit, on REPLACE and on KEEP. The train never
  calls `load_models`, so only the read changes.
- Each fit's code block also names `readswa_sha256`. A fit's code is compared with its base's over the base's keys only, so
  the new key is not compared, as for the stage modules' own.

**Why no number changes.** `lean_gpu.read_carve` runs each candidate's model on the same batch, one forward per candidate, and
no candidate's forward depends on another's. In eval zg6r draws nothing (its generator draws only while training). So p@swa's
rows (the top five, the hits, the first questions' scores) are the rows a full read gives, up to the GPU's own read noise (ML1's
amendment A1). readswa's self-test runs the real `read_carve` on a toy carve over several batches: p@swa's rows and the
references are the same read alone and among nine.

**The records.** read.json and each carve's npz list p@swa alone, so their `candidates` say which read a fit had. models.pt keeps
every state, so a full read can still be made from it.

**Which fits.** Every RTR fit (none has started).
A fit that started before the check passed runs the old code and reads all nine candidates; its read.json lists them.

**The queue.** A new CPU item, `readswa-check`, runs `python outputs/mp_unified/readswa.py --check-host` on the host:

- readswa's self-test;
- the self-tests of the five stages (zltg, zdep, zrtr, zira, zisoz), each in its own process;
- ML1's seed-0 zg6r fits of both WQ splits, loaded as the read loads them, under `on_read()` and lean_screen's zg6r arm. Each
  must give one ZG6R model, p@swa, holding p@swa's state, with lean_gpu's own `load_models` back after. The check also
  confirms that `on_read()` patches the lean_gpu module that zg6r's read calls.

It writes `outputs/readswa/check.json`. Every GNN fit of LTG, DEP, zg6-z, RTR and IRA not yet started waits on it.

**Self-tests.** `python outputs/mp_unified/readswa.py --selftest` and `python outputs/mp_unified/zrtr.py --selftest` passed on
the laptop's CPU before this commit. The stage's self-test now checks that its fit context installs
`on_read()` on REPLACE and on KEEP, that the code block names readswa, and that lean_gpu's own `load_models` is back after.

**What does not change.** Sections 1 to 9: the rounds, the fits, the base, the reads, the rule and what follows.

## 11. Amendment A3, 3 November 2026, before any RTR fit: the seed-0 stop, and RTR's open

Both changes come from DEP's seed-0 stop (docs/DEP_GNN_DEPTH.md section 12, `outputs/mp_unified/s0stop.py`).

**RTR's open.** RTR's fits wait on `rtr-open`, which passes when DEP's screen is NO_GAIN or NOT_ADOPTED. Under DEP's stop,
DEP's NOT_ADOPTED can stand in `outputs/dep/s0stop.json` while `outputs/dep/calls.json` says INCOMPLETE. So `rtr-open` now
runs `python outputs/mp_unified/s0stop.py gate --stage dep --want NO_GAIN,NOT_ADOPTED`, still after `dep-calls`. Without a
STOP the gate reads calls.json's `screen.verdict`, as verdict_gate.py did.

**RTR's own stop.** RTR's screen rule is DEP's sixteen-read rule: zrtr.py calls `zdep.decide_screen`. Eight of the sixteen
reads are fixed once `scr-rtr-r2-s0` is read: the WQ-musique split's six datasets, and musique's 3-hop and 4-hop slices.
When one of the eight is a LOSS, no read of `scr-rtr-r2-s0-hp` can change the screen's verdict, so that fit is not run.

- `rtr-s0stop`: CPU 1, 8 GB, after `rtr-r2-s0`. It runs `s0stop.py check --stage rtr`, which runs `zrtr.py compare-all`
  and `zrtr.py calls`. The decision is GO on any problem or missing read, STOP on a LOSS among the eight, and GO otherwise.
- `rtr-r2-s0-hp` also waits on `rtr-s0stop`, and its command is wrapped in `s0stop.py run --stage rtr --`.
- The records are `outputs/rtr/s0stop.json` and, on a STOP, `outputs/rtr/s0stop_skip.json`. On a STOP, `rtr-calls`'s
  `screen.verdict` is INCOMPLETE, because the WQ-hotpotqa reads are missing. The screen's verdict is the NOT_ADOPTED in
  s0stop.json. `rtr-gate` wants ADOPT_FOR_FULL_RUN, so the full run does not start.

**Why no number changes.** The stop decides only what DEP's sixteen-read rule already decides. The floors, the rows and the
calls are zrtr.py's own. On GO the fit runs as before.

**Self-test.** `python outputs/mp_unified/s0stop.py --selftest` passed on the laptop's CPU before this commit. It checks
RTR's fit names, OUT and declaration against zrtr.py, and the gate's three codes.

**What does not change.** Sections 1 to 10: the rounds, the fits, the base, the reads, the rule and what follows.

## 12. Amendment A4, 3 November 2026, before any RTR fit: the full run before the freeze is the stack's

**What happened.** On 3 November the user asked to finish faster. RTR's full run is four fits, about 12 hours of the card,
and four other GNN forms are screened beside RTR (docs/STK_GNN_STACK.md section 1). No RTR fit has run: RTR opens only on
DEP's NO_GAIN or NOT_ADOPTED.

**The change.** Section 6's ADOPT_FOR_FULL_RUN row now says that RTR joins the stack of docs/STK_GNN_STACK.md, declared in
the same commit. The stack is zg6r with every GNN form whose screen passes, and its full run is the GNN's one full run before
the freeze.

- When IRA's screen passes too, IRA is the member, and RTR's two rounds come in it (STK section 2).
- If RTR is the stack's only member, the stack is RTR's own full run. Its items are released, and section 4's full-run rule
  decides.
- With two or more members, the stack is read under DEP's sixteen-read rule, which RTR uses, and the other members' rules
  (STK section 4).
- On the stack's ADOPT, the stack is the GNN of record, and Claim 1 and Claim 3 are measured again on it.
- Otherwise RTR's own full run follows after the freeze, unless STK section 6 picks it sooner. The next form, three rounds,
  follows after the freeze.

**The queue.** `rtr-r2-s1`, `rtr-r2-s2`, `rtr-r2-s1-hp` and `rtr-r2-s2-hp` now also wait on `hold-full-for-stack`, a name
no job has. The feeder keeps them waiting and does not drop them. `rtr-compare-full` and `rtr-calls-full` wait on those fits.
`rtr-gate` runs as before.

**What does not change.** Sections 1 to 11, apart from that row: the rounds, the fits, the base, the reads, the rule, RTR's
open, the seed-0 stop, and the NO_GAIN, NOT_ADOPTED and INCOMPLETE rows.
