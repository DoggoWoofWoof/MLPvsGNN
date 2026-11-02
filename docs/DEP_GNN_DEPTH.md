# DEP: the GNN of record with a fourth message-passing layer, on PKO's pools

Declared 2 November 2026, before any DEP fit. Stage module: `outputs/mp_unified/zdep.py`. It imports `zml1s.py`, `ziso.py`,
`zg2.py`, `zg6.py` and `zg6r.py` and edits none of them.

## 1. The question

- **What the program names.** MUSIQUE_DIAGNOSIS step 3: "Depth in the universal models. Use three or four message-passing layers,
  or read-time rounds that go back to the neighbours of the top candidates." ML1's section 13 orders the GNN's next forms: Claim 1
  (C1R, filed), the isolation arms (ISO, running), then the depth form, graded on musique's 3- and 4-hop questions, then Claim 3
  timed again.
- **Why depth, now.** The knee (PKO_DEPTH_KNEE section 7) found DEPTH_FLAT on zrckp. musique's R@5 stays at 0.5051 at every pool
  depth while the pool's reach goes from 0.922 to 0.976, so the ranker is the limit. C1R (section 5) found the MLP keeps 0.886
  [0.832, 0.940] of zg6r's gain on musique in-domain, so message passing is where musique's ranking comes from.
- **What going back means here.** The user ruled on 5 October that going back is neighbour summarisation by message passing, not
  EM rounds. S6 part 14 found that read-time EM rounds add about nothing. zg6r has three layers, so a candidate's state sees the
  pool graph three edges out. musique has 2-, 3- and 4-hop questions.
- **So the question is:** does a fourth message-passing layer in the GNN of record raise R@5 on the twelve reads, and on musique's
  3-hop and 4-hop questions?

## 2. What runs

- **The arm.** The GNN of record after ISO, with four layers instead of three. zg6's layer count (taken from zg2's, which zg6r takes
  from zg6) goes from 3 to 4. Each layer is zg6's, with G6r's typed messages, and the fourth is built and initialised as the others
  are.
  - Everything else is the base's: the trainer, the card and its epochs, PKO's roots and typed links, the KT input, the 64-wide
    state, the seed offsets, and the read cap of 32,768.
  - The message-passing stack and readout go from 180,218 to 220,877 parameters at hidden 128: zg6's part from 121,802 to 159,373,
    and G6r's from 58,416 to 61,504.
- **The base, fixed by ISO's filed verdict** (`outputs/iso/verdict.json`, ISO section 6).
  - On KEEP: zg6r, ML1's fits `scr-ml1-zg6r-s<k>` (with `-hp` on WQ-hotpotqa). The four-layer arm starts from the same zspk seed-0
    encoder.
  - On REPLACE: zg6r-m, ISO's fits `scr-iso-zg6rm-s<k>`. The four-layer arm takes its own draw, through ISO's own-start patches.
  - On anything else, no DEP fit, compare or call runs.
- **How the count is set.** zg6 and zg6r bind the count, and every parameter name and size built from it, when they are imported.
  So `zdep.import_deep` sets zg2's count to 4 before zg6 is first imported in the fit's process. Then it checks that zg6, zg6r, their
  per-layer name lists and G6r's parameter names all say four. A fit refuses when zg6 was imported before.
- **The fits (screen): seed 0 on both WQ splits.** `scr-dep-l4-s0` (WQ-musique) and `scr-dep-l4-s0-hp` (WQ-hotpotqa), on the host's
  GPU as the base's fits were.
  - Each is stamped under the key `dep` with its layer and parameter counts, its base fit and ISO's verdict, together with zml1s's
    stamp (device, environment, gate, code).
- **The compares.** Each DEP fit against the base's fit of its seed and split, through zwq's `run_compare`, as ML1's.

## 3. The reads

- **ML1's twelve reads**: two splits by six datasets, p@swa on the s1eval carves. The zero-shot reads are among them.
- **Four slices**: musique's 3-hop questions (760 of 2,417) and 4-hop questions (405), on both splits. musique's ids carry the hop
  count (`3hop1`, `3hop2`, `4hop1` ...); no other dataset's do.
  - Each slice's R@5 comes from the fit's musique s1eval read file: p@swa's row, over the slice's questions that have a gold, as
    lean_gpu's `metrics_of` counts it.
  - The base's slices come from its fits' read files at seeds 0, 1 and 2. All six of ML1's zg6r files are on the host. Only their
    keys and shapes were checked before this commit; no value was read.
  - Each file's R@5 over all its questions must equal its compare's musique R@5 within 0.00006, or the stage is INCOMPLETE.

## 4. The rule (fixed now)

**The screen (seed 0).** Per read, over the twelve reads and the four slices (sixteen):

- The floor is max(0.0075, 2 x the base's spread). The spread is the root mean square of the base's seed-1 and seed-2 differences
  from its seed 0 (`zml1.spread`, then `zml1.pair_floor` with the base's spread twice), as DOB's.
- The delta is DEP's seed-0 R@5 minus the base's seed-0 R@5.
- **LOSS:** the delta is at or below minus the floor. **GAIN:** at or above the floor. **WITHIN:** otherwise.
- **ADOPT_FOR_FULL_RUN:** at least one GAIN and no LOSS. **NO_GAIN:** every read WITHIN. **NOT_ADOPTED:** any LOSS.

**The full run** (seeds 1 and 2, added by an amendment that adds items only):

- ML1's rule (`zml1.verdict`) on the twelve reads against the base's three seeds.
- Each slice is called by `zml1.call` with the pair floor of both arms' spreads.
- **ADOPT:** zml1 says ADOPT and no slice is a LOSS. **NOT_ADOPTED** otherwise.

**INCOMPLETE** when any of these holds:

- a fit, a compare or a read file is missing;
- ISO's verdict is not REPLACE or KEEP;
- a fit's train or read stamp does not name four layers, its base fit and ISO's verdict;
- its encoder's start is not its base's (KEEP: the same zspk fit, seed and sha256, copied at least once; REPLACE: its own draw, no
  copy);
- it ran on another device than its base;
- its trainer's code differs from its base's (stage modules aside);
- the base's seed-0 R@5 in DEP's compare differs from its own stage's;
- a read file's R@5 differs from its compare's.

A verdict closes only the form it tested.

## 5. Reported beside, deciding nothing

- The mean delta over the twelve reads.
- The parameter counts (each fit's stamp), seconds per fit and the GPU peaks (`ml1_timing.json`).
- Where the base's spread makes a floor wide (metaqa's reads), the floor is printed beside the delta.

## 6. What follows (fixed now)

| outcome | what follows |
| --- | --- |
| ADOPT_FOR_FULL_RUN | The full run, seeds 1 and 2 on both splits, in an amendment that adds items only; then section 4's full-run rule. On ADOPT, the four-layer GNN is the GNN of record. Claim 1 (C1R's intervals) and Claim 3 (cold timing) are measured again on it, and the next depth form is five layers, in its own file. |
| NO_GAIN or NOT_ADOPTED | Three layers stay. The next depth form is step 3's other one: read-time rounds that go back to the neighbours of the top candidates by message passing, in its own file. |
| INCOMPLETE | What is missing is fixed in an amendment, before any number is read. |

## 7. Risks, stated before any number

- **Seed 0 only in the screen.** It is a one-fit screen. The full run decides any adoption.
- **The base waits on ISO.** DEP's fits start after ISO's verdict. The GPU is held by ISO's fits until then, so the wait costs no
  GPU time.
- **Depth is not reach.** The knee says the deeper pool's extra reach is not ranked. A fourth layer changes the ranking, not the pool.
- **Deeper message passing can blur states.** Candidates' states can grow alike over more layers. The slices and the twelve reads show
  where it helps or hurts.
- **Memory.** The layers are checkpointed in training, so the fourth adds one stored layer input per batch. The read keeps one
  layer's tensors at a time. The items keep zg6r's caps. A fit that runs out of memory stops INCOMPLETE, and the caps change in an
  amendment before it runs again.
- **Time.** A fourth layer adds about a third of the message-passing time. ML1's zg6r read took 77 minutes on WQ-musique.
- **The self-test's model is a toy.** It shows that four layers build, match zg6r's by-hand score and train. It says nothing about R@5.

## 8. How it runs

Feeder items on the host:

1. `dep-l4-s0`: GPU share 0.42, cap 0.40, 7 GB. `dep-l4-s0-hp`: GPU share 0.75, cap 0.72, 12 GB. Both come after `iso-verdict`, behind
   ISO's hotpotqa fits in the GPU queue.
2. `dep-compare`, after both fits.
3. `dep-calls`, after the compare.

```
python outputs/host_ops/retry_cmd.py --tries 3 --wait 90 --cuda -- python outputs/host_ops/cuda_alloc.py --frac 0.40 --gc 0.9 -- outputs/mp_unified/zdep.py fit --name scr-dep-l4-s0
python outputs/mp_unified/zdep.py compare-all
python outputs/mp_unified/zdep.py calls          (outputs/dep/calls.json and .md)
```

The self-test, `python outputs/mp_unified/zdep.py --selftest`, passed before this commit. It covers:

- the names, the pairs, and the base each ISO verdict picks; any other verdict stops;
- in a process of its own:
  - the four-layer import: zg6's and zg6r's counts, their name lists and parameter counts;
  - zg6r's toy carve with four layers: the score in eval against zg6r's by-hand layers (within 1e-4), a checkpointed training
    forward and backward, and a gradient on each of the fourth layer's twelve parameters;
  - the refusal of a second import;
- the patches (names, stamp, declaration, ISO's own start on REPLACE), and that they are put back, also on a failure;
- the slices from a made-up read file;
- the screen's and the full run's rules on made-up reads, including INCOMPLETE on a hole.

## 9. Amendment A1, 2 November 2026, before any DEP fit: the full run queued behind the screen's verdict

Section 6 adds the full run "in an amendment that adds items only" when the screen gives ADOPT_FOR_FULL_RUN. If that amendment
waited for the screen's numbers, the GPU would sit idle between `dep-calls` and the amendment. So the items are added now, before
any DEP fit has run, behind a gate that lets them start only on ADOPT_FOR_FULL_RUN.

Sections 1 to 8 are unchanged. The fits, the bases, the reads and both rules are the ones zdep.py already holds:

- its FITS has seeds 0, 1 and 2 on both splits;
- `compare-all` compares the full run's fits once they exist;
- `calls` applies section 4's full-run rule once all six fits are read.

zdep.py is not changed.

**The gate.** `dep-gate` comes after `dep-calls` and runs:

```
python outputs/mp_unified/verdict_gate.py --json outputs/dep/calls.json --key screen.verdict --want ADOPT_FOR_FULL_RUN
```

- It exits 0 only on ADOPT_FOR_FULL_RUN.
- On NO_GAIN, NOT_ADOPTED or INCOMPLETE it exits 1. A missing record or key exits 2.
- After a non-zero exit and its one retry, the feeder drops everything queued behind the gate.

**The items**, after `iso-zg6z-report`, the last item queued so far:

1. `dep-gate`: CPU 1, 1 GB, after `dep-calls`.
2. `dep-l4-s1` and `dep-l4-s2`: the musique split, with `dep-l4-s0`'s line (share 0.42, cap 0.40, 7 GB) and deps
   `pkoh-gate,dep-gate`. The two share the card.
3. `dep-l4-s1-hp` and `dep-l4-s2-hp`: the hotpotqa split, with `dep-l4-s0-hp`'s line (share 0.75, cap 0.72, 12 GB) and the same
   deps. They run one at a time.
4. `dep-compare-full`, after the four fits: `zdep.py compare-all`. The two screen compares are kept and the four new ones run.
5. `dep-calls-full`, after the compare: `zdep.py calls`. It rewrites `outputs/dep/calls.json` and `.md` with the screen block
   and the full-run block. The screen block is recomputed from the same compares and must match the filed screen, which is
   committed before the full run's fits end.

**Bases.** The base's seeds 1 and 2 exist before the gate can open: ML1's zg6r s1 and s2 on KEEP, and ISO's zg6r-m s1 and s2 on
REPLACE. ISO's verdict needs all six of its fits, and `dep-calls` needs ISO's verdict.

**What does not change.** The screen decides only whether the full run starts. The full run alone decides adoption (section 4).
If the screen's verdict is anything other than ADOPT_FOR_FULL_RUN, section 6's next form (read-time rounds) is declared in its
own file.

**Time.** This adds about 10 hours of GPU behind the screen: two musique-split fits side by side, then two hotpotqa-split fits
one after the other. Each is a fourth longer than zg6r's (section 7).

**Self-test.** `python outputs/mp_unified/verdict_gate.py --selftest` passed before this commit. It covers:

- each of the four verdicts against ADOPT_FOR_FULL_RUN, and a list of wanted values;
- a top-level key and a nested key;
- a value that is not a string;
- a missing key, a step into null, and a missing file.

## 10. Amendment A2, 2 November 2026, before any DEP fit: ML1's start read under ML1's names

ISO's verdict is KEEP (docs/ISO_ZG6R_OWN_START.md, section 11), so DEP's base is ML1's zg6r and DEP's fits start from
ML1's seed-0 zspk encoder. That start could not be read under DEP's names: LTG's first fit stopped on it 2 s in, before any
train, and docs/LTG_NEXT_HOP_GNN.md (amendment A1) gives the cause. No DEP fit had started.

**The change.**

- The fit context in `outputs/mp_unified/zdep.py` installs `ml1start.ml1_load` in zml1s on KEEP, where it left ML1's loader. That is ML1's loader
  with ML1's declaration, stage and key in force for the length of the load only. The tensors and the record are the loader's,
  bit for bit, and a zspk fit without ML1's read stamp is still refused. REPLACE is unchanged (ISO's `own_load`).
- Each fit's code block also names `ml1start_sha256`. CODE_SKIP, which LTG, RTR and IRA use as well, now holds it beside zml1s's and ziso's, so the code
  comparison with the base skips it: these are stage modules, not the trainer's code.

**What does not change.** Every earlier section: the arm, the fits, the base, the reads, the rule and what follows.

**The queue.** DEP's fits also wait on `ml1start-check` (LTG's A1), which runs this stage's self-test on the host among
others.

**Self-test.** `python outputs/mp_unified/zdep.py --selftest` passed on the laptop's CPU before this commit. It now checks that the KEEP context
installs `ml1_load` and that the code block names ml1start.

## 11. Amendment A3, 3 November 2026, before DEP's fits start: a fit's read keeps p@swa only

**What happened.** A GNN fit's read takes about as long as its train: ML1's zg6r at seed 0 trained in 4,320 s and read in
4,618 s. lean_gpu's read scores every candidate a fit saved, that is each epoch's state and the SWA state (nine at 8 epochs).
DEP's rule reads p@swa alone, and so do the compares, DEP's slices, C1R's intervals and the knee. Each finds p@swa by its name.

**The change.**

- `outputs/mp_unified/readswa.py` (new). `on_read()` wraps `lean_gpu.load_models` so that it cuts the fit's candidates to
  p@swa before it builds the models. lean_gpu's read then writes read.json and each carve's npz from that list. The wrap is
  undone after, also on a failure, and entered twice it wraps once. A fit without p@swa, or without its state, is refused.
- `zdep.py`'s fit context enters `on_read()` for the whole fit, on REPLACE and on KEEP. The train never
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

**Which fits.** Every DEP fit that starts after `readswa-check` passes.
A fit that started before the check passed runs the old code and reads all nine candidates; its read.json lists them.

**The queue.** A new CPU item, `readswa-check`, runs `python outputs/mp_unified/readswa.py --check-host` on the host:

- readswa's self-test;
- the self-tests of the five stages (zltg, zdep, zrtr, zira, zisoz), each in its own process;
- ML1's seed-0 zg6r fits of both WQ splits, loaded as the read loads them, under `on_read()` and lean_screen's zg6r arm. Each
  must give one ZG6R model, p@swa, holding p@swa's state, with lean_gpu's own `load_models` back after. The check also
  confirms that `on_read()` patches the lean_gpu module that zg6r's read calls.

It writes `outputs/readswa/check.json`. Every GNN fit of LTG, DEP, zg6-z, RTR and IRA not yet started waits on it.

**Self-tests.** `python outputs/mp_unified/readswa.py --selftest` and `python outputs/mp_unified/zdep.py --selftest` passed on
the laptop's CPU before this commit. The stage's self-test now checks that its fit context installs
`on_read()` on REPLACE and on KEEP, that the code block names readswa, and that lean_gpu's own `load_models` is back after.

**What does not change.** Sections 1 to 10: the depth, the fits, the base, the reads, the rule and what follows.

## 12. Amendment A4, 3 November 2026, before any DEP number is seen: the seed-0 stop

**What happened.** The screen's two fits run one after the other. The WQ-hotpotqa fit has the card alone for about 4.5 hours
(share 0.75). Section 4's screen rule gives NOT_ADOPTED on one LOSS among its sixteen reads. Eight of them are fixed once
`scr-dep-l4-s0` is read: the WQ-musique split's six datasets, and musique's 3-hop and 4-hop slices. The base's three seeds are
filed, and DEP's one read of each comes from that fit. So when one of the eight is a LOSS, no read of `scr-dep-l4-s0-hp` can
change the screen's verdict.

`dep-l4-s0` ended at 02:26 IST (rc 0), and the feeder started `dep-l4-s0-hp` at 02:28, before this amendment. No DEP compare
has run, and no DEP number has been seen.

**The change.** `outputs/mp_unified/s0stop.py` (new) has three verbs.

- `check --stage dep` runs `zdep.py compare-all` and `zdep.py calls`, as the queue's items do, and reads the calls record they
  wrote. It decides:
  - GO when the record lists a problem, or when one of the eight reads has no row;
  - STOP when one of the eight rows is a LOSS;
  - GO otherwise.

  It writes `outputs/dep/s0stop.json`: the decision, the verdict (NOT_ADOPTED on a STOP), the eight rows, the LOSS rows and the
  problems. The record also holds the sha256 of the seed-0 fit's read.json and of the calls record. A failure inside the check
  is a problem, so it gives GO. The check always exits 0.
- `run --stage dep -- <command>` runs the WQ-hotpotqa fit's command and returns its exit code. There is one exception: when
  s0stop.json says STOP and was made from the seed-0 fit's read.json as it is now, `run` writes `outputs/dep/s0stop_skip.json`
  and exits 0 without running the fit.
- `gate --stage dep --want <verdicts>` gives the screen's verdict: NOT_ADOPTED on a current STOP, else calls.json's
  `screen.verdict`. It exits 0 when the verdict is one of `--want`, 1 when it is not, and 2 when calls.json is missing. These
  are verdict_gate.py's codes.

**Why no number changes.** The stop decides only what section 4's rule already decides. A LOSS among the eight WQ-musique
reads gives NOT_ADOPTED whatever the WQ-hotpotqa reads are. The floors, the rows and the calls are zdep.py's own, from its own
`calls`. On GO the fit runs as before, and every later step is the same.

**The records.** On a STOP, `dep-calls` still runs. Its `screen.verdict` is INCOMPLETE, because the WQ-hotpotqa reads are
missing. The screen's verdict is the NOT_ADOPTED in s0stop.json, and the gate reads it from there. The full run does not
start: `dep-gate` wants ADOPT_FOR_FULL_RUN.

**The queue.**

- `dep-s0stop`: CPU 1, 8 GB, after `dep-l4-s0`.
- `dep-l4-s0-hp` is already running, so the wrapper cannot hold it. On a STOP:
  1. `dep-compare` is pointed at `dep-s0stop` in place of `dep-l4-s0-hp`, so the cancel drops nothing;
  2. the running fit is then cancelled. Its folder is never read.

  On GO it runs on. If the fit is ever queued again, it is wrapped in `s0stop.py run --stage dep --`.
- `rtr-open` and `ira-open` run `s0stop.py gate --stage dep --want NO_GAIN,NOT_ADOPTED` in place of verdict_gate.py, still
  after `dep-calls`. So RTR and IRA open on a STOP.
- The same stop holds the seed-0 WQ-hotpotqa fits of LTG, RTR and IRA (their amendments of 3 November) and of GNX (its
  declaration).

**Self-test.** `python outputs/mp_unified/s0stop.py --selftest` passed on the laptop's CPU before this commit. It checks:

- the deciding reads of both rules: eight under DEP's sixteen-read rule, four under LTG's in-domain rule;
- each stage's fit names, OUT and declaration, taken from its own module;
- the decision on toy records. A LOSS on WQ-musique stops the fit. A LOSS on WQ-hotpotqa or on a read beside the rule, a
  problem, a missing read, or no LOSS runs it;
- the check's record on a toy tree, including no read, a stage step that fails, and a calls record the check did not write;
- the wrapper: a STOP skips the command, and a stale record or GO runs it with its exit code;
- the gate's three codes.

**What does not change.** Sections 1 to 11: the depth, the fits, the base, the reads, both rules and what follows. A STOP
closes only this form, and section 6's next form (docs/RTR_GO_BACK_ROUNDS.md) opens as before.

## 13. Amendment A5, 3 November 2026, before DEP's screen verdict: the full run before the freeze is the stack's

**What happened.** On 3 November the user asked to finish faster. DEP's full run is four fits, about 12 hours of the card,
and four other GNN forms are screened beside DEP. docs/STK_GNN_STACK.md section 1 has the arithmetic. `dep-l4-s0-hp` is
running. `dep-s0stop` read `scr-dep-l4-s0` at 02:40 IST and gave GO: all eight WQ-musique reads are WITHIN (STK section 7).
The screen's verdict waits on `scr-dep-l4-s0-hp`, and no WQ-hotpotqa number has been seen.

**The change.** Section 6's ADOPT_FOR_FULL_RUN row now says that DEP joins the stack of docs/STK_GNN_STACK.md, declared in
the same commit. The stack is zg6r with every GNN form whose screen passes, and its full run is the GNN's one full run before
the freeze.

- If DEP is the stack's only member, the stack is DEP's own full run. Its items are released, and section 4's full-run rule
  decides.
- With two or more members, the stack is read under DEP's sixteen-read rule and the other members' rules (STK section 4).
- On the stack's ADOPT, the stack is the GNN of record, and Claim 1 and Claim 3 are measured again on it.
- Otherwise DEP's own full run follows after the freeze, unless STK section 6 picks it sooner. The next depth form, five
  layers, follows after the freeze.

**The queue.** `dep-l4-s1`, `dep-l4-s2`, `dep-l4-s1-hp` and `dep-l4-s2-hp` now also wait on `hold-full-for-stack`, a name
no job has. The feeder keeps them waiting and does not drop them. `dep-compare-full` and `dep-calls-full` wait on those fits.
`dep-gate` runs as before.

**What does not change.** Sections 1 to 12, apart from that row: the depth, the fits, the base, the reads, both rules, the
seed-0 stop, and the NO_GAIN, NOT_ADOPTED and INCOMPLETE rows. RTR and IRA still open on DEP's NO_GAIN or NOT_ADOPTED
(section 12).
