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
