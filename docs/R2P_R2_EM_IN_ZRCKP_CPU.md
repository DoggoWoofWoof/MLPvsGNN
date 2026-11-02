# R2P: R2's EM inside zrckp's fit, on PKO's pools, on the host's CPU

Declared on 2 November 2026, before any R2P fit. The code is `outputs/mp_unified/zr2p.py`. It imports `zml1s.py`,
`zdob.py`, `zr2br.py`, `zwq.py` and `zp1c3p.py` unchanged and patches them in process. No committed file is edited.
Nothing here is a result.

## 1. The question, and why now

- ML1 (docs/ML1_REFITS_ON_PKO.md, e248d55) filed zrckpt NOT_ADOPTED with no GAIN (WITHIN 12). Its section 6 says what
  follows: "zrckp stays. Its next form is U2n's section 6, under WQ." Its section 13 says: "the MLP's work: U2n section 6
  under WQ."
- U2n's section 6 (docs/U2N_TYPED_CHANNELS.md), for zrckpt NOT_ADOPTED with no GAIN: "R2's EM inside the MLP's fit, after
  GL's R2 verdict and the user's ruling on zrckpb (one declared file)". This is that file.
- **GL's R2 verdict is filed.** R2's GNN half under WQ, R2EG, filed zg6e NOT_ADOPTED (1 GAIN, 2 LOSS).
- **The user ruled on zrckpb on 2 November 2026:** "yeah zrckpb is fine, reading vectors isnt message passing it is
  more akin to beam search". R2's flag 3 (docs/R2_BRIDGE_EM.md section 2) is closed by it. Section 8 there records the
  ruling.
- **Why it matters for the paper.** C1R (8c5dabf, a0bad12) filed CLAIM1_NOT_ALL. Two of the in-domain reads that do not
  hold are multi-hop:
  - WQ-hotpotqa musique, 0.886 [0.832, 0.940], MISSES;
  - WQ-musique hotpotqa, 0.903 [0.885, 0.920], NOT_SHOWN.
- **Why the bridge head.** The knee (8c5dabf) filed DEPTH_FLAT: musique's R@5 is 0.5051 at every cut, so the ranker is
  the limit, not the pool. R2's bridge head is the MLP form aimed at that gap. Each row is read against the first
  stage's top five rows, a beam fixed before any row is scored.

**What happened before, on other pools.** zrckpb has never been fitted on PKO's pools. Its earlier results:

| stage | zrckpb form | pools | verdict |
|---|---|---|---|
| R2t | zrckpb on the L-splits | earlier | NOT_ADOPTED (a GAIN and a LOSS) |
| R2WS | zrckpb under WQ, seeds 1 and 2 | U2s's | NOT_ADOPTED |
| R2EW | zrckpbe, R2's head with link flags, under WQ | U2s's | NOT_ADOPTED (1 LOSS, no GAIN) |

R2WS lost on both reads above at both seeds:

- WQ-hotpotqa musique: −0.0175 and −0.0215;
- WQ-musique hotpotqa: −0.0125 and −0.0115.

Section 7 states this risk before the numbers. The program's rule stands: no model change is adopted on today's pools.
PKO's pools are the pools of record, and R2P asks the question there.

## 2. What runs

**The arm.** zrckpb, from `zr2br.py` unchanged: zrckp's model (`zp1c3p.ZRCKP`) plus R2's bridge head.

- R2's part has 59,537 parameters.
- R2's declared constants: K 5, r 32, LAM 1.0, prior_c 60, seed_offset 5502, head [3, 16, 1], w1_scale 0.1.
- It trains with R2's EM term (`zr2br.r2_em`): lean_gpu's listwise loss plus LAM times the batch mean of L_br, over the
  batch's questions with a term.
- **The bridges.** Per question, the five rows with the highest first-stage RRF (`c_rrf`), computed once at carve load.
  A row reads the bridges' frozen vectors. It reads no score and no edge: R2's SETTINGS hold `score_input: False` and
  `edges_read: False`.

**The trainer.** zp1c3p's train and read with zp1c3p's arm swapped to zrckpb (R2GW's swap).

- It runs under `zwq.on_wq` on PKO's roots, through `zml1s.on_ml1`, with PKO's gate READY.
- It takes the same path as DOB's base. Two things differ: the arm, and R2's term in the loss.
- zml1s's `fit` runs it, with its train and read replaced by zr2p's for the length of the fit:
  - a fit read before is kept;
  - a fit trained but not read is only read;
  - otherwise the fit trains, then reads.

**The device.** DOB's device: the host's CPU at 6 threads, torch 2.8.0+cpu, with the GPU hidden. zr2p uses zdob's
`cpu_env`, `cpu_ok` and card.

**The inputs and the read.**

- The pick is the whole pick, lean_mlp8's nine blocks, as in DOB's base. zr2p checks it and drops nothing.
- The read is p@swa on the six s1eval carves, at zrckp's read cap of 65,536.

**The base.** DOB's CPU zrckp at seeds 0, 1 and 2 on both splits: `scr-dob-zrckp-s{0,1,2}` and
`scr-dob-zrckp-s{0,1,2}-hp`. They are DOB's first items. R2P adds no base fit.

**The fits.**

- The screen: `scr-r2p-zrckpb-s0` (WQ-musique) and `scr-r2p-zrckpb-s0-hp` (WQ-hotpotqa).
- The full run (section 6): seeds 1 and 2 on both splits.

**The compares.** They run through `zwq.run_compare` from the repository's root.

- Each R2P fit is compared with DOB's base fit of the same seed and split: `scr-r2p-zrckpb-vs-zrckp-s{k}[-hp]`.
- R2P runs its own copies of the base's null: DOB's base at seeds 1 and 2 against its seed 0,
  `scr-r2p-zrckp-s{1,2}-vs-s0[-hp]`. So R2P's verdict does not wait for DOB's 16 drop fits.

## 3. The reads

The twelve: on each split, metaqa, squad, musique, hotpotqa, 2wiki and webqsp, each a p@swa R@5 on s1eval.

| split | trains on | zero-shot |
|---|---|---|
| WQ-musique | webqsp, squad, hotpotqa, 2wiki | musique, metaqa |
| WQ-hotpotqa | webqsp, squad, musique, 2wiki | hotpotqa, metaqa |

## 4. The rule

**The screen.** DEP's screen rule (docs/DEP_GNN_DEPTH.md section 4), on the twelve reads.

- Per read: floor = max(0.0075, 2 × the base's spread over its three seeds). d = zrckpb's seed 0 minus the base's seed 0.
- d at or below −floor is a LOSS. d at or above +floor is a GAIN.
- Any LOSS: NOT_ADOPTED. Otherwise any GAIN: ADOPT_FOR_FULL_RUN. Otherwise NO_GAIN.

**The full run.** ML1's section 4 (`zml1.verdict`), at three seeds against the base's three.

- Per read: floor = max(0.0075, 2 × the root mean square of both arms' spreads).
- A GAIN needs seed 0 at the floor and seeds 1 and 2 above zero. A seed-0 gain they do not confirm counts as WITHIN.
- A LOSS stands on seed 0 alone.
- ADOPT when there is no LOSS and at least one GAIN. Otherwise NOT_ADOPTED.

**INCOMPLETE.** Either verdict is INCOMPLETE when any of these holds:

- a read is missing;
- the base's R@5 at a seed differs between two compares;
- a fit's train or read did not run on DOB's device at 6 threads with the whole pick;
- R2's block is not R2's declared one: its declaration, constants, parameter list or count;
- R2's term ran on no batch, or LAM is not 1. A fit whose training had no batch with the term stops unstamped;
- a fit names a gate that is not PKO's present gate;
- the trainer's code differs from its base fit's in any file the base names: zml1s, zml1, zwq, zp1c3, zp1c3p, zp1c2 and
  zdob.

**Naming.** Under the user's ruling, zrckpb is an MLP arm, called "zrckp + R2's bridge head".

- A full-run ADOPT makes it the MLP's candidate on the CPU.
- It becomes the MLP of record only after its GPU fits (section 6).

## 5. Beside, deciding nothing

- The two C1R reads are marked in the table: WQ-hotpotqa musique and WQ-musique hotpotqa.
- R2's term per fit: the share of training questions with the term, and L_br at the first and last epoch.
- The train and read seconds per fit.

## 6. What follows

| verdict | next |
|---|---|
| screen ADOPT_FOR_FULL_RUN | the full run, already queued behind `r2p-gate` (section 8) |
| full run ADOPT | zrckpb on the host's GPU at three seeds on both splits, against ML1's zrckp, in its own declared file; then C1R again with it as the MLP |
| screen NO_GAIN or NOT_ADOPTED, or full run NOT_ADOPTED | zrckp stays, and R2's EM leaves the MLP's forms on PKO's pools. The MLP's next form is the one DOB's calls name (docs/DOB_DROP_ONE_BLOCK_CPU.md section 6) |
| INCOMPLETE | what is missing is supplied; nothing else changes |

A verdict here closes only this form, zrckpb on PKO's pools.

## 7. Risks, stated before the numbers

- **R2WS's losses.** On U2s's pools, zrckpb lost both multi-hop reads that C1R needs, at both seeds. The same may happen
  here.
- **PFS A2 (2310bb5).** Similarity to the first stage's top rows, as fixed columns, gave NOTHING on all five datasets on
  PKO's pools.
  - R2's head reads the same rows.
  - It differs in two ways: learned projections, and the EM term.
- **The screen reads seed 0 alone.** Only the full run decides adoption.
- **Time.** DOB's CPU fits train in about 45 minutes at 6 threads, and their reads take longer than 30 minutes. R2's
  head adds the bridge logits on top. The seconds are recorded beside.
- **Memory.** 18 GB on WQ-musique and 22 GB on WQ-hotpotqa: DOB's 16 and 20, plus 2 for the bridges' vectors.

## 8. How it runs

The items go into the feeder's items file. A fit's command is
`python outputs/host_ops/retry_cmd.py --tries 3 --wait 90 -- python outputs/mp_unified/zr2p.py fit --name <fit>`.

| item | CPU | GB | after | what |
|---|---:|---:|---|---|
| `r2p-selftest-cpu` | 1 | 4 | nothing; placed before DOB's base | `zr2p.py --selftest` |
| `r2p-zrckpb-s0` | 6 | 18 | `pkoh-gate`, `r2p-selftest-cpu`; placed after DOB's base fits, before DOB's drops | `zr2p.py fit --name scr-r2p-zrckpb-s0` |
| `r2p-zrckpb-s0-hp` | 6 | 22 | the same | `zr2p.py fit --name scr-r2p-zrckpb-s0-hp` |
| `r2p-compare` | 1 | 8 | the two fits and DOB's six base fits | `zr2p.py compare-all` |
| `r2p-calls` | 1 | 4 | `r2p-compare` | `zr2p.py calls`: `outputs/r2p/calls.json` and `.md` |
| `r2p-gate` | 1 | 1 | `r2p-calls` | `verdict_gate.py --json outputs/r2p/calls.json --key screen.verdict --want ADOPT_FOR_FULL_RUN` |
| `r2p-zrckpb-s1`, `-s2` | 6 | 18 | `pkoh-gate`, `r2p-gate` | the full run's fits on WQ-musique |
| `r2p-zrckpb-s1-hp`, `-s2-hp` | 6 | 22 | `pkoh-gate`, `r2p-gate` | the full run's fits on WQ-hotpotqa |
| `r2p-compare-full` | 1 | 8 | the four fits | `zr2p.py compare-all`: the screen's compares kept, the four new run |
| `r2p-calls-full` | 1 | 4 | `r2p-compare-full` | `zr2p.py calls`: the screen block and the full-run block |

**The gate.** On anything other than ADOPT_FOR_FULL_RUN, `r2p-gate` exits non-zero. After its one retry, the feeder drops
everything queued behind it.

**What is committed.**

- This file and `zr2p.py`.
- After each verdict, `outputs/r2p/calls.json` and `.md`.
- Per-fit records and the compares stay on the host and the laptop.

**Self-test.** `python outputs/mp_unified/zr2p.py --selftest` passed on the laptop before this commit. It covers:

- the names, pairs and nulls;
- that zml1s's `fit` reaches `train`, `read`, `device_env`, `where_ok` and `stamped` through its globals, so the patches
  reach them;
- zr2br as R2's declared form, with zrckpb in lean_screen's table;
- zp1c3p's train guard passing for zrckpb under the swap, and zrckp's again after;
- `same_modules` failing on a foreign lean_gpu;
- a fit with no term stopping;
- the patches on stand-ins, and their restoring, also on a failure;
- the pick guard;
- the compares read from records of the filed shape (a Windows-path base key, R@5 first), including the base's seed 0 at
  two values;
- both rules on made-up reads: NO_GAIN, ADOPT_FOR_FULL_RUN, NOT_ADOPTED on a LOSS and on a LOSS beside a GAIN, the
  0.0075 floor, INCOMPLETE, and the full run's ADOPT, UNCONFIRMED_GAIN, LOSS and INCOMPLETE.

The text encoder stays frozen. Test splits are never read. There is no LLM, and no new graph or pool.

## 9. Results (filed 2 November 2026)

The screen ran on the host's CPU (6 threads, torch 2.8.0+cpu): zrckpb at seed 0 on both splits, against DOB's CPU zrckp at
seeds 0, 1 and 2. `zr2p.py compare-all` and `calls` ran at 18:40 and 18:41 IST. No fit is missing and `problems` is empty.
Records: `outputs/r2p/calls.json`, `calls.md`.

**Screen: NOT_ADOPTED.** GAIN 2, LOSS 2, WITHIN 8.

| split | dataset | base s0, s1, s2 | zrckpb s0 | delta | floor | call |
|---|---|---|---:|---:|---:|---|
| WQ-musique | metaqa | 0.3406, 0.3670, 0.3860 | 0.4138 | +0.0732 | 0.0743 | WITHIN |
| WQ-musique | squad | 0.9127, 0.9122, 0.9138 | 0.9107 | -0.0020 | 0.0075 | WITHIN |
| WQ-musique | musique | 0.5152, 0.5086, 0.5134 | 0.5204 | +0.0052 | 0.0097 | WITHIN |
| WQ-musique | hotpotqa (C1R) | 0.8411, 0.8424, 0.8388 | 0.8389 | -0.0022 | 0.0075 | WITHIN |
| WQ-musique | 2wiki | 0.9070, 0.9079, 0.9049 | 0.9067 | -0.0003 | 0.0075 | WITHIN |
| WQ-musique | webqsp | 0.6522, 0.6583, 0.6566 | 0.6725 | +0.0203 | 0.0106 | GAIN |
| WQ-hotpotqa | metaqa | 0.3847, 0.4037, 0.3797 | 0.4038 | +0.0191 | 0.0278 | WITHIN |
| WQ-hotpotqa | squad | 0.9138, 0.9129, 0.9133 | 0.9106 | -0.0032 | 0.0075 | WITHIN |
| WQ-hotpotqa | musique (C1R) | 0.5749, 0.5718, 0.5752 | 0.5667 | -0.0082 | 0.0075 | LOSS |
| WQ-hotpotqa | hotpotqa | 0.8103, 0.8099, 0.8101 | 0.8020 | -0.0083 | 0.0075 | LOSS |
| WQ-hotpotqa | 2wiki | 0.8987, 0.9082, 0.9063 | 0.9054 | +0.0067 | 0.0172 | WITHIN |
| WQ-hotpotqa | webqsp | 0.6446, 0.6524, 0.6514 | 0.6610 | +0.0164 | 0.0146 | GAIN |

- Both GAINs are on webqsp (+0.0203, +0.0164), the knowledge-base set that trains under WQ.
- The LOSSes are musique in-domain under WQ-hotpotqa (-0.0082 against a floor of 0.0075), one of Claim 1's reads, and
  hotpotqa read zero-shot under WQ-hotpotqa (-0.0083).
- Beside, deciding nothing: R2's term reached 99.3% of the training questions on both splits. Its bridge loss fell from 5.68
  to 4.85 (WQ-musique) and from 6.15 to 5.07 (WQ-hotpotqa). Training took 3,210 s and 5,819 s; the reads 4,097 s and
  3,556 s.

**What follows (section 6).** zrckp stays the MLP of record, and R2's EM leaves the MLP's forms on PKO's pools. The full run
(`r2p-zrckpb-s1`, `-s2` and their `-hp` twins) waits on `r2p-gate`, which stops on this verdict, so those items drop. The
MLP's next form is the one DOB's calls name (docs/DOB_DROP_ONE_BLOCK_CPU.md section 6). LTM (docs/LTM_NEXT_HOP_MLP.md),
which runs on the same CPU base, is read beside it.
