# ML1: the free GNN (zg6r) and the typed MLP (zrckpt), fitted again on PKO's pools with their bases and three seeds

Declared 31 October 2026, before any fit of this stage. It is the model line of [PROGRAM_2026_10_30.md](PROGRAM_2026_10_30.md)
(section 12) and stage 9 there. It fills PKO's section 8 (items 1 and 2), G6RW's section 7 (row ADOPT_FOR_FULL_RUN) and U2n's
section 6. Code: one new module and one session script, written and self-tested after this file and before the first fit;
this file fixes the rule and the code follows it. Nothing here is a result, and nothing is adopted.

## 1. The question

The user's aim, 8 October 2026, and again on 31 October: improve the GNN to the state of the art, improve the MLP to be as
close to it as possible, and choose the inputs that help both on all six datasets and zero-shot. Both claims are measured
against that.

No model has been adopted by a full run since the lean pair (zrckp, zspk) on 14 October. U2h's full run and P1g-b's were
NOT_ADOPTED. Two forms have passed a screen and wait only for PKO's pools:

- **zg6r**, the GNN that is its own model (G6RW, filed e0e2c51): ADOPT_FOR_FULL_RUN, 5 GAIN and 0 LOSS on twelve reads
  against zspk, on today's pools, at one seed.
- **zrckpt**, the MLP with a pointwise head on 96 fixed columns (U2n): ADOPT_FOR_FULL_RUN under KG's splits with one GAIN,
  L-musique musique zero-shot +0.0098, clearing its floor of 0.0075 by 0.0023. It has never been read under WQ.

The question: fitted on PKO's pick O pools, with its base fitted there too, at three seeds, does each beat its base on the
reads where it can be called, and lose on none?

## 2. What runs

**Regime.** WQ (docs/WQ_WEBQSP_TRAINS.md, docs/WQK_WQ_ON_T4.md): two splits. WQ-musique trains on webqsp, squad,
hotpotqa and 2wiki fit carves and holds musique out; WQ-hotpotqa trains on webqsp, squad, musique and 2wiki and holds
hotpotqa out. metaqa is read zero-shot in both. Twelve reads per arm: two splits, six datasets, p@swa on the s1eval carves.

**Pools.** PKO's roots: pick O, built under PKO's amendments A1 to A5 at `--code c53fff2e6055`, fetched and sha-checked.
Every fit here reads PKO's roots and no other pools. The KT input, typed links and tau0 are the ones PKO's records name.

**Arms and fits.** Four arms, three seeds (0, 1, 2), two splits: 24 fits. Each seed-k fit differs from the arm's seed-0 fit
of its split in the seed alone, as WQG's did.

| arm | role | trained by | read cap |
|---|---|---|---|
| zspk | the GNN base (the lean GNN) | zp1c3's train, through zwq.on_wq | as WQK's |
| zrckp | the MLP base (the lean MLP) | zp1c3p's train, through zwq.on_wq | as WQK's |
| zg6r | the GNN under test | zg6rw.train | 32,768 |
| zrckpt | the MLP under test | U2n's trainer (zu2n), moved onto WQ's carves by the new stage module, as zwqk moves zwq's | as U2n's |

- zg6r's encoder starts from the same split's seed-0 zspk fit on PKO's roots, sha-pinned, as G6RW section 4 fixes and as
  WQG's seed-1 and seed-2 fits did. zspk therefore fits first.
- zg6 is not fitted again. WQG's zg6r-against-zg6 beside, on today's pools, stands as filed.
- Names: `scr-ml1-<arm>-s<k>` for WQ-musique and `scr-ml1-<arm>-s<k>-hp` for WQ-hotpotqa.

## 3. The reads and the floors

**The read.** Per arm, twelve. For a seed k, delta_k = R@5 of the arm minus R@5 of its base at the same seed, same split and
dataset, both on PKO's roots, on the same device.

**The floor** is WQG's pair floor, on this build: max(0.0075, 2 x the root mean square of the two arms' spreads). An arm's
spread on a read is the root mean square of its seed-1 and seed-2 differences from its own seed-0 fit. The floors of
today's pools are not reused; PKO's roots are a different build.

## 4. The rule (fixed now)

Per arm, over its twelve reads:

- **GAIN.** delta_0 reaches the floor (screen_recall.recall's call) **and** delta_1 and delta_2 are both above zero. A seed-0
  call that the other seeds do not confirm is reported as UNCONFIRMED_GAIN and counts as WITHIN.
- **LOSS.** delta_0 is at or below minus the floor. A LOSS stands on seed 0 alone, whether or not the other seeds agree;
  it is reported with their signs.
- **ADOPT** (the arm replaces its base): no LOSS on any of its twelve reads and at least one GAIN.
- **NOT_ADOPTED:** otherwise.
- **INCOMPLETE:** a fit is missing or has not exited 0, a fit is not the declared one (the checks of WQG's section 3: card,
  code sha256, records of the inputs), a compare ran on two devices, or an arm has fewer than twelve reads.

The two arms are decided separately: zg6r against zspk, and zrckpt against zrckp. Why GAINs must be confirmed and LOSSes
need not be: WQG found that seed 0 can sit in the tail of the spread (zg6r on WQ-musique musique reads 0.4106 at seed 0 and
0.4608 and 0.4618 at seeds 1 and 2), and the 20 October audit found that single-seed GAINs of this size were mostly seed luck.
An adoption should be hard to reach and easy to block.

## 5. Reported beside, deciding nothing

- Delta R@5 for every seed, FC@5 and hit@1, per read.
- The count of GAIN, LOSS, WITHIN and UNCONFIRMED per arm, split into the four zero-shot reads and the eight in-domain ones.
- The MLP's share of the GNN's gain over rrf, (zrckpt - rrf) / (zg6r - rrf), where the denominator exceeds 0.002, for every
  pairing of the lean arms and the arms under test. This is the input to Claim 1's re-measure, not the measure.
- Seconds and peak memory per fit, against G6RW's and WQK's. Cost is Kaggle's T4 time; there is no Modal spend.
- zg6r against zspk with the floors of today's pools (WQG's), to show how much of any call comes from the new build.

## 6. What follows (fixed now)

| outcome | what follows, each its own declared file |
|---|---|
| zg6r ADOPT | The GNN line's incumbent becomes zg6r. Claim 1 re-measured with P3c's intervals against the strongest GNN; Claim 3 re-timed cold with zg6r in the table; then the isolation arms (G6RW section 7, third row: relation messages alone, encoder start alone) to say where the gain comes from, and the depth form (docs/MUSIQUE_DIAGNOSIS.md, step 3: more layers, or read-time rounds that go back to the neighbours of the top candidates), graded on the 3-hop and 4-hop slices. |
| zg6r NOT_ADOPTED with a GAIN and a LOSS | G6RW section 7, second row, under WQ. |
| zg6r NOT_ADOPTED with no GAIN | The isolation arms (G6RW section 7, third row). |
| zrckpt ADOPT | The MLP line's incumbent becomes zrckpt. Claim 1 and Claim 3 as above with it. |
| zrckpt NOT_ADOPTED | zrckp stays. Its next form is U2n's section 6, under WQ. |
| INCOMPLETE | Supply what is missing; nothing else changes. |

If both are NOT_ADOPTED, the lean pair stands, and the model line's next stage is the depth form (the read-time rounds)
alone. This stage never closes a family: a verdict closes only the form it tested.

## 7. How it runs

- **Start.** When PKO's eleven carves are folded, fetched and their shas match the ledger; no date is promised. Nothing
  here runs on today's pools, by the 13 October rule.
- **Where.** Kaggle's T4, both kernels, in parallel (mpr-kg and mpr-kg-g6b2), under the account's two-session limit. The
  dataset package (mpr-pkg-*) carries the arrays. Only I launch sessions.
- **Order.** zspk, then zg6r (needs zspk's seed-0 encoder); zrckp, then zrckpt. Seed 0 of every arm before seeds 1 and 2.
- **Time.** About 13 T4-hours by the filed timings: zg6r's six fits at G6RW's 2,942 s and 4,947 s a split, three seeds
  each, about 6.6 hours; the eighteen lean fits at about the 6.6 hours that KG's fourteen took. The first fit re-times it
  on PKO's roots, whose rows per question differ from today's, and the estimate is replaced by the measurement.
- **Self-tests** before the first fit: the rule on hand-built reads (a confirmed GAIN, an unconfirmed one, a LOSS with and
  without seed agreement, the INCOMPLETE cases), on the real shape of WQG's records, not a toy (the lesson of WQG's guard).
- **The rule's code** is `outputs/mp_unified/zml1.py` (31 October, after this file). Its self-test passes: the hand-built
  cases, and the spreads and floors of zg6r, zspk, zrckp and zg6 reproduced from the filed seed-0 R@5 and seed differences of
  WQG's and WQK's floors records. It trains nothing; the fits and the reading of PKO's records are written when PKO's
  arrays exist and their shapes can be checked.
- **A dry run, disclosed.** `zml1.py dryrun-today` applies the rule to the seeds already filed on today's pools, zg6r
  against zspk, at the pair floor of the two arms' filed spreads. It is not ML1's result: other pools, no refit, bases not
  fitted again. It called 5 GAIN, 0 LOSS and 7 WITHIN, every GAIN confirmed by seeds 1 and 2 (WQ-musique hotpotqa and
  webqsp; WQ-hotpotqa musique, 2wiki and webqsp), which on those pools is an ADOPT. Four of the five clear their floor by
  0.0001 to 0.0071. It was run to test the code. It changes nothing in sections 1 to 6, and ML1's verdict reads only PKO's
  pools.

## 8. Risks, stated before any number

- **The build changes.** PKO's pools are the same rule as the pools G6RW read, built by a different program. The rows per
  question may differ, and the numbers will not equal G6RW's.
- **Two reads cannot call.** On today's pools zg6r's floors are 0.1014 (WQ-musique musique) and 0.0866 (WQ-hotpotqa metaqa).
  Whatever PKO's build gives, those two will most likely read WITHIN. The rule does not need them.
- **Where zg6r's gain came from.** G6RW's own text says the three passage-graph GAINs may come from message passing and
  the encoder start, not the relation messages. This stage cannot separate them. An ADOPT here does not say the relation
  messages matter.
- **zrckpt's evidence is thin.** One GAIN, 0.0023 over its floor, under KG's splits, never read under WQ. A flat outcome
  is the likeliest, and it would be a clean negative.
- **Memory.** zg6 peaked at 9.13 GB on WQ-hotpotqa under WQK, and the T4 has 15 GB. If a fit runs out of memory the session
  records the exit code, the verdict is INCOMPLETE, and a smaller batch is an amendment declared before its rerun.
- **Three seeds are few.** A floor from two differences is itself noisy. More seeds, for the reads that sit near their
  floors, would be a separate declared file.

## 9. Not claimed

No adoption, and no claim 1, 2 or 3 number. Nothing about latency, about the SEMB z-score ruling (still the user's), or about
any pool other than PKO's pick O. A NOT_ADOPTED here closes the two forms as tested, on one build, at three seeds.

## 10. Amendment A1 (31 October 2026, before any ML1 fit): the placement is the host's GPU

Sections 1 to 6, 8 and 9 stand as written. This amendment replaces the "Where" and "Time" lines of section 7 and nothing else.

**Why.** PKO is built on the lab host (docs/PKOH_PKO_ON_THE_HOST.md), so its arrays, about 88 GiB, are on the host when they are
done. Pulling them to the laptop takes about 35 hours at the link's rate and the upload to Kaggle comes after that. The host's
GPU is the placement WQ used for all twelve fits of the base set (docs/WQ_WEBQSP_TRAINS.md section 5): the same card and the same
trainers. Its task-level qualification (docs/GPU_TASK_QUALIFICATION.md, 29 September 2026): evaluation
TASK_EQUIVALENT, training TRAINING_REPRODUCIBLE.

**The placement, quoted from configs/gpu_task_qualification.yaml (candidate host_gpu_det).** DESKTOP-SLQMEQH, rx env
mpr-cu128@62fc45e9e1ba (torch 2.8.0+cu128, CUDA 12.8), NVIDIA RTX 4500 Ada Generation, driver 596.71; float32 only;
torch.backends.cuda.matmul.allow_tf32 = False, torch.backends.cudnn.allow_tf32 = False, float32 matmul precision "highest",
cudnn.benchmark = False; torch.use_deterministic_algorithms(True, warn_only=True) with CUBLAS_WORKSPACE_CONFIG=:4096:8 set before
CUDA starts; every determinism warning recorded.

**What this changes.**

- **Where.** All 24 fits, their reads and their compares run on this one placement, as feeder items on the host. Kaggle's T4 is
  not used by this stage. Every fit of every arm, the bases zspk and zrckp included, is fitted on it, so no number crosses
  placements (the host-native protocol's same_placement_for_every_comparator).
- **Inputs.** PKO's arrays are built on the host, by the code at `--code c53fff2e6055`, whose check (c) rebuilt webqsp's s1eval
  shard 0/2 there and found every field equal to the laptop's filed unit, npz and KT sha256 included (31 October, no
  tolerance). PKO's gate must be PASS before any ML1 item reads them. Every ML1 record names its git commit.
- **One repeat.** `scr-ml1-zrckp-s0-rep`: zrckp, seed 0, WQ-musique, fitted a second time in a fresh process. Whether it is
  bit-identical to `scr-ml1-zrckp-s0` (the final state_dict sha256 and the reads) is filed beside the verdict. It decides nothing:
  the protocol asks for the repeat to be filed, and ADOPT / NOT_ADOPTED read the 24 fits only.
- **Read caps** stay as section 2 fixes them (zg6r 32,768, zrckpt 65,536 rows a batch, whole questions). The card is larger than
  the T4; the caps are not raised, so the batches, and the scores, are the declared ones.
- **Time.** The 13 T4-hours of section 7 do not carry over. The first fit re-times the stage on the host and the estimate is
  replaced by that measurement. Fits share the card as the feeder's GPU shares allow.
- **Section 5's last row** (zg6r against zspk with the floors of today's pools, WQG's) compares calls, not scores. WQG's floors
  were made on the T4; the row names both placements and is read as a comparison of calls only.
- **If the host is lost** the stage stops with INCOMPLETE. It does not move to another device in the middle: the T4 would be a
  new amendment with all 24 fits fitted again there.

**Start.** Unchanged: when PKO's eleven carves are folded, their shas match the ledger and the gate is PASS. The stage module
(fit, read, compare, floors, verdict through zml1.py) is written once PKO's records exist and their shapes can be checked, as
section 7 says, and is self-tested on real record shapes before the first fit. Nothing in this amendment is a result, and
nothing is adopted.
