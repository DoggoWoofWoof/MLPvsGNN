# P1g-c: P1g-b's pools under the adopted models

Declared 18 October 2026, after P1g-b's verdict and before any number of this stage. Code:
`outputs/mp_unified/zp1gc.py`. Numbers here are development numbers. Status: filed 27 October (section 8):
pools and input both NOT_ADOPTED; the full run takes P1c2's pools, with zrckpr and zspx.

## 1. Why this, now

**P1g-b was NOT_ADOPTED** (docs/P1GB_POOLS_REFIT.md, section 7). It refit the lean pair, unchanged, on P1g2's chosen
pools: ALL ≥ 0.95 on every dataset, about 3.4 times today's rows. After the seed-null re-call:
- **the MLP (zrckp):** 1 GAIN (L-musique webqsp, +0.0141), no LOSS;
- **the GNN (zspk):** no GAIN, 1 LOSS (L-hotpotqa musique, −0.0092). On L-musique, musique held out rose +0.0157 and
  webqsp +0.0125, both WITHIN.

Both models read the same pools, so P1g-b's joint rule did not adopt them, and its full run's gate shut.

**That verdict closes only its form** (docs/SCREENS.md, 3b): the lean pair as it stood on 17 October, refit on the new
pools. The pools had signal (the MLP's GAIN). The user's target is coverage first (15 October: "ALL ≥ 0.95 on every
dataset, coverage first, then the fewest rows"), and only P1g2's pools meet it.

**Two model changes have passed their screens since,** both on P1c2's pools:
- **G5's zspx** (docs/G5_COMBINED_GNN.md): zsp's step end to end, G3's residual message and G4's second step, three
  steps in all;
- **R1's zrckpr** (docs/R1_RESIDUAL_INPUT.md): R1's nine residual columns through a pointwise head on the MLP.

Each one's "what follows" sends its arm to P1g-b's pools next, in its own file. They are also the two changes most
likely to rank the pools' extra golds:
- zspx's three steps reach further from the first stage's rows;
- R1's columns tell each row how far its score sits from its linked neighbours'.

So P1g-c puts P1g-b's pools under the adopted models. It asks two things:
1. **Pools:** do the adopted models rank better on P1g-b's pools than on P1c2's? This decides the pools.
2. **Input:** do R1's columns help the GNN too? R1's verdict left this to the next file (F1's one-input-set rule: an
   input the lean pair shares is one both models read).

## 2. The arms

| arm | the model | the carve | as filed in |
| --- | --- | --- | --- |
| zspx | zg5mix.ZSPX | G3's (zspk's KT link carve, with c and s beside) | G5 |
| zrckpr | zr1res.ZRCKPR | R1's MLP carve (zrckp's KT chain carve, with the nine columns beside) | R1 |
| zspxr | zspx, with R1's head added to the score before zspx's first step | G3's, with R1's columns beside as well | here |

**zspxr** puts R1's head where zspkr put it, before the first step reads the score. Its class order is ZSPX, ZSPR,
ZSPK, ZProp, R1Mix, KTMix, ZRM, so ZSPX's pre-step score is R1Mix's: the KT head's score plus R1's head. R1's head is
R1's own: 9 → 16 → 1, 177 parameters, drawn from its own generator (seed + 5501), last layer at zero. So zspxr starts
as zspx bit for bit, and no draw of zspx's moves.

Every setting, seed, batch, loss and loop is the arm's own. The graph, the encoder, the carves and the relation tables
are today's. The MLP reads no message: R1's columns are fixed before any row is scored, and its head reads only the
row's own nine columns.

## 3. The builds, and the gate on them

**What has to be built again.** P1g-b's caches are on the host (53 GB). The host's clean-up removed the rest of
P1g-b's builds at 23:19 on 17 October, after its verdict, when no queued item named them:

| root | GB |
| --- | ---: |
| looks | 54.13 |
| links | 17.39 |
| chains (rmatch's and zrc's) | 11.14 |
| KT columns | 2.69 |

They are built again by **P1g-b's own items, unchanged**: the same commands and threads, under new names (`p1gc-*`).
That is 177 look shards, then 11 link builds and 6 chain builds from them. The 11 KT builds read the caches and pools
and start at once.

**The snapshot (`zp1gc.py snap`), before any rebuild:**
- it copies every link, chain and KT record to `outputs/p1gc/p1gb_records`;
- it first checks them against the record sha256s that P1g-b's four fits stamped: zlink's, rmatch's and zrc's in
  full, and the KT records of each fit's training carves;
- a record that is not the one a fit read stops it.

**The gate (`zp1gc.py gate`), after the rebuild.** A record holds its run's seconds and time, so a rebuilt record's
file differs even when its arrays do not. The gate therefore compares what the fits read:
- every snapshot record is on disk again with the same `arrays` (each array's sha256, dtype and shape) and the same
  question and row counts;
- no record is there that the snapshot lacks;
- otherwise it is SHUT, and its record names each difference.

**Then the stage's own builds,** behind the gate:
- **G3's residual arrays:** `zg3res.build`, unchanged, on each of P1g-b's 140 step-1 parts, into `outputs/p1gc/res`;
- **R1's nine columns:** `zr1res.build`, unchanged, from them, into `outputs/p1gc/cols`.

Both read P1g-b's caches, looks and links as their roots (`zp1gb.on_p1gb`).

**Every build, fit and read checks the gate first.** Each refuses while the gate is shut, and refuses a P1g-b record
rebuilt after the gate opened.

## 4. The fits and the calls

**Fits:** one card fit per arm on L-musique and L-hotpotqa, six in all: `scr-p1gc-zspx`, `scr-p1gc-zrckpr`,
`scr-p1gc-zspxr`, each with its `-hp` fit. Card settings are each arm's own: variant p, config 2e-3:1e-4:0.1:8:2,
seed 0, hidden 128. Each fit is read on the six s1eval carves of P1g-b's pools at p@swa.

**The pools call** decides the pools for the adopted models. Each arm is compared with its own card fit of the split
on P1c2's pools:

| arm | on P1g-b's pools | against, on P1c2's pools |
| --- | --- | --- |
| zspx | scr-p1gc-zspx, -hp | G5's scr-g5-zspx, -hp |
| zrckpr | scr-p1gc-zrckpr, -hp | R1's scr-r1-zrckpr, -hp |

The reads use the same questions and gold totals. R@5 is over the question's golds, so a gold a pool misses counts as
a miss. The re-call's base screens are each P1c2-pools fit compared with step 1's fit of the split
(`outputs/p1gc/base-pools-<arm>-<split>.json`).

**The input call** decides R1's columns on the GNN. zspxr (scr-p1gc-zspxr, -hp) is compared with P1g-c's own zspx fit
of the same split, on the same pools. The base screens are P1g-c's zspx fits against step 1's
(`outputs/p1gc/base-input-zspx-<split>.json`).

**Each call** is relz's pair, then its re-call under the seed null (the four null files of docs/SCREENS.md). GAIN,
LOSS and WITHIN are docs/SCREENS.md's, on R@5.

### The rules, fixed before any number

**Pools (P1g-b's rule, with the adopted arms):**
- **ADOPT_FOR_FULL_RUN:** neither zspx's nor zrckpr's pools re-call has a LOSS on any of its twelve reads, and the two
  together have at least one GAIN. Both models read the same pools, so pools that cost either arm are not adopted;
- **NOT_ADOPTED:** otherwise;
- **INCOMPLETE:** a re-call is missing, is not P1g-c's pools re-call of that arm, or does not hold twelve reads.

**Input (per arm):**
- **ADOPT_FOR_FULL_RUN:** zspxr's re-call has no LOSS on any of its twelve reads and at least one GAIN;
- **NOT_ADOPTED:** otherwise;
- **INCOMPLETE:** as above.

**Reported beside, deciding nothing:**
- webqsp and metaqa on both splits (webqsp never trains);
- musique in domain (L-hotpotqa), Claim 1's read;
- each arm against P1g-b's own fit of the lean arm on the same pools: zspx and zspxr against scr-p1gb-zspk, zrckpr
  against scr-p1gb-zrckp (`outputs/screen/scr-p1gc-lean-<arm>`). This is the model change's gain on P1g-b's pools;
- FC@5 and hit@1 per read, as the compares give them.

## 5. What follows (fixed now)

| pools | input | the full run (its own file, declared before its numbers) |
| --- | --- | --- |
| ADOPT_FOR_FULL_RUN | ADOPT_FOR_FULL_RUN | P1g-b's pools; MLP zrckpr, GNN zspxr (R1's columns in both models) |
| ADOPT_FOR_FULL_RUN | NOT_ADOPTED | P1g-b's pools; MLP zrckpr, GNN zspx |
| NOT_ADOPTED | ADOPT_FOR_FULL_RUN | P1c2's pools; MLP zrckpr, GNN zspx; zspxr's two screen fits on P1c2's pools come first in that file, decided against G5's zspx fits by the input rule |
| NOT_ADOPTED | NOT_ADOPTED | P1c2's pools; MLP zrckpr, GNN zspx |

Each full run is six splits per arm under grade-nullx, as P1c3's and P1c3p's were.

**Next forms, named now:**
- **Pools NOT_ADOPTED:** P1g-b3. Each row's place in P1g-b's pool becomes an input to both models through R1's
  pointwise head: today's fill, the fill beyond it, or the extra set. Both models can then learn how far to trust the
  rows the bigger pools add. It is decided as P1g-b was, in its own file. That is the last form owed before the
  bigger pools close: the pools idea will then have reached both models as an input.
- **Input, a GAIN with a LOSS:** zspxrb, R1's head bounded, 0.2 tanh(h / 0.2), on zspx (one declared file).
- **Input, no GAIN:** zspxs. R1's nine columns are read by zspx's first step head beside zprop's inputs, so the step
  weighs a row's neighbours by its residual (one declared file).
- **INCOMPLETE:** the missing re-call is run; nothing else changes.

A verdict closes only its form (docs/SCREENS.md, 3b).

## 6. Cost

**The rebuild, from P1g-b's measured items:**

| items | count | time (sum of runtimes) | peak memory each |
| --- | ---: | ---: | ---: |
| looks | 177 | 346,830 s | at most 9.8 GB |
| links | 11 | 2,626 s | at most 1.7 GB |
| chains | 6 | 2,531 s | at most 2.3 GB |
| KT columns | 11 | 8,733 s | at most 6.2 GB |

The looks dominate. At the feeder's 140 GB cap, beside the card's fits, about 11 run at once, so the rebuild takes
about 6 to 9 hours. G3's builds then take about 4 minutes a part and R1's about 11 seconds. P1g-b's parts hold the
same bytes per part as P1c2's (0.38 against 0.39 GB), so G3's and R1's measured sizes carry over: 4 and 3 GB.

**Disk:**
- the rebuild writes back the 85.35 GB that the clean-up removed;
- G3's arrays add about 7.4 GB (2 bytes per edge; P1g-b's links are 17.39 GB at about 5 bytes per edge);
- R1's columns add about 2.7 GB (18 bytes per row).

That is about 95 GB of the 120.8 GB free at the time of declaring. The looks are read by the link and chain builds
and by G3's builds, so they stay until those are done; they stay pinned while P1g-c's items are queued.

The rebuild alone leaves about 35 GB free, so the feeder's 30 GB hold (only items that write little go in below it)
can pause the last G3 and R1 builds. Nothing on the host is removable by the auto clean-up while the queued stages
read it (the 00:49 catalogue: 203.8 of 208.4 GB named by or pinned for queued or running items). So:
- G3's and R1's builds go in with the training carves first (metaqa, squad, 2wiki and hotpotqa fit, musique s1fit),
  each part's columns right after its residual, so the six fits can start even if the eval parts wait;
- if the hold pauses them, I list for the user the folders no queued item reads any more (a list delete is the
  user's; host_cleanup's record keeps what went and how to rebuild it), and the eval parts go in once it is freed.

**The card:** P1g-b's fits took 972 to 1,739 s at 10.8 to 15.5 GB peak, one at a time with the full card (share
0.94). zspx took 1.25 and 1.32 times zspk's time at the same share. So each P1g-c fit should take about 1,200 to
2,300 s, and its read about 7 minutes: about 3.5 hours for the six fits and their reads.

## 7. How it runs

**Self-test, passed before this commit.** `python outputs/mp_unified/zp1gc.py --selftest` (9 s) checks:
- **The arms.** zspx and zrckpr are as filed. zspxr has the class order above, on G3's carve with R1's columns
  beside. zspk's, zrckp's and zspkr's arms are unchanged.
- **The roots.** P1g-b's roots are in place inside, with P1g-c's residual and column roots, and everything is
  restored after. zspkr's column root is untouched.
- **The snapshot** on toy records:
  - it is checked against the fits' stamps and copied byte for byte;
  - a stamp that differs on a link record or a KT record stops it, and so does a fit without a stamp.
- **The gate:**
  - SHUT before the rebuild, and OPEN after it when only seconds and times moved;
  - SHUT on a changed array, a changed count, an extra record or a missing snapshot;
  - gate_open refuses another stage's gate and a record rebuilt after the gate opened.
- **The builds.** They are G3's and R1's on P1g-b's caches, looks and links, into P1g-c's roots, and they refuse
  behind a shut or missing gate.
- **The refusals:**
  - train refuses zspk's and zspkr's arms, another split, the wrong fit name, P1c2's caches, the CPU and a shut gate;
  - read refuses G5's, R1's and P1g-b's fits, and a shut gate.
- **The calls.** relz's pair and re-call run for each call and arm against its own base fits, stamped P1g-c's with
  the call, and relz is restored after. The calls refuse an arm they do not hold.
- **The verdict.** It covers the pools and input outcomes, the four full runs of section 5, the next forms, three
  INCOMPLETE cases, and the lean-pair deltas beside.
- **The records.** Each arm needs its own G3 and R1 records.
- **The model on P1c3's toy carve:**
  - the batch is G3's plus the columns;
  - zspxr is zspx bit for bit at the start (eval, and training with the same dropout draws), with R1's head as R1
    draws it;
  - with every head moved, zspxr is zspx's three steps over (the KT score + R1's head), bit for bit; without the
    columns it is zspx, and without links it is the KT score + R1's head;
  - toy fits through lean_gpu's loop repeat bit for bit and R1's head learns. Without the columns, zspxr's fit is
    zspx's on every shared key, and R1's last layer stays at zero.

**Host items** (`p1gc-*`: the twelve GPU items right after P1g-b's own GPU items, ahead of A5 to A13's screens as
P1g-b's were; the rest at the end of the file):
- `p1gc-copy-records` (the snapshot);
- P1g-b's 177 look shards, 11 link builds, 6 chain builds and 11 KT builds, under P1g-c's names, each sized by its
  P1g-b twin's measured peak;
- `p1gc-gate`;
- 140 `p1gc-res-*` builds (through pylib_run, 1 CPU and 4 GB each), and 140 `p1gc-cols-*` builds (1 CPU, 3 GB);
- six trains and six reads at `--frac 0.92`, share 0.94, one at a time as P1g-b's were, sized from P1g-b's measured
  trains and reads;
- on the CPU: four base compares for the pools call (any time), two for the input call (after P1g-c's zspx reads),
  six compares for the calls and six against P1g-b's lean-pair fits beside, three pairs, three re-calls and the
  verdict: 524 items in all.

## 8. Results

Filed 27 October 2026 from the host's records (`outputs/p1gc/screen-verdict.json`, written 2026-10-20T01:47:49Z; the
re-calls `outputs/screen/scr-p1gc-pools-zspx-pair-recall.json`, `scr-p1gc-pools-zrckpr-pair-recall.json` and
`scr-p1gc-input-zspxr-pair-recall.json`, host paths made relative). **Pools NOT_ADOPTED and input NOT_ADOPTED, so the
full run takes P1c2's pools, with zrckpr as the MLP and zspx as the GNN** (section 5's last row; the verdict's
`full_run`: pools P1c2's, MLP zrckpr, GNN zspx, zspxr not carried). P1g-b's pools gain one read on the GNN and cost both
models musique in domain on L-hotpotqa; R1's columns before zspx's first step lose webqsp zero-shot on both splits.

**The gate opened** at 2026-10-19T20:51:42Z with no problems (`outputs/p1gc/gate.json` and `.md`): every snapshot record
was on disk again with the same arrays and question and row counts, 140 link records, 41 of rmatch's, 41 of zrc's and 11
KT records, and no record the snapshot lacks. The rebuilt looks they were built from had been checked first
(docs/P1GC_LOOK_ALIAS.md, section 7).

- **Pools, zspx on P1g-b's pools against its card fit on P1c2's (G5's `scr-g5-zspx`): 1 GAIN, 1 LOSS.** The GAIN is
  webqsp zero-shot on L-musique: +0.0309 [+0.0170, +0.0444] against its 0.0102 floor, with FC@5 +0.0299. The LOSS is
  musique in domain on L-hotpotqa: -0.0087 [-0.0141, -0.0034] against its 0.0075 floor, with FC@5 -0.0170. Two filed
  GAINs are WITHIN under the re-call: musique zero-shot on L-musique, +0.0133 [+0.0067, +0.0197] against its 0.0720
  floor, and webqsp zero-shot on L-hotpotqa, +0.0158 [+0.0025, +0.0293] against its 0.0233 floor. Six more intervals
  exclude 0, all inside their floors: on L-musique, metaqa -0.0038 [-0.0055, -0.0021], squad +0.0039 [+0.0019, +0.0060],
  hotpotqa +0.0031 [+0.0006, +0.0057] and 2wiki -0.0062 [-0.0078, -0.0046]; on L-hotpotqa, metaqa -0.0043 [-0.0060,
  -0.0027] and 2wiki +0.0026 [+0.0008, +0.0044].
- **Pools, zrckpr on P1g-b's pools against its card fit on P1c2's (R1's `scr-r1-zrckpr`): 0 GAIN, 1 LOSS.** The LOSS is
  musique in domain on L-hotpotqa: -0.0090 [-0.0148, -0.0036] against its 0.0075 floor, with FC@5 -0.0087. musique
  zero-shot on L-musique falls -0.0144 [-0.0199, -0.0085], a LOSS by its pair's interval that is WITHIN under its 0.0720
  floor. Three more intervals exclude 0, all on L-hotpotqa and inside their floors: metaqa -0.0029 [-0.0045, -0.0013],
  hotpotqa zero-shot -0.0039 [-0.0072, -0.0007] and 2wiki -0.0028 [-0.0046, -0.0011].
- **Input, zspxr against P1g-c's own zspx fit on P1g-b's pools: 0 GAIN, 2 LOSS.** Both LOSSes are webqsp zero-shot:
  -0.0175 [-0.0272, -0.0085] on L-musique against its 0.0102 floor (FC@5 -0.0166), and -0.0357 [-0.0453, -0.0259] on
  L-hotpotqa against its 0.0233 floor (FC@5 -0.0279). musique zero-shot on L-musique falls -0.0079 [-0.0134, -0.0021], a
  LOSS by its pair's interval that is WITHIN under its 0.0720 floor. Four more intervals exclude 0, all inside their
  floors: on L-musique, metaqa +0.0035 [+0.0025, +0.0045], hotpotqa -0.0031 [-0.0055, -0.0009] and 2wiki +0.0018
  [+0.0002, +0.0034]; on L-hotpotqa, hotpotqa zero-shot -0.0055 [-0.0084, -0.0028]. Two more on L-hotpotqa, metaqa
  +0.0010 [+0.0000, +0.0020] and squad +0.0017 [+0.0000, +0.0033], have a lower end stored as 0.0, which does not say
  whether it is above 0.
- **Beside: the adopted changes still gain on P1g-b's pools; the pools cost musique in domain.** Against P1g-b's own
  fits of the lean arms on the same pools (pair calls at the 0.0075 floor, not re-called), zspx gains webqsp zero-shot
  on L-musique (+0.0159) and, on L-hotpotqa, musique in domain (+0.0152), hotpotqa zero-shot (+0.0091) and webqsp
  zero-shot (+0.0264); zrckpr gains webqsp zero-shot on L-musique (+0.0105); zspxr gains musique in domain on L-hotpotqa
  (+0.0177) and loses webqsp zero-shot there (-0.0093). musique in domain on L-hotpotqa, Claim 1's read, is the pools
  call's one LOSS on each model (-0.0087 and -0.0090).

Not adopted, on either call. This closes only these forms: P1g-b's pools under the adopted models, and R1's columns
before zspx's first step (docs/SCREENS.md section 3b). The next forms, fixed in section 5 and recorded in the verdict:
**for the pools,** the pools stay P1c2's for the adopted models; the full run of zrckpr and zspx on P1c2's pools (its
own file), six splits per arm under grade-nullx; and the pools' next form P1g-b3, each row's place in P1g-b's pool
(today's fill, the fill beyond it, the extra set) as an input to both models through R1's pointwise head, decided as
P1g-b was (its own file), the last form owed before the bigger pools close. **For the input, with no GAIN:** zspxs: R1's
nine columns read by zspx's first step head beside zprop's inputs, so the step weighs a row's neighbours by its residual
(one declared file).

The refits on P1g-b's pools that waited on this verdict, A5cp, A3vp and A7p, are therefore beside, deciding nothing
(each file's section 5, "The pools"): on P1c2's pools their stages' own verdicts decide, A5c's and A7's for both arms
and A3v's for zrckpy, all ADOPT_FOR_FULL_RUN.

The floors are the seed null of 7 October (docs/SCREENS.md section 2: `outputs/screen/scr-null-s1.json`,
`scr-null-s2.json` and their `-hp` files), not N2's fresh floors (docs/N2_FRESH_SEED_NULL.md).

Records, filed with this section: the gate `outputs/p1gc/gate.json` and the verdict `outputs/p1gc/screen-verdict.json`,
each with its `.md`; for each call (`pools-zspx`, `pools-zrckpr` and `input-zspxr`), the compares
`outputs/screen/scr-p1gc-<call>.json` and `-hp.json`, the pair `-pair.json` and its re-call `-pair-recall.json`, each
with its `.md`; the compares beside against P1g-b's lean fits, `outputs/screen/scr-p1gc-lean-<arm>.json` and `-hp.json`
for zspx, zrckpr and zspxr, each with its `.md`; the re-calls' base screens `outputs/p1gc/base-pools-<arm>-<split>.json`
(zspx and zrckpr) and `outputs/p1gc/base-input-zspx-<split>.json`, each with its `.md`; and the eleven look-alias
records `outputs/p1gc/look_alias/<dataset>/<carve>.json` (docs/P1GC_LOOK_ALIAS.md, section 7). The host paths were made
relative in each re-call's two `rel` entries and nowhere else, so the `recall_sha256` each verdict holds is the host
file's, from before that change. The snapshot (`outputs/p1gc/p1gb_records`), the residual arrays (`outputs/p1gc/res`),
R1's columns (`outputs/p1gc/cols`) and the looks' per-part records stay on the host.

The twelve reads of the pools call after the seed-null re-call (delta = the arm's R@5 on P1g-b's pools minus its card
fit's on P1c2's, on the same questions and gold totals; a call is shown with its filed call where the re-call changed
it):

| split | dataset | read | zspx on P1c2's pools R@5 | zspx on P1g-b's pools: delta [95% CI], call | zrckpr on P1c2's pools R@5 | zrckpr on P1g-b's pools: delta [95% CI], call |
| --- | --- | --- | --- | --- | --- | --- |
| L-musique | metaqa | in-domain | 0.8030 | -0.0038 [-0.0055, -0.0021], WITHIN | 0.8012 | -0.0008 [-0.0024, +0.0008], WITHIN |
| L-musique | squad | in-domain | 0.9092 | +0.0039 [+0.0019, +0.0060], WITHIN | 0.9127 | +0.0008 [-0.0010, +0.0026], WITHIN |
| L-musique | musique | zero-shot | 0.5073 | +0.0133 [+0.0067, +0.0197], WITHIN (filed GAIN) | 0.5197 | -0.0144 [-0.0199, -0.0085], WITHIN (filed LOSS) |
| L-musique | hotpotqa | in-domain | 0.9074 | +0.0031 [+0.0006, +0.0057], WITHIN | 0.9083 | -0.0024 [-0.0049, +0.0001], WITHIN |
| L-musique | 2wiki | in-domain | 0.9351 | -0.0062 [-0.0078, -0.0046], WITHIN | 0.9361 | +0.0004 [-0.0012, +0.0019], WITHIN |
| L-musique | webqsp | zero-shot | 0.3204 | +0.0309 [+0.0170, +0.0444], **GAIN** | 0.3762 | +0.0111 [-0.0002, +0.0218], WITHIN |
| L-hotpotqa | metaqa | in-domain | 0.8039 | -0.0043 [-0.0060, -0.0027], WITHIN | 0.8040 | -0.0029 [-0.0045, -0.0013], WITHIN |
| L-hotpotqa | squad | in-domain | 0.9129 | -0.0009 [-0.0028, +0.0008], WITHIN | 0.9114 | -0.0002 [-0.0021, +0.0019], WITHIN |
| L-hotpotqa | musique | in-domain | 0.5901 | -0.0087 [-0.0141, -0.0034], **LOSS** | 0.5779 | -0.0090 [-0.0148, -0.0036], **LOSS** |
| L-hotpotqa | hotpotqa | zero-shot | 0.8547 | +0.0028 [-0.0003, +0.0059], WITHIN | 0.8558 | -0.0039 [-0.0072, -0.0007], WITHIN |
| L-hotpotqa | 2wiki | in-domain | 0.9312 | +0.0026 [+0.0008, +0.0044], WITHIN | 0.9331 | -0.0028 [-0.0046, -0.0011], WITHIN |
| L-hotpotqa | webqsp | zero-shot | 0.2914 | +0.0158 [+0.0025, +0.0293], WITHIN (filed GAIN) | 0.3411 | -0.0041 [-0.0172, +0.0093], WITHIN |

The twelve reads of the input call after the seed-null re-call (delta = zspxr's R@5 minus P1g-c's zspx fit's of the
split, both on P1g-b's pools):

| split | dataset | read | zspx on P1g-b's pools R@5 | zspxr: delta [95% CI], call |
| --- | --- | --- | --- | --- |
| L-musique | metaqa | in-domain | 0.7992 | +0.0035 [+0.0025, +0.0045], WITHIN |
| L-musique | squad | in-domain | 0.9131 | -0.0012 [-0.0028, +0.0005], WITHIN |
| L-musique | musique | zero-shot | 0.5206 | -0.0079 [-0.0134, -0.0021], WITHIN (filed LOSS) |
| L-musique | hotpotqa | in-domain | 0.9105 | -0.0031 [-0.0055, -0.0009], WITHIN |
| L-musique | 2wiki | in-domain | 0.9289 | +0.0018 [+0.0002, +0.0034], WITHIN |
| L-musique | webqsp | zero-shot | 0.3513 | -0.0175 [-0.0272, -0.0085], **LOSS** |
| L-hotpotqa | metaqa | in-domain | 0.7996 | +0.0010 [+0.0000, +0.0020], WITHIN |
| L-hotpotqa | squad | in-domain | 0.9120 | +0.0017 [+0.0000, +0.0033], WITHIN |
| L-hotpotqa | musique | in-domain | 0.5814 | +0.0025 [-0.0026, +0.0076], WITHIN |
| L-hotpotqa | hotpotqa | zero-shot | 0.8575 | -0.0055 [-0.0084, -0.0028], WITHIN |
| L-hotpotqa | 2wiki | in-domain | 0.9338 | +0.0007 [-0.0009, +0.0023], WITHIN |
| L-hotpotqa | webqsp | zero-shot | 0.3072 | -0.0357 [-0.0453, -0.0259], **LOSS** |

Beside, deciding nothing: each read's FC@5 and hit@1 deltas against the same bases, from the compare screens
(`outputs/screen/scr-p1gc-<call>.json` and `-hp.json`); and each arm's R@5 delta against P1g-b's own fit of the lean arm
on the same pools (zspx and zspxr against `scr-p1gb-zspk`, zrckpr against `scr-p1gb-zrckp`), from
`outputs/screen/scr-p1gc-lean-<arm>.json` and `-hp.json`, with the pair's call at the 0.0075 floor where it is not
WITHIN. The verdict's `against_p1gb_lean_r5` holds the same deltas taken between the rounded R@5s; they differ from the
compares' by 0.0001 on 8 of the 36 reads.

| split | dataset | zspx FC@5 | zspx hit@1 | zrckpr FC@5 | zrckpr hit@1 | zspxr FC@5 | zspxr hit@1 | zspx vs P1g-b's zspk R@5 | zrckpr vs P1g-b's zrckp R@5 | zspxr vs P1g-b's zspk R@5 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| L-musique | metaqa | -0.0026 | -0.0097 | +0.0011 | -0.0022 | +0.0038 | +0.0115 | -0.0012 | +0.0012 | +0.0023 |
| L-musique | squad | +0.0039 | +0.0017 | +0.0008 | +0.0008 | -0.0012 | -0.0048 | +0.0002 | +0.0008 | -0.0010 |
| L-musique | musique | +0.0058 | +0.0174 | -0.0074 | -0.0021 | -0.0012 | +0.0128 | +0.0038 | +0.0040 | -0.0041 |
| L-musique | hotpotqa | +0.0045 | -0.0022 | -0.0047 | -0.0138 | -0.0023 | +0.0012 | +0.0018 | +0.0024 | -0.0013 |
| L-musique | 2wiki | -0.0148 | +0.0026 | +0.0015 | -0.0017 | +0.0033 | -0.0037 | -0.0055 | -0.0012 | -0.0038 |
| L-musique | webqsp | +0.0299 | -0.0080 | +0.0080 | +0.0200 | -0.0166 | -0.0106 | +0.0159 GAIN | +0.0105 GAIN | -0.0016 |
| L-hotpotqa | metaqa | -0.0051 | -0.0111 | -0.0025 | -0.0029 | +0.0021 | +0.0059 | -0.0015 | +0.0015 | -0.0005 |
| L-hotpotqa | squad | -0.0009 | -0.0039 | -0.0002 | -0.0014 | +0.0017 | +0.0024 | -0.0007 | -0.0010 | +0.0010 |
| L-hotpotqa | musique | -0.0170 | -0.0132 | -0.0087 | -0.0066 | +0.0079 | +0.0021 | +0.0152 GAIN | -0.0006 | +0.0177 GAIN |
| L-hotpotqa | hotpotqa | +0.0020 | +0.0103 | -0.0096 | +0.0005 | -0.0100 | +0.0042 | +0.0091 GAIN | -0.0027 | +0.0036 |
| L-hotpotqa | 2wiki | +0.0076 | -0.0120 | -0.0104 | +0.0035 | -0.0010 | +0.0046 | +0.0013 | -0.0037 | +0.0020 |
| L-hotpotqa | webqsp | +0.0120 | +0.0093 | -0.0053 | +0.0040 | -0.0279 | -0.0093 | +0.0264 GAIN | +0.0006 | -0.0093 LOSS |
