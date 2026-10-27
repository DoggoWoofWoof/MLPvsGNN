# A11x: our LLM-free phrase links as a fourth family in all three of the GNN's steps

Declared 19 October 2026, before any number. The next form A11's verdict named for zspk4
(docs/A11_FOURTH_FAMILY_LINKS.md, section 5), under docs/SCREENS.md section 3b. Code: `outputs/mp_unified/za11x.py`
(the arm, the screen and the call; the build is A11's, `outputs/mp_unified/za11p4.py`, unchanged).
Filed 27 October: NOT_ADOPTED with no GAIN and no LOSS; the next form is A11m (section 5).

## 1. Why this, now

**What A11 found.** A11 put our phrase links into zspk's one propagation step as a fourth link family (zspk4). Its
verdict (outputs/a11/screen-verdict.json, 18 October) is NOT_ADOPTED: 0 GAIN and 1 LOSS of twelve reads after the
seed-null re-call (outputs/screen/scr-a11-zspk4-pair-recall.json).

The reads that moved, zspk4 minus zspk on R@5 with the re-call's interval and the read's floor:

| fit | read | difference [interval] | floor | call |
| --- | --- | --- | --- | --- |
| L-musique | webqsp (zero-shot) | −0.0137 [−0.0217, −0.0054] | 0.0102 | LOSS |
| L-hotpotqa | hotpotqa (zero-shot) | +0.0045 [+0.0014, +0.0073] | 0.0200 | WITHIN |
| L-hotpotqa | 2wiki | +0.0024 [+0.0008, +0.0040] | 0.0075 | WITHIN |
| L-musique | metaqa | +0.0014 [+0.0007, +0.0022] | 0.0075 | WITHIN |

The other eight reads' intervals cross zero (differences −0.0016 to +0.0058). The largest of them are musique in
domain, +0.0047 [−0.0006, +0.0099], and webqsp, +0.0058 [−0.0043, +0.0154], both on L-hotpotqa.

**Why that leads here, not to a close.**
- The LOSS is on webqsp, a knowledge base. There zspk4 is zspk by rule: no passages, so no fourth-family edge. The
  move comes from the fit's other datasets, through the shared scorer.
- The three reads whose intervals sit above zero (hotpotqa and 2wiki on L-hotpotqa, metaqa on L-musique) are small,
  each inside its floor.
- A11 tested one step only. G5 found that depth pays: zspx (zspk's step with the gradient kept, G3's residual step,
  G4's second step) is the adopted GNN, with 2 GAIN and 0 LOSS against zspk. Both GAINs are on L-hotpotqa: musique
  +0.0147 [+0.0095, +0.0198] and webqsp +0.0235 [+0.0133, +0.0327] (outputs/screen/scr-g5-zspx-pair-recall.json;
  filed in ac9a1e9).
- A11's section 5 fixed this form for its outcome (NOT_ADOPTED with no GAIN): "A11x, the fourth family in all three
  of zspx's steps (the adopted GNN, where depth paid), its own declared file".

Under section 3b, A11's verdict closed only zspk4's one-step form.

**The family's reach** (A11's build records, commit 96f2489): the phrase links reach 83% to 85% of musique's pool rows
(6.4 to 7.6 edges per row), 81% of squad's (5.1 to 5.3), and 45% to 51% of hotpotqa's and 2wiki's (2.2 to 2.7).

## 2. The form

**The base is zspx** (zg5mix.ZSPX): zspk's scorer and KT head give s0, then three steps over the link graph's three
families:

    s1 = s0 + W2sp gelu(W1sp x10(s0) + b1sp) + b2sp      zspx's first step (zspkg's): x10 keeps the gradient
    s2 = s1 + W2rs gelu(W1rs r10(s1) + b1rs) + b2rs      G3's bridge-residual step: no gradient
    s3 = s2 + W2p2 gelu(W1p2 x10(s2) + b1p2) + b2p2      G4's second step: no gradient

**zspx4 adds the fourth family to each step's head, before its GELU:**

    s1 = s0 + W2sp gelu(W1sp x10(s0) + b1sp + W4sp x4(s0)) + b2sp
    s2 = s1 + W2rs gelu(W1rs r10(s1) + b1rs + W4rs x4(s1)) + b2rs
    s3 = s2 + W2p2 gelu(W1p2 x10(s2) + b1p2 + W4p2 x4(s2)) + b2p2

- **x4(s)** is A11's three columns of a score s, per row: the mean and the soft maximum (log-sum-exp) of its
  fourth-family neighbours' z-scores (within the question, clipped at ±8), and log1p of their count. It is
  zprop.prop_inputs's own code with every edge in its first family (za11p4.p4_inputs).
- **The gradient follows each step's own inputs.** In the first step, x4 keeps the gradient back to s0, as x10 does
  there (zg4prop.prop_inputs_grad's code; the same numbers). In the second and third steps x4 carries none, as r10 and
  x10 carry none there.
- **Why the residual step reads x4, not residual columns.** G3's residual columns need each edge's bridge similarity,
  which G3 builds only for zlink's edges. The fourth family's edges have none, and building it would be a new build.
  So the residual step reads the same three columns as the other two.
- **The weights.** Each W4 is 32 × 3, uniform in ±1/√10 (its head's W1 scale: every head has 10 inputs and 32 hidden
  units), from its own generator: seed + 1101 for the first step (A11's own draw, so W4sp equals zspk4's W4), seed +
  1102 for the second and seed + 1103 for the third. That is 288 parameters. Every other weight is zspx's own draw.
- **Where zspx4 is zspx bit for bit.** A row with no fourth-family neighbour reads x4 = 0. So zspx4 is zspx at the
  start (each step's W2 is zero), on the knowledge bases (metaqa and webqsp hold no passages, so no edges), and on any
  question the family does not reach. Once a step's W2 moves, a question the family reaches can move on all its rows:
  the later steps read the question's z-scores, which its moved rows change.

**The edges are A11's build, unchanged** (za11p4.py build). For each question of a P1c2 cache part, every pair of pool
rows whose passages share a phrase that 2 to 25 passages hold gives two directed edges, stored int16 and pool-local,
tied to the part's step-1 record. No new constant, no new graph, no LLM, no encoder change.

## 3. The arms

| arm | the model | the carve |
| --- | --- | --- |
| zspx4 | zspx with W4 over the fourth family's columns in each of its three steps | zspx's (G3's residual carve over P1c3's KT link carve) with the fourth family's edges beside (za11p4.p4_carve) |

**GNN only.** The MLP takes no message. The family reaches the MLP as a fixed per-row column (A11m, section 5).

## 4. The fits and the call

- **Fits:** one card fit on L-musique and one on L-hotpotqa (`scr-a11x-zspx4`, `-hp`), on P1c2's roots with P1c3's KT
  input. Card fit settings: variant p, config 2e-3:1e-4:0.1:8:2, seed 0, hidden 128 (zspx's).
- **Reads:** each fit on the six s1eval carves at p@swa.
- **Compares:** each fit against zspx's card fit of the same split (scr-g5-zspx, scr-g5-zspx-hp), which decides.
  Step 1's fit is reported beside.
- **The re-call's base screens:** A9's, zspx's card fits compared with step 1's of the same split
  (outputs/a9/base-zspx-<split>).
- **Call:** relz's pair and its re-call under the seed null (the four null files of docs/SCREENS.md). GAIN, LOSS and
  WITHIN are docs/SCREENS.md's, on R@5.

### The rule, fixed before any number

- **ADOPT_FOR_FULL_RUN:** zspx4's re-call has no LOSS on any of its twelve reads and at least one GAIN;
- **NOT_ADOPTED:** otherwise;
- **INCOMPLETE:** its re-call is missing, is not A11x's re-call of zspx4, or does not hold the twelve reads.

The verdict closes only this form (SCREENS.md section 3b), and it names the next form (section 5).

**Reported beside, deciding nothing:**
- webqsp and metaqa, both splits (the knowledge bases, where zspx4 is zspx by rule: any move there comes from the
  fit's other datasets);
- musique in domain (L-hotpotqa): Claim 1's read;
- hit@1 and FC@5 per read;
- each rebuilt part's counts: edges per row, and the share of rows with a fourth-family neighbour.

## 5. What each outcome leads to (fixed now)

| its re-call | the next form |
| --- | --- |
| ADOPT_FOR_FULL_RUN | zspx4 fitted again on P1g-b's pools (A11's build run on them first), then its full run (its own declared file); and A11m for the MLP: the family's per-row degree (log1p; no score in it) as a fixed column in zrckp through R1's pointwise head, as zrckpr reads R1's columns (its own declared file) |
| NOT_ADOPTED, with at least one GAIN and a LOSS | A11xd, the back-off form: zspx4 under edge-family dropout over its four families (FD 0.25, qd_gnn13's fd25 rule through zg2.family_keep, as A9 ran it on zspx), so no step can lean on the fourth family where its mix differs; and A11m for the MLP (one declared file) |
| NOT_ADOPTED, with no GAIN | A11m in both models: the family's per-row degree (log1p; no score in it) as a fixed column through R1's pointwise head, in zrckp (as zrckpr reads R1's columns) and in zspx (as zspxr does), each free to learn how much to use it (one declared file). This closes zspx4's form only: as a message the family reached zspk's one step (A11) and all three of zspx's steps (A11x) with no GAIN, so it goes on as a fixed input |
| INCOMPLETE | the missing re-call is run; nothing else changes |

No model change is adopted on today's pools (the user's rule of 13 October).

## 6. How it runs

**Self-test, passed before this commit.** `python outputs/mp_unified/za11x.py --selftest` (5 s) checks, on a toy
carve with G3's residuals, P1c3's KT columns and A11's toy fourth family (a third of the questions draw their rows from
passages the family links to none):
- **The arm.** It is zspx plus three W4s on G3's carve with the fourth family beside. zspx's, zspk4's and zspk's arms
  are unchanged. The 288 parameters and the three seed offsets, none another head's.
- **The rebuild.** With a part's arrays deleted and its record kept (the host's state now), the records pass and the
  carve stops. A11's build, run again over the existing record, writes the three arrays back byte for byte: their
  sha256 equal the first build's and the new record's.
- **The batch.** Every key of zspx's batch is unchanged. The fourth family's edges are a brute force's; a knowledge
  base gets none.
- **Bit-for-bit cases.**
  - zspx4's state is zspx's plus the three W4s, each its own generator's draw; W4sp equals zspk4's W4.
  - At the start, zspx4 is zspx.
  - With one step's W2 moved (each in turn), zspx4 moves only the rows with a fourth-family neighbour. With the first
    step's alone, it is A11's zspk4 on the same weights.
  - With all three moved, the questions the family does not reach are zspx's, and some rows without a neighbour in
    a reached question move. It is zspx on the knowledge base, without the family's key, and without links.
- **The columns** match a float64 brute force, and rows without a neighbour read 0. The first step's gradient-kept
  columns are the same numbers and carry the gradient to the scorer: with them detached, the scorer's gradient
  changes; on the knowledge base it is zspx's.
- **Toy fits through lean_gpu's loop.** A repeat is bit for bit. Each W4 ends elsewhere than in the knowledge-base
  fit, where only weight decay moves it. The family moves the fit. On the knowledge base, the fit is zspx's bit for
  bit on every shared key.
- **The rest.** The train and read refusals; relz's pair and re-call stamped A11x's; the verdict's outcomes and next
  forms, with A11's own re-call refused as INCOMPLETE.

**Host items** (`a11x-*`, 25), each starting when its inputs are ready:
- **16 rebuilds** (`a11x-b-p4-<dataset>-<carve>-<part>`): A11's build items with only the name changed, because A11's
  output folder holds only records now. The build does not skip a part whose record exists: it writes the arrays
  (each through a temporary file) and then the record again.
  - the fit carves: squad 1, hotpotqa 1, 2wiki 1, musique's s1fit 6;
  - the s1eval carves: squad 1, hotpotqa 1, 2wiki 2, musique 3.

  Each takes 1 CPU through pylib_run, with A11's memory: 2wiki 12 GB, hotpotqa 8 GB, musique 3 GB, squad 2 GB.
- **The screen:**
  - 2 trains and 2 reads, GPU sized as A9's zspx items: `--frac 0.40` (share 0.42); memory: trains 6 GB (-hp 15 GB),
    reads 6 GB;
  - the trains wait on their split's G3 residual rebuilds (`n19-res-*`, the carve lists A9's trains used: 11 for
    L-musique, 16 for L-hotpotqa) and the fourth family's fit-carve rebuilds (3 and 8); the reads wait on their train,
    the 22 s1eval residual rebuilds and the 7 s1eval family rebuilds;
  - 2 compares, the pair, the re-call and the verdict. The re-call reads A9's base screens, already on the host.

The commands (each train and read under `retry_cmd.py --tries 3 --wait 90 --cuda` and `cuda_alloc.py --frac 0.40
--gc 0.9`, as in the items):

    python outputs/host_ops/pylib_run.py outputs/mp_unified/za11p4.py build --dataset musique --carve s1fit \
        --part 0 --host
    python outputs/mp_unified/za11x.py train --split L-musique --name scr-a11x-zspx4 --arm zspx4 --device cuda \
        --host --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za11x.py read --name scr-a11x-zspx4 --device cuda --host \
        --cache-root outputs/p1c2/cache
    python outputs/mp_unified/za11x.py compare --new outputs/screen/fits/scr-a11x-zspx4 \
        --base outputs/screen/fits/scr-g5-zspx,outputs/step1/fits/L-musique --out outputs/screen/scr-a11x-zspx4
    python outputs/mp_unified/za11x.py pair \
        --screens outputs/screen/scr-a11x-zspx4.json,outputs/screen/scr-a11x-zspx4-hp.json \
        --out outputs/screen/scr-a11x-zspx4-pair
    python outputs/mp_unified/za11x.py recall --null outputs/screen/scr-null-s1.json,outputs/screen/scr-null-s2.json,\
    outputs/screen/scr-null-s1-hp.json,outputs/screen/scr-null-s2-hp.json \
        --pair outputs/screen/scr-a11x-zspx4-pair.json --out outputs/screen/scr-a11x-zspx4-pair-recall
    python outputs/mp_unified/za11x.py verdict

L-hotpotqa's train, read and compare are the same with `--split L-hotpotqa`, `scr-a11x-zspx4-hp`, `scr-g5-zspx-hp`
and `outputs/step1/fits/L-hotpotqa`.

## 7. Results

Filed 27 October 2026 from the host's records (`outputs/a11x/screen-verdict.json`, written 2026-10-20T10:49:26Z; the
re-call `outputs/screen/scr-a11x-zspx4-pair-recall.json`, host paths made relative). **NOT_ADOPTED, with no GAIN and no
LOSS:** with our LLM-free phrase links as a fourth family in all three of zspx's steps, no read moves past its floor,
and the one filed LOSS, webqsp zero-shot on L-hotpotqa, is inside its floor under the re-call.

- **zspx4: 0 GAIN, 0 LOSS.** webqsp zero-shot on L-hotpotqa falls -0.0145 [-0.0229, -0.0061], a LOSS by its pair's
  interval that is WITHIN under its 0.0233 floor, with FC@5 -0.0093. Four more intervals exclude 0, all inside their
  floors: on L-musique, metaqa -0.0012 [-0.0020, -0.0004] and squad +0.0025 [+0.0005, +0.0045]; on L-hotpotqa, metaqa
  +0.0011 [+0.0001, +0.0019] and 2wiki +0.0015 [+0.0001, +0.0029].
- **Beside: A11's LOSS is gone, and none takes its place past a floor.** In zspk's one step (A11's zspk4), the family
  lost webqsp zero-shot on L-musique (-0.0137); in zspx's three steps that read moves -0.0028, and webqsp zero-shot on
  L-hotpotqa moves from +0.0058 to -0.0145. On musique zero-shot on L-musique, R@5 moves -0.0020 while hit@1 rises
  +0.0439.

Not adopted. This closes only this form: the fourth family as a message in all three of zspx's steps (docs/SCREENS.md
section 3b). The next form, fixed in section 5 and recorded in the verdict, is A11m in both models: the family's per-row
degree (log1p; no score in it) as a fixed column through R1's pointwise head, in zrckp (as zrckpr reads R1's columns)
and in zspx (as zspxr does), each free to learn how much to use it (one declared file). This closes zspx4's form only:
as a message the family reached zspk's one step (A11) and all three of zspx's steps (A11x) with no GAIN, so it goes on
as a fixed input.

The floors are the seed null of 7 October (docs/SCREENS.md section 2: `outputs/screen/scr-null-s1.json`,
`scr-null-s2.json` and their `-hp` files), not N2's fresh floors (docs/N2_FRESH_SEED_NULL.md).

Records, filed with this section: the verdict `outputs/a11x/screen-verdict.json` and `.md`; for the arm `zspx4`, the
compares `outputs/screen/scr-a11x-zspx4.json` and `-hp.json`, the pair `-pair.json` and its re-call `-pair-recall.json`,
each with its `.md`. The host paths were made relative in each re-call's two `rel` entries and nowhere else, so the
`recall_sha256` each verdict holds is the host file's, from before that change. The re-calls' nulls also read the base
screens `outputs/a9/base-zspx-L-musique.json` and `outputs/a9/base-zspx-L-hotpotqa.json`, fetched from the host with
this filing (A9's own filing had left them there).

The twelve reads after the seed-null re-call (delta = the arm's R@5 minus its base's, zspx's card fit of the split; a
call is shown with its filed call where the re-call changed it):

| split | dataset | read | zspx R@5 | zspx4: delta [95% CI], call |
| --- | --- | --- | --- | --- |
| L-musique | metaqa | in-domain | 0.8030 | -0.0012 [-0.0020, -0.0004], WITHIN |
| L-musique | squad | in-domain | 0.9092 | +0.0025 [+0.0005, +0.0045], WITHIN |
| L-musique | musique | zero-shot | 0.5073 | -0.0020 [-0.0072, +0.0034], WITHIN |
| L-musique | hotpotqa | in-domain | 0.9074 | +0.0011 [-0.0011, +0.0034], WITHIN |
| L-musique | 2wiki | in-domain | 0.9351 | -0.0012 [-0.0025, +0.0003], WITHIN |
| L-musique | webqsp | zero-shot | 0.3204 | -0.0028 [-0.0087, +0.0031], WITHIN |
| L-hotpotqa | metaqa | in-domain | 0.8039 | +0.0011 [+0.0001, +0.0019], WITHIN |
| L-hotpotqa | squad | in-domain | 0.9129 | +0.0002 [-0.0018, +0.0022], WITHIN |
| L-hotpotqa | musique | in-domain | 0.5901 | -0.0020 [-0.0069, +0.0028], WITHIN |
| L-hotpotqa | hotpotqa | zero-shot | 0.8547 | -0.0017 [-0.0043, +0.0011], WITHIN |
| L-hotpotqa | 2wiki | in-domain | 0.9312 | +0.0015 [+0.0001, +0.0029], WITHIN |
| L-hotpotqa | webqsp | zero-shot | 0.2914 | -0.0145 [-0.0229, -0.0061], WITHIN (filed LOSS) |

Beside, deciding nothing: each read's FC@5 and hit@1 deltas against the same base, from the compare screens
(`outputs/screen/scr-a11x-zspx4.json` and `-hp.json`); and A11's R@5 deltas with the family in zspk's one step (zspk4,
its own re-call `outputs/screen/scr-a11-zspk4-pair-recall.json`, against zspk), on the same read (a call is shown where
it is not WITHIN):

| split | dataset | zspx4 FC@5 | zspx4 hit@1 | zspk4 (A11) R@5 |
| --- | --- | ---: | ---: | ---: |
| L-musique | metaqa | -0.0008 | -0.0034 | +0.0014 |
| L-musique | squad | +0.0025 | -0.0013 | +0.0009 |
| L-musique | musique | -0.0046 | +0.0439 | +0.0033 |
| L-musique | hotpotqa | +0.0026 | +0.0001 | -0.0016 |
| L-musique | 2wiki | -0.0014 | +0.0064 | -0.0012 |
| L-musique | webqsp | +0.0007 | +0.0060 | -0.0137 LOSS |
| L-hotpotqa | metaqa | +0.0005 | +0.0044 | -0.0003 |
| L-hotpotqa | squad | +0.0002 | -0.0012 | -0.0014 |
| L-hotpotqa | musique | -0.0058 | -0.0050 | +0.0047 |
| L-hotpotqa | hotpotqa | -0.0024 | -0.0014 | +0.0045 |
| L-hotpotqa | 2wiki | +0.0025 | -0.0059 | +0.0024 |
| L-hotpotqa | webqsp | -0.0093 | -0.0067 | +0.0058 |
