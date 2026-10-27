# A3z: step 1's whole-pool z-scores beside zret's, in both lean models

Declared 19 October 2026, before any number. The next form A3v's verdict named for the GNN arm
(docs/A3V_VSHARE_TWO_SIDED.md, sections 5 and 7: zspky NOT_ADOPTED with no GAIN), under docs/SCREENS.md section 3b and
docs/VERDICT_AUDIT_2026_10_17.md. Code: `outputs/mp_unified/za3z.py`.

## 1. Why this, now

**Two references.** zret (`lean_screen3.py`) takes every z-score the lean model reads against the pool's retrieved
rows, the rows with rrf > 0. Step 1's model (`lean_mlp.py`) took them against the whole pool. Both lean models, zrckp
and zspk, are built on zret.

**The share idea so far.** A3, A3t and A3v (17 to 19 October) moved zret's reference toward the whole pool in training.
The pool's retrieved rows held a share t of the reference:
- A3: t in [0.55, 0.95];
- A3t: t in [0.08, 0.20];
- A3v: vshare's two-sided rule, the rich range on pools whose own retrieved share is under 0.375 and the thin range on
  the others.

On the GNN (zspk's model), none of the three arms gained past the seed null. On the MLP, A3v's zrckpy is
ADOPT_FOR_FULL_RUN on one read, webqsp zero-shot on L-musique (+0.0112 [+0.0008, +0.0211], floor 0.0102). It is being
fitted again on P1g-b's pools (A3vp). A3v's verdict (`outputs/a3v/screen-verdict.json`, written 2026-10-19T13:44:28Z)
named this form for zspky.

**The GNN's reads that moved.** R@5 delta against zspk's card fit, from the re-calls
(`outputs/screen/scr-a3-zspkv-pair-recall.json`, `scr-a3t-zspku-`, `scr-a3v-zspky-`). A call is shown with its filed
call where the re-call changed it:

| read | floor | zspk R@5 | zspkv (A3) | zspku (A3t) | zspky (A3v) |
| --- | ---: | ---: | --- | --- | --- |
| L-musique, musique zero-shot | 0.0720 | 0.5012 | +0.0186 [+0.0131, +0.0243], WITHIN (filed GAIN) | +0.0083 [+0.0027, +0.0142], WITHIN (filed GAIN) | +0.0243 [+0.0183, +0.0306], WITHIN (filed GAIN) |
| L-musique, webqsp zero-shot | 0.0102 | 0.3229 | −0.0103 [−0.0198, −0.0012], LOSS | −0.0027 [−0.0127, +0.0077], WITHIN | −0.0014 [−0.0100, +0.0072], WITHIN |
| L-musique, 2wiki in domain | 0.0075 | 0.9383 | −0.0038 [−0.0052, −0.0024], WITHIN | −0.0019 [−0.0033, −0.0005], WITHIN | −0.0041 [−0.0055, −0.0026], WITHIN |
| L-hotpotqa, musique in domain | 0.0075 | 0.5754 | +0.0072 [+0.0021, +0.0125], WITHIN | −0.0007 [−0.0063, +0.0049], WITHIN | −0.0029 [−0.0081, +0.0027], WITHIN |
| L-hotpotqa, hotpotqa zero-shot | 0.0200 | 0.8499 | −0.0009 [−0.0039, +0.0020], WITHIN | +0.0048 [+0.0018, +0.0078], WITHIN | +0.0003 [−0.0024, +0.0030], WITHIN |
| L-hotpotqa, webqsp zero-shot | 0.0233 | 0.2680 | −0.0086 [−0.0191, +0.0015], WITHIN | +0.0088 [−0.0012, +0.0184], WITHIN | +0.0087 [−0.0006, +0.0177], WITHIN |

What the records show:
- **musique zero-shot rose under every form**, +0.0083 to +0.0243, each interval above 0 and each short of its 0.0720
  floor. The seed null on that read is wide.
- **The other moves change sign between forms:** musique in domain on L-hotpotqa (+0.0072, −0.0007, −0.0029) and webqsp
  zero-shot on L-hotpotqa (−0.0086, +0.0088, +0.0087).
- **The augmentation never shows the model both references at once.** In a training forward each pool's z-scores take
  one reference, and every read takes zret's.

**Why the whole-pool z-scores, as inputs.** On musique's pools (2,092 rows a question on s1eval,
docs/P1C2_POOLS_KIND_T.md), the retrieved rows are a small share of the pool; on hotpotqa's (94 rows) they are most of
it. A3, A3t and A3v asked whether training against a reference nearer the whole pool helps. A3z asks whether reading
both helps: per column, where the row sits among the retrieved rows (zret's z-score) and among the whole pool (step 1's).
Where the two differ, their difference carries the pool's retrieved share, column by column. A3v's section 5 fixed this
form for no GAIN: the share idea reaches both models as an input before it closes (SCREENS.md section 3b).

The base stays the lean pair, as for A3, A3t and A3v, so A3z's reads sit beside theirs. The adopted arms (R1's zrckpr
and G5's zspx) come in the next form if this one finds no GAIN (section 5).

## 2. The form

Per row, before any row is scored, beside every input zret reads:

| block | columns added (each times the block's keep) |
| --- | --- |
| each fixed block (rank, dense_cos, topo_STRUCT, depth_STRUCT, SEED, WALK, DISTS, WALKF: 86 columns on the card fits) | each column's whole-pool z-score: 86 |
| SEMB (learned, 64 columns) | each column's whole-pool z-score: 64 |
| rrf | its whole-pool z-score, a second base term |

- **The statistic:** lean_gpu's `seg_zscore8D`, the z-score over the question's whole pool with the population sd, and 0
  where the sd is under 1e-6. It is step 1's formula on the device. It is also what zret's `seg_zscore_ref` falls back
  to when a pool has fewer than two retrieved rows, or a column's sd over them is under 1e-6.
- **Where the columns differ from zret's z-scores:** only on a pool with at least two retrieved rows and at least one
  other row, and there only on the columns whose sd over the retrieved rows is at least 1e-6. On a pool whose rows are
  all retrieved (squad's), or with fewer than two retrieved rows, A3z's columns are zret's z-scores bit for bit.
- **How they enter:** the 150 columns enter l1's pre-activation through a3z_w, a 128 × 150 matrix (19,200 weights), and
  rrf's whole-pool z-score enters the score through a3z_bw:

```
pre   = l1([zret's inputs]) + a3z_w · [A3z's columns]
score = base_w · z_ret(rrf) + a3z_bw · z_pool(rrf) + out(h)
```

  Both are zero at the start. So each arm starts as its base bit for bit, and no random number is drawn: the global
  stream, dropout's masks included, is the base's. This is A2's template (`za2rank.py`), with step 1's columns in place
  of A2's rank and top-50 columns.
- **The base term:** z_pool(rrf) is step 1's base term. On the cache carves, each batch's base_z is this statistic of
  the same column (`lean_gpu.CacheCarve`), and zret's forward never reads it.

**Worked example.** A3v's six rows, one column (dense_cos), with v1 and v2 retrieved. Both rows of numbers are checked
in the self-test:

| row | retrieved | dense_cos | z (zret's reference) | z (whole pool, A3z's column) |
| --- | --- | ---: | ---: | ---: |
| v1 | yes | 0.81 | 1.00 | 1.27 |
| v2 | yes | 0.78 | −1.00 | 1.10 |
| v3 | no | 0.62 | −11.67 | 0.23 |
| v4 | no | 0.55 | −16.33 | −0.15 |
| v5 | no | 0.40 | −26.33 | −0.98 |
| v6 | no | 0.31 | −32.33 | −1.47 |

zret reads v3 to v6 as far below the two retrieved rows (mean 0.795, sd 0.015). The whole-pool z-score places them
inside the pool's spread (mean 0.578, sd 0.183). A3z gives the model both.

**The MLP rule.** A3z reads no score and normalises nothing over the question's scores. The columns are within-pool
statistics of input columns, the kind zret's z-scores already are. SEMB's are of its learned columns, as zret's own SEMB
z-scores are.

**On record per fit.** Each fit's screen.json stamps the training question visits by pool kind:
- under two retrieved rows (zret's reference is the whole pool);
- all rows retrieved (the same);
- the rest, where the two references differ.

## 3. The arms

| arm | model | carve |
| --- | --- | --- |
| zrckpz | zrckp's model with A3ZMix just above zret (MRO: ZRCKP, KTMixP, ZRM, ChainMatch, A3ZMix, ZRet) | zrckp's (`zp1c3.KTChainCarveZRC`) |
| zspkz | zspk's model with A3ZMix in the same place (MRO: ZSPK, ZProp, KTMix, ZRM, ChainMatch, A3ZMix, ZRet); zsp's propagation reads the score A3z changes | zspk's (`zp1c3.KTLinkCarveBase`) |

A3ZMix re-implements zret's forward to reach l1's pre-activation, as A2Mix does. The base arms and every earlier arm are
unchanged. No build is needed: the columns are computed in each forward from the rows' inputs.

## 4. The fits and the call

A3v's, unchanged, per arm:
- **Fits:** one card fit on L-musique and one on L-hotpotqa (`scr-a3z-<arm>`, `-hp`), on P1c2's roots with P1c3's KT
  input, at the card fits' settings (variant p, config 2e-3:1e-4:0.1:8:2, seed 0, hidden 128).
- **Reads:** each fit on the six s1eval carves at p@swa.
- **Compares:** each fit against its base arm's card fit of the same split (zrckp: scr-p1c3p-zrckp and -hp; zspk:
  scr-p1c3-zspk and -hp), with step 1's fit beside. The re-call's base screens are L1's.
- **Call:** relz's pair and its re-call under the seed null (the four null files of docs/SCREENS.md).

### The rule, fixed before any number

Per arm:
- **ADOPT_FOR_FULL_RUN:** its re-call has no LOSS on any of its twelve reads and at least one GAIN;
- **NOT_ADOPTED:** otherwise;
- **INCOMPLETE:** its re-call is missing, is not A3z's re-call of that arm, or does not hold the twelve reads.

The verdict closes only this form (SCREENS.md section 3b), and it names the next form (section 5).

**Reported beside, deciding nothing:**
- musique on both splits: the A3 family's largest GNN move (zero-shot on L-musique), and Claim 1's read (in domain on
  L-hotpotqa);
- webqsp zero-shot (zrckpy's GAIN) and metaqa;
- hit@1 and FC@5 per read;
- per fit, the training question visits by pool kind;
- A3's, A3t's and A3v's deltas, set beside each read when the results are filed.

**Serving cost.** Per question, one more z-score per column over the pool (the statistic zret already computes as its
fallback), and a 128 × 150 product per row. It is timed with the full run, if one follows.

## 5. What each outcome leads to (fixed now)

| its re-call | the next form |
| --- | --- |
| ADOPT_FOR_FULL_RUN | the arm fitted again on P1g-b's pools, then its full run, its own declared file |
| NOT_ADOPTED, with at least one GAIN and a LOSS | the back-off form, A3zg: A3z's two terms (a3z_w's and a3z_bw's) scaled by 2σ(⟨w, c⟩ + b), w and b zero so the scale starts at 1, with c the pool's log size and retrieved share (input columns only). The model can then turn A3z down on the pools where it misleads; its own declared file |
| NOT_ADOPTED, with no GAIN | A3zx: A3z's columns on the adopted arms, R1's zrckpr and G5's zspx, each against its own base, its own declared file. The share idea closes as an input only if A3zx finds no GAIN either |
| INCOMPLETE | the missing re-call is run; nothing else changes |

No model change is adopted on today's pools (the user's rule of 13 October): an adopt is fitted again on P1g-b's pools
before its full run.

## 6. How it runs

**Commands** (on the host each train and read goes through `outputs/host_ops/retry_cmd.py` and
`outputs/host_ops/cuda_alloc.py`, as the items show):

```
python outputs/mp_unified/za3z.py train --split L-musique --name scr-a3z-zrckpz --arm zrckpz --device cuda --host --cache-root outputs/p1c2/cache
python outputs/mp_unified/za3z.py train --split L-hotpotqa --name scr-a3z-zrckpz-hp --arm zrckpz --device cuda --host --cache-root outputs/p1c2/cache
python outputs/mp_unified/za3z.py train --split L-musique --name scr-a3z-zspkz --arm zspkz --device cuda --host --cache-root outputs/p1c2/cache
python outputs/mp_unified/za3z.py train --split L-hotpotqa --name scr-a3z-zspkz-hp --arm zspkz --device cuda --host --cache-root outputs/p1c2/cache
python outputs/mp_unified/za3z.py read --name scr-a3z-zrckpz --device cuda --host --cache-root outputs/p1c2/cache
    (and scr-a3z-zrckpz-hp, scr-a3z-zspkz, scr-a3z-zspkz-hp)
python outputs/mp_unified/za3z.py compare --new outputs/screen/fits/scr-a3z-zrckpz --base outputs/screen/fits/scr-p1c3p-zrckp,outputs/step1/fits/L-musique --out outputs/screen/scr-a3z-zrckpz
python outputs/mp_unified/za3z.py compare --new outputs/screen/fits/scr-a3z-zrckpz-hp --base outputs/screen/fits/scr-p1c3p-zrckp-hp,outputs/step1/fits/L-hotpotqa --out outputs/screen/scr-a3z-zrckpz-hp
python outputs/mp_unified/za3z.py compare --new outputs/screen/fits/scr-a3z-zspkz --base outputs/screen/fits/scr-p1c3-zspk,outputs/step1/fits/L-musique --out outputs/screen/scr-a3z-zspkz
python outputs/mp_unified/za3z.py compare --new outputs/screen/fits/scr-a3z-zspkz-hp --base outputs/screen/fits/scr-p1c3-zspk-hp,outputs/step1/fits/L-hotpotqa --out outputs/screen/scr-a3z-zspkz-hp
python outputs/mp_unified/za3z.py pair --arm zrckpz --screens outputs/screen/scr-a3z-zrckpz.json,outputs/screen/scr-a3z-zrckpz-hp.json --out outputs/screen/scr-a3z-zrckpz-pair
python outputs/mp_unified/za3z.py recall --arm zrckpz --null outputs/screen/scr-null-s1.json,outputs/screen/scr-null-s2.json,outputs/screen/scr-null-s1-hp.json,outputs/screen/scr-null-s2-hp.json --pair outputs/screen/scr-a3z-zrckpz-pair.json --out outputs/screen/scr-a3z-zrckpz-pair-recall
    (pair and recall the same for zspkz)
python outputs/mp_unified/za3z.py verdict
```

**Self-test.** `python outputs/mp_unified/za3z.py --selftest` passes before this commit (4.4 s). It checks:
- **the arms:** the MROs, with A3ZMix just above zret in both models; the base arms unchanged;
- **the columns on a hand-made batch** of five pools (A3v's worked example and four more):
  - they are the whole-pool z-scores by hand; on the worked example, 1.27 to −1.47 beside zret's 1.00 to −32.33;
  - they are zret's own z-scores bit for bit on a pool whose rows are all retrieved, one with a single retrieved row, one
    whose column is constant over its retrieved rows (or over the pool), and a one-row pool;
  - the pool kinds are counted;
- **the columns on a toy batch** with rmatch's chains:
  - they match by hand, SEMB's learned columns included; a question without SEMB gets none;
  - rrf's is the cache carves' base_z (read from their source; the toy carve's base_z takes another column);
  - zret's own l1 input is unchanged;
  - the kinds are counted in training forwards only;
- **each arm against its base**, from one seed:
  - no random number is drawn, and the new parameters are a3z_w and a3z_bw, zero;
  - in eval and in training, the scores and every shared gradient are the base's bit for bit;
  - a3z_bw has a gradient at the start and a3z_w none, as l1 has none (out starts at zero);
  - with both moved, the scorer is zret's on A3z's inputs, and zspkz's propagation reads it;
- **toy fits** through lean_gpu's loop repeat bit for bit, A3z's weights learn, and the kinds are counted;
- **the refusals:** train refuses another arm, split, fit name, cache root or device; read refuses a fit that is not
  A3z's or not its split's;
- relz's pair and re-call per arm, stamped A3z's, with relz restored after; the verdict's outcomes and next forms (a
  re-call stamped A3v's, or of the other arm, or missing, is INCOMPLETE).

**Host items** (17, `a3z-*`), sized as A3v's:
- **trains:** 4, with no build to wait on, at A3v's `--frac` and GPU share (L-musique 0.20/0.22, L-hotpotqa 0.27/0.29);
- **reads:** 4 at 0.27/0.29. A2's items, the same template with 322 columns in place of 150, peaked at 5.86 GB on the
  card for a read and 5.55 GB for a train (the feeder's measured peaks), under 0.29 of the card's 24 GB;
- **the call:** 4 compares, 2 pairs, 2 re-calls (after L1's base screens) and the verdict;
- **memory:** A3v's, item by item (trains 4.5 GB, zspkz-hp 7 GB; reads 6 GB; compares 4 GB; the rest 2 GB). The 150
  columns add about 0.16 GB per float32 copy on a 262,144-row read batch.

They go after A4he's items. **Disk:** about 0.3 GB (four fits and their reads).

## 7. Results

Filed 27 October 2026 from the host's records (`outputs/a3z/screen-verdict.json`, written 2026-10-20T20:07:32Z; the
re-calls `outputs/screen/scr-a3z-zrckpz-pair-recall.json` and `scr-a3z-zspkz-pair-recall.json`, host paths made
relative). **Both arms NOT_ADOPTED, with no GAIN:** with step 1's whole-pool z-scores beside zret's, both models lose
webqsp zero-shot on L-musique, and the GNN's arm loses 2wiki in domain there as well.

- **zrckpz: 0 GAIN, 1 LOSS.** The LOSS is webqsp zero-shot on L-musique: -0.0153 [-0.0252, -0.0058] against its 0.0102
  floor, with FC@5 -0.0180. Three more intervals exclude 0, all inside their floors: on L-musique, metaqa -0.0018
  [-0.0028, -0.0009] and 2wiki -0.0042 [-0.0059, -0.0026]; on L-hotpotqa, metaqa -0.0014 [-0.0023, -0.0006]. No call
  changed under the re-call.
- **zspkz: 0 GAIN, 2 LOSS.** Both LOSSes are on L-musique: 2wiki in domain, -0.0089 [-0.0105, -0.0073] against its
  0.0075 floor, with FC@5 -0.0222; and webqsp zero-shot, -0.0122 [-0.0219, -0.0023] against its 0.0102 floor, with FC@5
  -0.0073. Two filed GAINs are WITHIN under the re-call: musique zero-shot on L-musique, +0.0183 [+0.0129, +0.0237]
  against its 0.0720 floor (FC@5 +0.0178), and webqsp zero-shot on L-hotpotqa, +0.0168 [+0.0068, +0.0268] against its
  0.0233 floor (FC@5 +0.0113). Four more intervals exclude 0, all inside their floors: on L-musique, metaqa +0.0014
  [+0.0004, +0.0024] and hotpotqa -0.0033 [-0.0055, -0.0009]; on L-hotpotqa, metaqa -0.0025 [-0.0035, -0.0015] and
  hotpotqa zero-shot -0.0072 [-0.0103, -0.0040].
- **Beside: A3v's GAIN does not carry over.** A3v's two-sided share on the MLP (zrckpy) gained webqsp zero-shot on
  L-musique (+0.0112); the whole-pool z-scores lose that read on both models (-0.0153 and -0.0122).

Not adopted. This closes only this form: step 1's whole-pool z-scores beside zret's in the lean pair (docs/SCREENS.md
section 3b). With no GAIN on either arm, the next form fixed in section 5 and recorded in the verdict for both arms is
A3zx, A3z's columns on the adopted arms, R1's zrckpr and G5's zspx, each against its own base, its own declared file;
the share idea closes as an input only if A3zx finds no GAIN either.

The floors are the seed null of 7 October (docs/SCREENS.md section 2: `outputs/screen/scr-null-s1.json`,
`scr-null-s2.json` and their `-hp` files), not N2's fresh floors (docs/N2_FRESH_SEED_NULL.md).

Records, filed with this section: the verdict `outputs/a3z/screen-verdict.json` and `.md`; for each arm (`zrckpz`,
`zspkz`), the compares `outputs/screen/scr-a3z-<arm>.json` and `-hp.json`, the pair `-pair.json` and its re-call
`-pair-recall.json`, each with its `.md`. The host paths were made relative in each re-call's two `rel` entries and
nowhere else, so the `recall_sha256` each verdict holds is the host file's, from before that change. The re-calls' nulls
also read the base screens `outputs/l1/base-zrckp-L-musique.json` and `outputs/l1/base-zrckp-L-hotpotqa.json` and
`outputs/l1/base-zspk-L-musique.json` and `outputs/l1/base-zspk-L-hotpotqa.json`, already in the repository.

Each arm's twelve reads after the seed-null re-call (delta = the arm's R@5 minus its base's, the base's card fit of the
split; a call is shown with its filed call where the re-call changed it):

| split | dataset | read | zrckp R@5 | zrckpz: delta [95% CI], call | zspk R@5 | zspkz: delta [95% CI], call |
| --- | --- | --- | --- | --- | --- | --- |
| L-musique | metaqa | in-domain | 0.8019 | -0.0018 [-0.0028, -0.0009], WITHIN | 0.8009 | +0.0014 [+0.0004, +0.0024], WITHIN |
| L-musique | squad | in-domain | 0.9116 | -0.0006 [-0.0023, +0.0010], WITHIN | 0.9114 | +0.0000 [-0.0019, +0.0019], WITHIN |
| L-musique | musique | zero-shot | 0.5121 | -0.0015 [-0.0063, +0.0037], WITHIN | 0.5012 | +0.0183 [+0.0129, +0.0237], WITHIN (filed GAIN) |
| L-musique | hotpotqa | in-domain | 0.9065 | -0.0018 [-0.0038, +0.0005], WITHIN | 0.9080 | -0.0033 [-0.0055, -0.0009], WITHIN |
| L-musique | 2wiki | in-domain | 0.9347 | -0.0042 [-0.0059, -0.0026], WITHIN | 0.9383 | -0.0089 [-0.0105, -0.0073], **LOSS** |
| L-musique | webqsp | zero-shot | 0.3627 | -0.0153 [-0.0252, -0.0058], **LOSS** | 0.3229 | -0.0122 [-0.0219, -0.0023], **LOSS** |
| L-hotpotqa | metaqa | in-domain | 0.8044 | -0.0014 [-0.0023, -0.0006], WITHIN | 0.8040 | -0.0025 [-0.0035, -0.0015], WITHIN |
| L-hotpotqa | squad | in-domain | 0.9127 | -0.0003 [-0.0020, +0.0017], WITHIN | 0.9135 | -0.0003 [-0.0024, +0.0018], WITHIN |
| L-hotpotqa | musique | in-domain | 0.5745 | -0.0023 [-0.0077, +0.0029], WITHIN | 0.5754 | -0.0040 [-0.0091, +0.0009], WITHIN |
| L-hotpotqa | hotpotqa | zero-shot | 0.8513 | +0.0029 [-0.0002, +0.0060], WITHIN | 0.8499 | -0.0072 [-0.0103, -0.0040], WITHIN |
| L-hotpotqa | 2wiki | in-domain | 0.9350 | -0.0005 [-0.0021, +0.0012], WITHIN | 0.9333 | -0.0003 [-0.0019, +0.0014], WITHIN |
| L-hotpotqa | webqsp | zero-shot | 0.3475 | +0.0039 [-0.0067, +0.0144], WITHIN | 0.2680 | +0.0168 [+0.0068, +0.0268], WITHIN (filed GAIN) |

Beside, deciding nothing: each read's FC@5 and hit@1 deltas against the same base, from the compare screens
(`outputs/screen/scr-a3z-<arm>.json` and `-hp.json`); and A3v's R@5 deltas with its two-sided share (zrckpy against
zrckp, zspky against zspk; their own re-calls, `outputs/screen/scr-a3v-<arm>-pair-recall.json`), on the same read (a
call is shown where it is not WITHIN):

| split | dataset | zrckpz FC@5 | zrckpz hit@1 | zspkz FC@5 | zspkz hit@1 | zrckpy (A3v) R@5 | zspky (A3v) R@5 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| L-musique | metaqa | -0.0012 | -0.0098 | +0.0025 | +0.0098 | +0.0003 | +0.0003 |
| L-musique | squad | -0.0006 | -0.0013 | +0.0000 | +0.0003 | +0.0015 | +0.0008 |
| L-musique | musique | -0.0004 | +0.0248 | +0.0178 | +0.0046 | -0.0111 | +0.0243 |
| L-musique | hotpotqa | -0.0023 | -0.0024 | -0.0068 | -0.0003 | -0.0022 | -0.0034 |
| L-musique | 2wiki | -0.0108 | +0.0033 | -0.0222 | +0.0014 | +0.0014 | -0.0041 |
| L-musique | webqsp | -0.0180 | -0.0113 | -0.0073 | -0.0106 | +0.0112 GAIN | -0.0014 |
| L-hotpotqa | metaqa | -0.0020 | +0.0005 | -0.0018 | -0.0106 | -0.0019 | +0.0003 |
| L-hotpotqa | squad | -0.0003 | +0.0004 | -0.0003 | -0.0034 | +0.0003 | -0.0004 |
| L-hotpotqa | musique | -0.0008 | -0.0033 | -0.0087 | -0.0025 | +0.0004 | -0.0029 |
| L-hotpotqa | hotpotqa | +0.0057 | -0.0153 | -0.0077 | -0.0190 | +0.0038 | +0.0003 |
| L-hotpotqa | 2wiki | -0.0015 | -0.0019 | +0.0031 | +0.0020 | -0.0013 | +0.0024 |
| L-hotpotqa | webqsp | +0.0033 | +0.0033 | +0.0113 | +0.0007 | +0.0036 | +0.0087 |
