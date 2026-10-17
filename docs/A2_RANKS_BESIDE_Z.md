# A2: pool ranks and top-50 statistics beside the z-scores, in both models

Declared 17 October 2026, before any number. The second reopening of docs/VERDICT_AUDIT_2026_10_17.md (section 2,
row 2), under docs/SCREENS.md section 3b. Code: `outputs/mp_unified/za2rank.py`. Filed 17 October: both arms NOT_ADOPTED (section 7).

## 1. Why this, now

Two input forms gained on musique zero-shot and were dropped:

| screen | what it did | musique zero-shot (L-musique) | metaqa in domain | re-call |
| --- | --- | --- | --- | --- |
| ztop50 (8 Oct, lean_screen4.py) | every z-score taken against the pool's 50 best retrieved rows by rrf, in place of all retrieved rows | +0.1387 [+0.1279, +0.1492], GAIN | −0.0176 (L-musique), −0.0252 (L-hotpotqa), LOSS | MIXED |
| prank (8 Oct, lean_screen5.py) | every fixed block's z-score replaced by two rank columns | +0.2016 [+0.1901, +0.2126], GAIN | −0.0310, −0.0320, LOSS | MIXED |
| pret (lean_screen8.py) | prank on the two retrieval blocks only | | | NO_GAIN |

All three **replaced** the within-pool z-scores, so metaqa lost what the z-scores carry. None was tried **beside**
them, and none reached the GNN. Under section 3b, a verdict closes only the form it tested, so the idea is still
open. The form is now: both signals beside zret's z-scores, in both of the lean pair's models.

All three ran on step 1's model and caches. Today's pair (zrckp, zspk) is on P1c2's pools, with the KT input and,
for the GNN, propagation. The screen is run again on today's pair, not carried forward.

## 2. The input

Nothing zret reads changes. For each row, before any row is scored, A2 adds:

| block | columns added (each times the block's keep) |
| --- | --- |
| each fixed block (rank, dense_cos, topo_STRUCT, depth_STRUCT, SEED, WALK, DISTS, WALKF: 86 columns on the card fits) | z_top, hi, lo per column: 258 |
| SEMB (learned, 64 columns) | z_top per column: 64 |
| rrf | its z_top, as a second base term |

- **z_top** is the column's z-score against the pool's 50 best retrieved rows by rrf (rrf > 0; ties by row order).
  This is ztop50's reference, through `lean_screen4.top_ref`. A pool with fewer than two such rows, or a column
  constant over them, takes the whole pool's z-score, as zret does (`lean_screen3.seg_zscore_ref`).
- **hi and lo** are the column's competition rank in the pool, from the top and from the bottom (ties share the best
  position), each as 60 / (60 + min(rank, 50)). These are prank's columns (`lean_screen5.rank_inputs`), unchanged.

Why both: a z-score moves with the pool's size and tail. In a 2,000-row pool a retrieved node sits far out in its
column's tail; in a 60-row pool it does not. A rank among the top 50, and a z-score against a fixed-size top 50,
move much less. The z-scores stay, because metaqa in domain needs what they carry.

**Worked example** (made-up numbers; k = 3 on a six-row pool stands in for k = 50 on a big one). One column,
dense_cos:

| row | rrf | dense_cos | z (retrieved rows, zret) | z_top (top 3 by rrf) | hi | lo |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v1 | 0.032 | 0.81 | 1.11 | 0.88 | 1.000 | 0.923 |
| v2 | 0.030 | 0.78 | 0.83 | 0.52 | 0.984 | 0.938 |
| v3 | 0.029 | 0.62 | −0.65 | −1.40 | 0.968 | 0.952 |
| v4 | 0.016 | 0.55 | −1.29 | −2.24 | 0.952 | 0.968 |
| v5 | 0 | 0.40 | −2.68 | −4.04 | 0.938 | 0.984 |
| v6 | 0 | 0.31 | −3.51 | −5.12 | 0.923 | 1.000 |

zret's reference is v1 to v4 (mean 0.690, sd 0.108); the top-3 reference is v1 to v3 (mean 0.737, sd 0.083). The
model keeps the z column and adds the other three.

**How the columns enter.** The 322 columns enter l1's pre-activation through a2_w, a 128 × 322 matrix that is zero
at the start. rrf's z_top enters through a scalar a2_bw, also zero at the start:

    pre = l1([zret's inputs]) + a2_w · [A2's columns]
    s   = base_w · z(rrf) + a2_bw · z_top(rrf) + out(h)

That is 41,217 new weights. Creating them draws no random number, so every other weight starts as the base's.
Each arm starts as its base bit for bit: its scores, and every shared weight's gradient.

**Why l1 and not a pointwise head.** The audit's section 2 sends a new input through a pointwise head (R1's
template). These columns are the same kind as zret's z-scores, which enter l1. ztop50's and prank's gains came
through l1, where the columns mix with each block's raw values. A separate head over the 322 columns alone would
lose that mixing. Both models still read the same input in the same place.

**The MLP rule.** A2 reads no score and normalises nothing over the question's scores. z_top, hi and lo are
within-pool statistics of input columns, the kind zret's z-scores already are, and rrf is a first-stage input both
models read. The rule of 8 and 14 October holds.

## 3. The arms

| arm | model | carve |
| --- | --- | --- |
| zrckpa | zrckp's model (`zp1c3p.ZRCKP`) with A2Mix just above zret's forward (MRO: ZRCKP, KTMixP, ZRM, ChainMatch, A2Mix, ZRet) | zrckp's (`zp1c3.KTChainCarveZRC`) |
| zspka | zspk's model (`zp1c3.ZSPK`) with A2Mix in the same place (MRO: ZSPK, ZProp, KTMix, ZRM, ChainMatch, A2Mix, ZRet); zsp's propagation reads the score A2 changes | zspk's (`zp1c3.KTLinkCarveBase`) |

The same input goes to both models (F1's one-input-set rule). No build is needed: the columns are computed in each
batch from the cache's own rows.

## 4. The fits and the call

- **Fits:** one card fit per arm on L-musique and one on L-hotpotqa (`scr-a2-<arm>`, `-hp`), on P1c2's roots with
  P1c3's KT input. These are the card fits' settings: variant p, config 2e-3:1e-4:0.1:8:2, seed 0, hidden 128.
- **Reads:** each fit on the six s1eval carves at p@swa.
- **Compares:** each fit against its base arm's card fit of the same split (zrckp: scr-p1c3p-zrckp, -hp; zspk:
  scr-p1c3-zspk, -hp), which decides. Step 1's fit is reported beside. The re-call's base screens are L1's
  (`outputs/l1/base-<base arm>-<split>.json`).
- **Call:** relz's pair and its re-call under the seed null (the four null files of docs/SCREENS.md), per arm. GAIN,
  LOSS and WITHIN are docs/SCREENS.md's, on R@5.

### The rule, fixed before any number

Per arm:
- **ADOPT_FOR_FULL_RUN:** its re-call has no LOSS on any of its twelve reads and at least one GAIN;
- **NOT_ADOPTED:** otherwise;
- **INCOMPLETE:** its re-call is missing, is not A2's re-call of that arm, or does not hold the twelve reads.

The verdict closes only this form (SCREENS.md section 3b), and it names the next form (section 5).

**Reported beside, deciding nothing:**
- metaqa in domain, per split and arm. This is the read the earlier forms lost; if A2 holds it while musique gains,
  "beside" is what the earlier forms lacked;
- musique, both splits and arms. The signal was musique zero-shot (L-musique); musique in domain (L-hotpotqa) is
  Claim 1's read;
- webqsp (zero-shot in both splits);
- hit@1 and FC@5 per read, as the compares give them.

**Flagged.**
- **Serving cost.** Per question there are two sorts of the pool per fixed column (86 columns) and z-scores against a
  50-row reference. On a 2,000-row pool that is about 86 × 2 × 2,000 × 11 ≈ 3.8 million comparisons, and l1 grows from
  309 to 631 inputs. Both models pay it. It is timed cold beside P4's timings before any latency claim, split into
  index, query-local compile and forward.
- **Parameters.** zrckp gains 41,217 weights, which roughly doubles l1. Both models gain the same.

## 5. What each outcome leads to (fixed now)

Per arm, the next form, each in its own file declared before its numbers:

| its re-call | the next form |
| --- | --- |
| ADOPT_FOR_FULL_RUN | the arm fitted again on P1g-b's pools, then its full run |
| NOT_ADOPTED, with at least one GAIN and a LOSS | the back-off form, A2g: A2's two terms (a2_w's and a2_bw's) scaled by 2σ(⟨w, c⟩ + b), w and b zero so the scale starts at 1, with c the pool's log size and retrieved share (input columns only). The model can then turn A2 down on the pools where it misleads |
| NOT_ADOPTED, with no GAIN | the halves apart: the rank columns alone beside, and the top-50 columns alone beside, each on both models |
| INCOMPLETE | the missing re-call is run; nothing else changes |

- **If both arms pass**, the input is the lean pair's, read by both.
- **If only one passes**, its full run is declared. The next file decides whether F1's one-input-set rule keeps the
  input.
- **What does not happen.** No model change is adopted on today's pools (the user's rule of 13 October).

## 6. How it runs

- **Self-test.** `python outputs/mp_unified/za2rank.py --selftest` passes before this commit (3.9 s). It checks:
  - the arms' MROs, with A2Mix just above zret's forward in both models, and the base arms unchanged;
  - the columns against references by hand: prank's ranks (`lean_screen5.rank_numpy`) and z-scores against
    ztop50's reference (`lean_screen4.top_ref_numpy`) in float64, at k = 50 and k = 3. At k = 50 on pools under 50
    retrieved rows, z_top equals zret's z; at k = 3 it differs exactly where a pool has more than three. A question
    without SEMB gets zero SEMB columns. All columns are finite;
  - each arm starts as its base bit for bit: its scores in eval and train mode, and every shared weight's gradient.
    Building the model draws no random number. a2_w has no gradient at the start, as l1 has none (out starts at
    zero); a2_bw has one;
  - with A2's weights moved, the scorer is zret's on A2's inputs, and zspka's propagation reads that score;
  - toy fits through lean_gpu's loop repeat bit for bit, and A2's weights learn;
  - train refuses another arm, split, fit name, cache root or device; read refuses a fit that is not A2's or not its
    split's;
  - relz's pair and re-call per arm, stamped A2's;
  - the verdict's outcomes, and the next form each one names.
- **Host items (17, `a2-*`)**, after P5c's verdict item:
  - four trains: `cuda_alloc --frac 0.32`, gpu share 0.34;
  - four reads at the same share;
  - four compares, two pairs, two re-calls, the verdict.
  The card fits peaked at 3.7 GB (L-musique) and 5.55 GB (L-hotpotqa) in training and 5.3 GB in reading,
  whatever the arm. A2 adds about 0.7 GB to a 262,144-row read batch: 322 float32 columns, plus the sort's scratch,
  16 columns at a time. The trains have no build to wait on, so they start when the card has room.
- **Disk:** about 0.3 GB of fits and reads. A2 reads P1c2's caches, so the clean-up after P1g-b's verdict keeps them
  until A2's and P5c's reads are done.
- **Verdict:** about 16:00 to 18:00 on 17 October, depending on the card.

## 7. Results

Filed 17 October 2026 from the host's records (`outputs/a2/screen-verdict.json`, written 2026-10-17T14:05:36Z; the
re-calls `outputs/screen/scr-a2-zrckpa-pair-recall.json` and `-zspka-`, host paths made relative). **Both arms
NOT_ADOPTED, with no GAIN.**

- **zrckpa: 0 GAIN, 1 LOSS.** hotpotqa in domain falls 0.0083 on L-musique.
- **zspka: 0 GAIN, 2 LOSS.** 2wiki in domain falls 0.0098 and webqsp zero-shot 0.0112, both on L-musique.
- **musique zero-shot rises on both** (+0.0104 and +0.0249, L-musique), filed GAIN, but re-called WITHIN that read's
  floor (0.072). That is the read ztop50 and prank gained on (+0.139, +0.202 on step 1's model; section 1). Beside
  the z-scores, metaqa no longer pays (−0.0046 at most, within its floor on every read), but most of musique's gain is gone,
  and the in-domain passage graphs pay a little on both models.
- **Beside:** musique in domain (L-hotpotqa) −0.0050 (MLP) and +0.0016 (GNN), both WITHIN; webqsp zero-shot falls on
  three of four reads (within the floor on all but zspka's L-musique read).

This closes only this form, the two halves together. The next form, fixed in section 5 for no GAIN, is **the halves
apart**: the rank columns alone beside, and the top-50 columns alone beside, each on both models, in their own declared
file.

Each arm's twelve reads after the seed-null re-call (delta = the arm's R@5 minus its base's; base R@5 shown as zrckp's /
zspk's):

| split | dataset | read | base R@5 | zrckpa: delta [95% CI], call | zspka: delta [95% CI], call |
| --- | --- | --- | --- | --- | --- |
| L-musique | metaqa | in-domain | 0.8019 / 0.8009 | -0.0039 [-0.0048, -0.0030], WITHIN | -0.0002 [-0.0013, +0.0008], WITHIN |
| L-musique | squad | in-domain | 0.9116 / 0.9114 | -0.0008 [-0.0028, +0.0012], WITHIN | +0.0011 [-0.0008, +0.0029], WITHIN |
| L-musique | musique | zero-shot | 0.5121 / 0.5012 | +0.0104 [+0.0048, +0.0159], WITHIN (filed GAIN) | +0.0249 [+0.0191, +0.0312], WITHIN (filed GAIN) |
| L-musique | hotpotqa | in-domain | 0.9065 / 0.9080 | -0.0083 [-0.0109, -0.0058], **LOSS** | -0.0050 [-0.0073, -0.0026], WITHIN |
| L-musique | 2wiki | in-domain | 0.9347 / 0.9383 | -0.0054 [-0.0070, -0.0038], WITHIN | -0.0098 [-0.0113, -0.0082], **LOSS** |
| L-musique | webqsp | zero-shot | 0.3627 / 0.3229 | -0.0097 [-0.0197, -0.0004], WITHIN (filed LOSS) | -0.0112 [-0.0231, -0.0004], **LOSS** |
| L-hotpotqa | metaqa | in-domain | 0.8044 / 0.8040 | -0.0042 [-0.0052, -0.0032], WITHIN | -0.0046 [-0.0057, -0.0036], WITHIN |
| L-hotpotqa | squad | in-domain | 0.9127 / 0.9135 | -0.0008 [-0.0027, +0.0010], WITHIN | -0.0009 [-0.0028, +0.0009], WITHIN |
| L-hotpotqa | musique | in-domain | 0.5745 / 0.5754 | -0.0050 [-0.0103, +0.0001], WITHIN | +0.0016 [-0.0036, +0.0066], WITHIN |
| L-hotpotqa | hotpotqa | zero-shot | 0.8513 / 0.8499 | +0.0055 [+0.0023, +0.0089], WITHIN | -0.0001 [-0.0031, +0.0030], WITHIN |
| L-hotpotqa | 2wiki | in-domain | 0.9350 / 0.9333 | -0.0060 [-0.0077, -0.0043], WITHIN | -0.0047 [-0.0064, -0.0031], WITHIN |
| L-hotpotqa | webqsp | zero-shot | 0.3475 / 0.2680 | -0.0129 [-0.0239, -0.0018], WITHIN (filed LOSS) | +0.0012 [-0.0088, +0.0113], WITHIN |
