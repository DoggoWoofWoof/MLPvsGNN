# A2h: A2's two halves apart, each beside the z-scores, on both models

Declared 18 October 2026, before any number. The next form A2's verdict named for both arms (no GAIN;
docs/A2_RANKS_BESIDE_Z.md, sections 5 and 7), under docs/SCREENS.md section 3b and docs/VERDICT_AUDIT_2026_10_17.md.
Code: `outputs/mp_unified/za2h.py`.

## 1. Why this, now

Two input forms gained musique zero-shot on step 1's model and were dropped, because each **replaced** zret's
within-pool z-scores and metaqa in domain paid:

| screen | what it did | musique zero-shot (L-musique) |
| --- | --- | --- |
| ztop50 | every z-score against the pool's 50 best retrieved rows by rrf | +0.1387, GAIN |
| prank | every fixed block's z-score replaced by two rank columns | +0.2016, GAIN |

**A2** (17 October) put both beside the z-scores, together, on today's lean pair. Both arms were NOT_ADOPTED with no
GAIN:
- **musique zero-shot** rose within its floor (MLP +0.0104, GNN +0.0249);
- **metaqa** no longer paid;
- **the in-domain passage graphs** paid a little (hotpotqa −0.0083 on the MLP; 2wiki −0.0098 and webqsp zero-shot
  −0.0112 on the GNN).

That closes only A2's form, the two halves together. Each half reads a different thing:
- **a rank** is pool-size free;
- **a top-50 z-score** sits on a fixed-size reference.

Together they put 322 columns into l1 beside its 309. A2's section 5 fixed the halves apart as the next form for no
GAIN, each on both models. One half may carry the signal while the other carries the cost.

## 2. The form

Nothing zret reads changes. Each half is exactly its columns of A2's input (section 2 of A2's file), computed in each
batch from the cache's own rows; no build.

| half | columns added (each times the block's keep) | base term |
| --- | --- | --- |
| rank | per fixed column: hi and lo, prank's 60 / (60 + min(rank, 50)) from the pool's top and bottom (ties share the best position); SEMB none. 172 columns on the card fits (86 fixed columns) | none |
| top-50 | per column of every block, fixed and SEMB: z_top, the z-score against the pool's 50 best retrieved rows by rrf (rrf > 0, ties by row order; a pool with fewer than two such rows, or a column constant over them, takes the whole pool's z-score, as zret does). 150 columns (86 + 64) | rrf's z_top, through a2m_bw |

**How they enter.** As in A2:

    rank half:    pre = l1([zret's inputs]) + a2i_w · [the rank columns]
                  s   = base_w · z(rrf) + out(h)
    top-50 half:  pre = l1([zret's inputs]) + a2m_w · [the top-50 columns]
                  s   = base_w · z(rrf) + a2m_bw · z_top(rrf) + out(h)

Every added weight is zero at the start, and creating them draws no random number. Each arm starts as its base bit for
bit: its scores and every shared weight's gradient.

| half | added weights on the card fits |
| --- | --- |
| rank | a2i_w: 128 × 172 = 22,016 |
| top-50 | a2m_w: 128 × 150 = 19,200, plus a2m_bw (1) |

**The MLP rule.** No score read, nothing normalised over the question's scores. Ranks and top-50 z-scores are
within-pool statistics of input columns, the kind zret's z-scores already are.

## 3. The arms

| arm | model | carve |
| --- | --- | --- |
| zrckpi | zrckp's model with the rank half (MRO: ZRCKP, KTMixP, ZRM, ChainMatch, A2IMix, ZRet) | zrckp's (`zp1c3.KTChainCarveZRC`) |
| zspki | zspk's model with the rank half (MRO: ZSPK, ZProp, KTMix, ZRM, ChainMatch, A2IMix, ZRet); zsp's propagation reads the score it changes | zspk's (`zp1c3.KTLinkCarveBase`) |
| zrckpm | zrckp's model with the top-50 half (MRO: ZRCKP, KTMixP, ZRM, ChainMatch, A2MMix, ZRet) | zrckp's |
| zspkm | zspk's model with the top-50 half (MRO: ZSPK, ZProp, KTMix, ZRM, ChainMatch, A2MMix, ZRet) | zspk's |

## 4. The fits and the call

A2's, unchanged, per arm:
- **Fits:** one card fit on L-musique and one on L-hotpotqa (`scr-a2h-<arm>`, `-hp`), on P1c2's roots with P1c3's KT
  input, at the card fits' settings (variant p, config 2e-3:1e-4:0.1:8:2, seed 0, hidden 128).
- **Reads:** each fit on the six s1eval carves at p@swa.
- **Compares:** each fit against its base arm's card fit of the same split (zrckp: scr-p1c3p-zrckp and -hp; zspk:
  scr-p1c3-zspk and -hp), with step 1's fit beside. The re-call's base screens are L1's.
- **Call:** relz's pair and its re-call under the seed null (the four null files of docs/SCREENS.md).

### The rule, fixed before any number

Per arm:
- **ADOPT_FOR_FULL_RUN:** its re-call has no LOSS on any of its twelve reads and at least one GAIN;
- **NOT_ADOPTED:** otherwise;
- **INCOMPLETE:** its re-call is missing, is not A2h's re-call of that arm, or does not hold the twelve reads.

The verdict closes only this form (SCREENS.md section 3b), and it names each arm's next form (section 5).

**Reported beside, deciding nothing:**
- musique (the read ztop50 and prank gained on, and Claim 1's read in domain), with the four arms side by side;
- metaqa (the read the replacing forms lost) and webqsp;
- hit@1 and FC@5 per read;
- A2's deltas, set beside each read when the results are filed.

**Serving cost.** Both halves are flagged, to be timed cold before any latency claim:
- the rank half pays A2's two sorts per fixed column;
- the top-50 half pays one top-50 reference per question and one z-score per column.

## 5. What each outcome leads to (fixed now)

| its re-call | rank half (zrckpi, zspki) | top-50 half (zrckpm, zspkm) |
| --- | --- | --- |
| ADOPT_FOR_FULL_RUN | the arm fitted again on P1g-b's pools, then its full run | the same |
| NOT_ADOPTED, with at least one GAIN and a LOSS | A2gi: the half's term scaled by 2σ(⟨w, c⟩ + b), w and b zero so the scale starts at 1, c the pool's log size and retrieved share (input columns only), on both models | A2gm: the same back-off on the top-50 half's terms |
| NOT_ADOPTED, with no GAIN | A2xi: prank's replacing form on the lean pair (each fixed block's z-score replaced by hi and lo), on both models | A2xm: ztop50's replacing form on the lean pair (zret's reference cut to the pool's top 50 retrieved rows), on both models |
| INCOMPLETE | the missing re-call is run; nothing else changes | the same |

**Why the replacing forms come next on no GAIN.** They carried the gains (+0.20 and +0.14), and they never ran on
today's pair. Beside the z-scores, an in-domain fit can lean on the z-scores and leave the new columns. Replacing
tests the signal where the model must read it, and any metaqa loss then goes to the back-off form.

No model change is adopted on today's pools (the user's rule of 13 October): an adopt is refitted on P1g-b's pools
before its full run.

## 6. How it runs

**Self-test.** `python outputs/mp_unified/za2h.py --selftest` passes before this commit (6.0 s). On rmatch's toy carves
(chains, links), it checks:
- **the arms:** the four MROs, with each half just above zret's forward; the base arms and A2's arms unchanged; no arm
  is an A2Mix;
- **the columns:** each half is its columns of A2's input bit for bit, from za2rank's A2Mix at k = 50 and k = 3. A2's
  own columns are checked there by hand against prank's ranks and ztop50's z-scores. zret's l1 input and rrf's z are
  A2's; the top half's base term is A2's second. A question without SEMB has no top-50 SEMB columns;
- **the start:** each arm is its base bit for bit, with no random number drawn: scores in eval and training, and every
  shared weight's gradient. The half's matrix has no gradient at the start, as l1 has none (out is zero); a2m_bw has
  one;
- **moved:** with the half's weights and zret's output layer set, the scorer is zret's on the half's inputs, and the
  GNN's propagation reads that score;
- **toy fits** through lean_gpu's loop repeat bit for bit, and the halves' weights learn;
- **the refusals:** train refuses another arm, split, fit name, cache root or device; read refuses a fit that is not
  A2h's, not its arm's or not its split's;
- relz's pair and re-call per arm, stamped A2h's; the verdict's outcomes and each half's next forms.

**Host items** (33, `a2h-*`), after A6e's:
- **trains:** 8, with no build to wait on, so they start when the card has room. They run at A4's fractions (L-musique
  0.20, L-hotpotqa 0.27, GPU share +0.02). A2's trains peaked at 3.7 and 5.55 GB, the same as A4's, and each half is
  about half of A2's columns;
- **reads:** 8 at 0.27, share 0.29 (A2's reads peaked at 5.86 GB);
- **the call:** 8 compares, 4 pairs, 4 re-calls and the verdict;
- **memory:** as A2's items.

**Disk:** about 0.6 GB of fits and reads.
