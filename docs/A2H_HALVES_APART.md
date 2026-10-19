# A2h: A2's two halves apart, each beside the z-scores, on both models

Declared 18 October 2026, before any number. The next form A2's verdict named for both arms (no GAIN;
docs/A2_RANKS_BESIDE_Z.md, sections 5 and 7), under docs/SCREENS.md section 3b and docs/VERDICT_AUDIT_2026_10_17.md.
Code: `outputs/mp_unified/za2h.py`. Filed 19 October: all four arms NOT_ADOPTED (section 7).

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

## 7. Results

Filed 19 October 2026 from the host's records (`outputs/a2h/screen-verdict.json`, written 2026-10-19T03:13:31Z; the
re-calls `outputs/screen/scr-a2h-zrckpi-pair-recall.json`, `-zspki-`, `-zrckpm-` and `-zspkm-`, host paths made
relative). **All four arms NOT_ADOPTED:** neither half gains on either model past the seed null, and three of the four
arms have a LOSS.

- **zrckpi** (the rank half, MLP): **0 GAIN, 1 LOSS.** webqsp zero-shot falls 0.0142 on L-musique [-0.0248, -0.0034].
  musique zero-shot rises +0.0128 [+0.0080, +0.0178], filed GAIN and re-called WITHIN against that read's 0.072 floor;
  webqsp zero-shot on L-hotpotqa rises +0.0155 (filed GAIN, re-called WITHIN, floor 0.0233).
- **zspki** (the rank half, GNN): **0 GAIN, 1 LOSS.** webqsp zero-shot falls 0.0137 on L-musique [-0.0244, -0.0035].
  musique zero-shot rises +0.0283 [+0.0221, +0.0346] (filed GAIN, re-called WITHIN).
- **zrckpm** (the top-50 half, MLP): **0 GAIN, 0 LOSS.** musique zero-shot rises +0.0198 [+0.0144, +0.0253] (filed GAIN,
  re-called WITHIN); webqsp zero-shot on L-hotpotqa falls 0.0144 (filed LOSS, re-called WITHIN); metaqa and hotpotqa in
  domain on L-musique fall 0.0053 and 0.0061, WITHIN.
- **zspkm** (the top-50 half, GNN): **0 GAIN, 1 LOSS.** 2wiki in domain falls 0.0083 on L-musique [-0.0099, -0.0067].
  musique zero-shot rises +0.0275 [+0.0214, +0.0342] (filed GAIN, re-called WITHIN); webqsp zero-shot on L-hotpotqa
  falls 0.0125 (filed LOSS, re-called WITHIN).
- **Beside:** musique in domain (L-hotpotqa, Claim 1's read) moves -0.0017, +0.0037, -0.0036 and +0.0014, all WITHIN.
  A2, with both halves together, moved musique zero-shot +0.0104 (MLP) and +0.0249 (GNN). Each half alone moves it as
  far as both together did, or further (the MLP's top-50 half about twice as far): the halves do not add up.

Each half moves the read it was meant to move (musique zero-shot on L-musique) in the direction prank and ztop50 moved
it on step 1's model, but by a fifth as much or less (+0.202 and +0.139 there), within that read's floor. The rank half
also costs webqsp zero-shot on L-musique on both models. This closes only these forms (each half beside zret's
z-scores). The next forms, fixed in section 5 and recorded in the verdict, are each half's own replacing form, on both
models, each in its own declared file:
- **zrckpi, zspki: A2xi**, prank's form: each fixed block's z-score replaced by the rank pair hi and lo;
- **zrckpm, zspkm: A2xm**, ztop50's form: zret's reference cut to the pool's top 50 retrieved rows.

Each arm's twelve reads after the seed-null re-call (delta = the arm's R@5 minus its base's; a call is shown with its
filed call where the re-call changed it; base R@5 shown as zrckp's / zspk's; the rank half first, then the top-50 half):

| split | dataset | read | base R@5 | zrckpi: delta [95% CI], call | zspki: delta [95% CI], call |
| --- | --- | --- | --- | --- | --- |
| L-musique | metaqa | in-domain | 0.8019 / 0.8009 | +0.0001 [-0.0008, +0.0009], WITHIN | +0.0010 [+0.0001, +0.0020], WITHIN |
| L-musique | squad | in-domain | 0.9116 / 0.9114 | +0.0000 [-0.0019, +0.0020], WITHIN | -0.0008 [-0.0027, +0.0011], WITHIN |
| L-musique | musique | zero-shot | 0.5121 / 0.5012 | +0.0128 [+0.0080, +0.0178], WITHIN (filed GAIN) | +0.0283 [+0.0221, +0.0346], WITHIN (filed GAIN) |
| L-musique | hotpotqa | in-domain | 0.9065 / 0.9080 | -0.0039 [-0.0061, -0.0016], WITHIN | -0.0020 [-0.0041, +0.0003], WITHIN |
| L-musique | 2wiki | in-domain | 0.9347 / 0.9383 | +0.0018 [+0.0004, +0.0032], WITHIN | -0.0030 [-0.0043, -0.0016], WITHIN |
| L-musique | webqsp | zero-shot | 0.3627 / 0.3229 | -0.0142 [-0.0248, -0.0034], **LOSS** | -0.0137 [-0.0244, -0.0035], **LOSS** |
| L-hotpotqa | metaqa | in-domain | 0.8044 / 0.8040 | -0.0015 [-0.0024, -0.0005], WITHIN | -0.0010 [-0.0018, -0.0002], WITHIN |
| L-hotpotqa | squad | in-domain | 0.9127 / 0.9135 | -0.0010 [-0.0029, +0.0009], WITHIN | -0.0007 [-0.0025, +0.0013], WITHIN |
| L-hotpotqa | musique | in-domain | 0.5745 / 0.5754 | -0.0017 [-0.0067, +0.0033], WITHIN | +0.0037 [-0.0014, +0.0089], WITHIN |
| L-hotpotqa | hotpotqa | zero-shot | 0.8513 / 0.8499 | -0.0030 [-0.0060, +0.0001], WITHIN | -0.0019 [-0.0047, +0.0009], WITHIN |
| L-hotpotqa | 2wiki | in-domain | 0.9350 / 0.9333 | -0.0027 [-0.0043, -0.0011], WITHIN | -0.0019 [-0.0035, -0.0003], WITHIN |
| L-hotpotqa | webqsp | zero-shot | 0.3475 / 0.2680 | +0.0155 [+0.0046, +0.0261], WITHIN (filed GAIN) | +0.0035 [-0.0056, +0.0126], WITHIN |

| split | dataset | read | base R@5 | zrckpm: delta [95% CI], call | zspkm: delta [95% CI], call |
| --- | --- | --- | --- | --- | --- |
| L-musique | metaqa | in-domain | 0.8019 / 0.8009 | -0.0053 [-0.0063, -0.0043], WITHIN | -0.0000 [-0.0011, +0.0010], WITHIN |
| L-musique | squad | in-domain | 0.9116 / 0.9114 | -0.0008 [-0.0025, +0.0010], WITHIN | +0.0007 [-0.0012, +0.0025], WITHIN |
| L-musique | musique | zero-shot | 0.5121 / 0.5012 | +0.0198 [+0.0144, +0.0253], WITHIN (filed GAIN) | +0.0275 [+0.0214, +0.0342], WITHIN (filed GAIN) |
| L-musique | hotpotqa | in-domain | 0.9065 / 0.9080 | -0.0061 [-0.0084, -0.0038], WITHIN | -0.0041 [-0.0065, -0.0019], WITHIN |
| L-musique | 2wiki | in-domain | 0.9347 / 0.9383 | -0.0031 [-0.0048, -0.0015], WITHIN | -0.0083 [-0.0099, -0.0067], **LOSS** |
| L-musique | webqsp | zero-shot | 0.3627 / 0.3229 | +0.0046 [-0.0055, +0.0149], WITHIN | -0.0093 [-0.0193, +0.0010], WITHIN |
| L-hotpotqa | metaqa | in-domain | 0.8044 / 0.8040 | -0.0039 [-0.0050, -0.0028], WITHIN | -0.0035 [-0.0045, -0.0024], WITHIN |
| L-hotpotqa | squad | in-domain | 0.9127 / 0.9135 | -0.0002 [-0.0020, +0.0017], WITHIN | +0.0008 [-0.0013, +0.0027], WITHIN |
| L-hotpotqa | musique | in-domain | 0.5745 / 0.5754 | -0.0036 [-0.0086, +0.0013], WITHIN | +0.0014 [-0.0040, +0.0065], WITHIN |
| L-hotpotqa | hotpotqa | zero-shot | 0.8513 / 0.8499 | +0.0013 [-0.0019, +0.0046], WITHIN | -0.0005 [-0.0033, +0.0026], WITHIN |
| L-hotpotqa | 2wiki | in-domain | 0.9350 / 0.9333 | -0.0050 [-0.0066, -0.0033], WITHIN | +0.0003 [-0.0013, +0.0019], WITHIN |
| L-hotpotqa | webqsp | zero-shot | 0.3475 / 0.2680 | -0.0144 [-0.0254, -0.0032], WITHIN (filed LOSS) | -0.0125 [-0.0226, -0.0018], WITHIN (filed LOSS) |

Beside, deciding nothing: each read's FC@5 and hit@1 deltas against the same base, from the compare screens
(`outputs/screen/scr-a2h-zrckpi.json` and the others, `-hp` for L-hotpotqa); and zrckpa's R@5 delta (A2, its own
re-call) and zspka's R@5 delta (A2, its own re-call), on the same read:

| split | dataset | zrckpi FC@5 | zrckpi hit@1 | zspki FC@5 | zspki hit@1 | zrckpa R@5 (A2) | zspka R@5 (A2) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| L-musique | metaqa | +0.0006 | -0.0035 | +0.0014 | +0.0075 | -0.0039 | -0.0002 |
| L-musique | squad | +0.0000 | -0.0035 | -0.0008 | -0.0012 | -0.0008 | +0.0011 |
| L-musique | musique | +0.0099 | +0.0070 | +0.0265 | +0.0331 | +0.0104 | +0.0249 |
| L-musique | hotpotqa | -0.0057 | -0.0185 | -0.0043 | +0.0008 | -0.0083 LOSS | -0.0050 |
| L-musique | 2wiki | +0.0058 | -0.0038 | -0.0080 | -0.0017 | -0.0054 | -0.0098 LOSS |
| L-musique | webqsp | -0.0200 | -0.0020 | -0.0093 | -0.0113 | -0.0097 | -0.0112 LOSS |
| L-hotpotqa | metaqa | -0.0024 | -0.0096 | -0.0009 | -0.0029 | -0.0042 | -0.0046 |
| L-hotpotqa | squad | -0.0010 | -0.0003 | -0.0007 | -0.0001 | -0.0008 | -0.0009 |
| L-hotpotqa | musique | +0.0004 | +0.0000 | +0.0037 | -0.0074 | -0.0050 | +0.0016 |
| L-hotpotqa | hotpotqa | -0.0038 | +0.0035 | -0.0059 | +0.0163 | +0.0055 | -0.0001 |
| L-hotpotqa | 2wiki | -0.0094 | +0.0045 | -0.0033 | +0.0002 | -0.0060 | -0.0047 |
| L-hotpotqa | webqsp | +0.0093 | +0.0053 | +0.0040 | -0.0040 | -0.0129 | +0.0012 |

| split | dataset | zrckpm FC@5 | zrckpm hit@1 | zspkm FC@5 | zspkm hit@1 | zrckpa R@5 (A2) | zspka R@5 (A2) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| L-musique | metaqa | -0.0041 | -0.0215 | +0.0011 | +0.0046 | -0.0039 | -0.0002 |
| L-musique | squad | -0.0008 | -0.0026 | +0.0007 | -0.0013 | -0.0008 | +0.0011 |
| L-musique | musique | +0.0207 | +0.0228 | +0.0290 | +0.0211 | +0.0104 | +0.0249 |
| L-musique | hotpotqa | -0.0104 | -0.0149 | -0.0097 | -0.0122 | -0.0083 LOSS | -0.0050 |
| L-musique | 2wiki | -0.0083 | +0.0036 | -0.0208 | +0.0024 | -0.0054 | -0.0098 LOSS |
| L-musique | webqsp | -0.0013 | -0.0007 | -0.0093 | -0.0133 | -0.0097 | -0.0112 LOSS |
| L-hotpotqa | metaqa | -0.0046 | -0.0142 | -0.0029 | -0.0137 | -0.0042 | -0.0046 |
| L-hotpotqa | squad | -0.0002 | -0.0003 | +0.0008 | +0.0015 | -0.0008 | -0.0009 |
| L-hotpotqa | musique | -0.0004 | -0.0095 | -0.0037 | -0.0095 | -0.0050 | +0.0016 |
| L-hotpotqa | hotpotqa | +0.0038 | -0.0182 | +0.0024 | -0.0116 | +0.0055 | -0.0001 |
| L-hotpotqa | 2wiki | -0.0139 | +0.0071 | +0.0030 | -0.0009 | -0.0060 | -0.0047 |
| L-hotpotqa | webqsp | -0.0113 | -0.0040 | -0.0086 | -0.0160 | -0.0129 | +0.0012 |
